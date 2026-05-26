"""
FAG-VLA Evaluation Script

Runs pi0.5 on LIBERO tasks with optional FAG-enhanced instruction rewriting.
Designed to work both in simulation (LIBERO MuJoCo) and with real robots
(real-robot interface stub provided for future hardware integration).

Modes
-----
  --mode baseline   : pure pi0.5, no FAG intervention (records attention for analysis)
  --mode fag_offline: post-hoc FAG analysis on saved attention data
  --mode fag_online : live FAG rewriting during rollout (requires --rewrite)

Output
------
  Per-episode JSON results in FAG_BASE_DIR/data/outputs/
  Attention data in FAG_BASE_DIR/data/attention_data/
  Console: success rate table comparing baseline vs. FAG-rewritten

Usage
-----
  python fag_eval.py \\
      --mode fag_online \\
      --tasks libero_object \\
      --task_ids "[0,1,2]" \\
      --n_episodes 5 \\
      --rewrite_threshold 0.25
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Ensure the FAG-VLA src is on the path
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import fag_vla.settings as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Real-robot interface stub
# (Replace with actual hardware SDK when robot arm arrives)
# ---------------------------------------------------------------------------

def _make_real_robot_env(task_name: str, can_port: str = "can0", max_steps: int = 200):
    """
    Create and connect a real PIPER6DOF + D435i environment.
    Returns a PiperD435iInterface instance ready to use as a VectorEnv.
    """
    from fag_vla.piper_d435i_interface import PiperD435iInterface
    env = PiperD435iInterface(
        task_name=task_name,
        can_port=can_port,
        max_steps=max_steps,
    )
    env.connect()
    return env


# ---------------------------------------------------------------------------
# FAG online intervention wrapper
# ---------------------------------------------------------------------------

class FAGOnlineWrapper:
    """
    Wraps the FAG pipeline for online instruction rewriting during rollout.

    Three-layer trigger logic based on LVCI trend:
      Layer 1: pre-emptive check — high VTI at step 0 → light proactive rewrite
      Layer 2: trend trigger (main) — ΔLVCI > threshold → rewrite
      Layer 3: VTI confirmation — determines rewrite strength (light/medium/strong)
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.step_results = []
        self._true_original: Optional[str] = None
        self._lvci_history: list = []         # LVCI values per scored step
        self._prior_check_done: bool = False  # Layer 1 fired at most once per episode
        self._cooldown_remaining: int = 0     # checkpoints to skip after a rewrite

    def reset_episode(self):
        """Call at the start of each episode to clear state."""
        self._true_original = None
        self.step_results = []
        self._lvci_history = []
        self._prior_check_done = False
        self._cooldown_remaining = 0

    def _compute_trend(self) -> float:
        """
        ΔLVCI = mean(second half) - mean(first half) of LVCI history.
        Positive trend → LVCI rising → model stuck → trigger.
        Returns 0.0 if fewer than 2 data points.
        """
        n = len(self._lvci_history)
        if n < 2:
            return 0.0
        mid = n // 2
        early = sum(self._lvci_history[:mid]) / max(mid, 1)
        late  = sum(self._lvci_history[mid:]) / max(n - mid, 1)
        return late - early

    def _decide_rewrite_strength(self, vti_signal: float) -> str:
        """Layer 3: VTI-based rewrite strength decision."""
        if vti_signal >= cfg.VTI_CONFIRM_THRESHOLD + 0.10:   # > 0.75
            return "strong"
        elif vti_signal >= cfg.VTI_CONFIRM_THRESHOLD:         # 0.65–0.75
            return "medium"
        return "light"

    @staticmethod
    def _normalize_attn(layer_dict: dict) -> dict:
        """Convert compact {"agg": tensor} format → {0: tensor} for FAG pipeline."""
        if not layer_dict:
            return layer_dict
        if "agg" in layer_dict:
            return {0: layer_dict["agg"]}
        return layer_dict

    def maybe_rewrite(
        self,
        step: int,
        tracer,
        current_instruction: str,
    ) -> str:
        """
        Three-layer trigger logic.

        Layer 1 (step 0 only): VTI prior check — if VTI_0 > VTI_PRIOR_THRESHOLD,
                               apply a light proactive rewrite immediately.
        Layer 2 (main):        LVCI trend trigger — ΔLVCI > LVCI_TREND_THRESHOLD
                               after at least 2 scored checkpoints.
        Layer 3 (on trigger):  VTI confirmation — determines rewrite strength.

        Returns (possibly rewritten) instruction string.
        """
        if self._true_original is None:
            self._true_original = current_instruction

        vlm_attn = self._normalize_attn(tracer.language_attn.get(step, {}))
        raw_exp  = tracer.expert_attn.get(step, {})
        exp_attn = {ts: self._normalize_attn(v) for ts, v in raw_exp.items()
                    if isinstance(v, dict)}

        if not vlm_attn:
            return current_instruction

        lang_info = dict(tracer.language_info) if tracer.language_info else {}

        # ---- Score this step (always) ----
        try:
            # Run pipeline without rewriting first to get the score
            _, step_result_scored = self.pipeline.run_online_step(
                step=step,
                vlm_layer_attn=vlm_attn,
                expert_attn_ts=exp_attn,
                current_instruction=current_instruction,
                lang_info=lang_info,
                true_original_instruction=self._true_original,
                rewrite_override=False,   # scoring pass only
            )
        except Exception as e:
            logger.debug(f"Step {step:03d} — FAG scoring error ({e}); keeping instruction.")
            return current_instruction

        lvci = step_result_scored.alignment_score
        vti  = step_result_scored.diagnostics.get("vti_signal", 0.0)
        self._lvci_history.append(lvci)

        # ---- Trigger decision ----
        trend   = self._compute_trend()
        rewrite_override = False
        rewrite_strength = "medium"

        # Decrement cooldown counter
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # Layer 1: prior VTI check (first step only)
        if not self._prior_check_done and step == 0:
            self._prior_check_done = True
            # only pre-empt if LVCI also exceeds floor (avoids false positives when grounded)
            if vti >= cfg.VTI_PRIOR_THRESHOLD and lvci >= cfg.LVCI_FLOOR:
                rewrite_override = True
                rewrite_strength = "light"
                logger.info(
                    f"Step {step:03d} — Layer1 prior: VTI={vti:.3f} "
                    f">= {cfg.VTI_PRIOR_THRESHOLD}, LVCI={lvci:.3f} → pre-emptive light rewrite"
                )

        # Layer 2: trend trigger (needs ≥ 2 checkpoints)
        # dual condition — trend AND absolute LVCI above floor AND not in cooldown
        if not rewrite_override and len(self._lvci_history) >= 2:
            if (trend > cfg.LVCI_TREND_THRESHOLD
                    and lvci >= cfg.LVCI_FLOOR
                    and self._cooldown_remaining <= 0):
                rewrite_override = True
                # Layer 3: VTI-based strength
                rewrite_strength = self._decide_rewrite_strength(vti)
                logger.info(
                    f"Step {step:03d} — Layer2 trend: ΔLVCI={trend:+.3f} "
                    f"> {cfg.LVCI_TREND_THRESHOLD} | LVCI={lvci:.3f} | VTI={vti:.3f} "
                    f"→ {rewrite_strength} rewrite"
                )

        if not rewrite_override:
            self.step_results.append(step_result_scored)
            return current_instruction

        # ---- Rewrite: set cooldown immediately ----
        self._cooldown_remaining = cfg.REWRITE_COOLDOWN_STEPS

        # ---- Rewrite strategy: revert-to-original unless VTI is very high ----
        # When VTI < VTI_STRONG_THRESHOLD, the model just needs re-grounding to the
        # known-good original — no novel LLM rewrite needed.
        # When VTI >= VTI_STRONG_THRESHOLD, the model is severely decoupled and
        # front-loading via LLM may help more than a plain revert.
        if vti < cfg.VTI_STRONG_THRESHOLD:
            new_instr = self._true_original
            self.step_results.append(step_result_scored)
            logger.info(
                f"Step {step:03d} — LVCI={lvci:.3f} trend={trend:+.3f} VTI={vti:.3f} "
                f"→ revert-to-original: '{current_instruction}' → '{new_instr}'"
            )
            return new_instr

        # High VTI path: use LLM front-loading (strong rewrite only)
        rewrite_strength = "strong"
        try:
            new_instr, step_result_rw = self.pipeline.run_online_step(
                step=step,
                vlm_layer_attn=vlm_attn,
                expert_attn_ts=exp_attn,
                current_instruction=current_instruction,
                lang_info=lang_info,
                true_original_instruction=self._true_original,
                rewrite_override=True,
                lvci_trend=trend,
                vti_signal_hint=vti,
                rewrite_strength=rewrite_strength,
            )
        except Exception as e:
            logger.debug(f"Step {step:03d} — FAG rewrite error ({e}); reverting to original.")
            new_instr = self._true_original
            self.step_results.append(step_result_scored)
            return new_instr

        self.step_results.append(step_result_rw)

        if step_result_rw.was_rewritten:
            logger.info(
                f"Step {step:03d} — LVCI={lvci:.3f} trend={trend:+.3f} VTI={vti:.3f} "
                f"→ llm-strong: '{current_instruction}' → '{new_instr}'"
            )
        else:
            # LLM rewrite rejected by OOD validation — fall back to true original
            new_instr = self._true_original
            logger.info(
                f"Step {step:03d} — LLM rewrite OOD-rejected → reverting to original"
            )
        return new_instr


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

class FAGEvaluator:
    def __init__(self, args):
        self.args = args
        cfg.ensure_dirs()
        self.results: Dict[str, List] = {"baseline": [], "fag": []}

    # ------------------------------------------------------------------
    def run(self):
        mode = self.args.mode

        if mode == "fag_offline":
            self._run_offline_analysis()
        elif mode in ("baseline", "fag_online"):
            self._run_rollout_eval(with_fag=(mode == "fag_online"))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # ------------------------------------------------------------------
    def _run_offline_analysis(self):
        """Analyse pre-saved attention data without running simulation."""
        from fag_vla.fag_pipeline import FAGPipeline

        attn_dir = Path(self.args.attn_dir) if self.args.attn_dir else cfg.ATTN_DATA_DIR
        logger.info(f"Running offline FAG analysis on: {attn_dir}")

        pipeline = FAGPipeline(
            tokenizer_path=str(cfg.TOKENIZER_PATH),
            save_graphs=True,
            rewrite_enabled=self.args.rewrite,
        )
        result = pipeline.run_offline(
            attn_data_dir=attn_dir,
            episode_id=0,
            task_name=self.args.tasks,
        )

        logger.info(
            f"Offline analysis complete — "
            f"mean alignment: {result.mean_alignment:.3f}, "
            f"rewrites: {result.rewrite_count}/{len(result.steps)}"
        )
        self._print_step_table(result.steps)

    # ------------------------------------------------------------------
    def _run_rollout_eval(self, with_fag: bool = False):
        """Run rollouts (simulation or real robot) with optional FAG rewriting."""
        _add_lerobot_to_path()

        import torch
        from lerobot.policies.pi05.collector import ATTENTION_TRACER
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.envs.factory import make_env, make_env_pre_post_processors
        from lerobot.envs.utils import preprocess_observation, add_envs_task
        from lerobot.utils.constants import ACTION

        task_ids   = json.loads(self.args.task_ids)
        n_episodes = self.args.n_episodes

        logger.info(f"Loading pi0.5 from: {cfg.POLICY_PATH}")
        device     = "cuda" if torch.cuda.is_available() else "cpu"
        policy_cfg = PreTrainedConfig.from_pretrained(str(cfg.POLICY_PATH), device=device)

        # For real robot, fall back to libero_object for normalization stats
        # (pi0.5 was fine-tuned on LIBERO-Object; the model's distribution reference)
        is_real = getattr(self.args, "real_robot", False)
        norm_suite = "libero_object" if is_real else self.args.tasks
        env_cfg    = LiberoEnvConfig(
            task=norm_suite,
            task_ids=task_ids if not is_real else [0],
        )

        policy = _load_pi05_policy(str(cfg.POLICY_PATH))
        policy.eval()

        # Build preprocessor with dataset normalization stats injected.
        # The checkpoint's policy_preprocessor.json has features:{} (stats missing),
        # so we construct the pipeline directly using make_pi05_pre_post_processors,
        # which embeds the correct MEAN_STD stats for LIBERO.
        dataset_stats = _load_libero_normalization_stats(
            libero_suite=norm_suite,
            device=device,
        )
        preprocessor, postprocessor = _make_pi05_processors_with_stats(
            policy_cfg=policy_cfg,
            dataset_stats=dataset_stats,
            device=device,
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=policy_cfg
        )

        # FAG pipeline
        fag_wrapper = None
        if with_fag:
            from fag_vla.fag_pipeline import FAGPipeline
            pipeline = FAGPipeline(
                tokenizer_path=str(cfg.TOKENIZER_PATH),
                save_graphs=True,
                rewrite_enabled=self.args.rewrite,
            )
            fag_wrapper = FAGOnlineWrapper(pipeline)

        # ------------------------------------------------------------------
        # Build environment(s)
        # ------------------------------------------------------------------
        if getattr(self.args, "real_robot", False):
            # Real robot: iterate over task_ids one by one
            task_name  = getattr(self.args, "real_robot_task",
                                 f"real_robot_task_{task_ids[0]}")
            can_port   = getattr(self.args, "can_port", "can0")
            max_steps  = getattr(self.args, "max_steps", None) or 150
            envs_iter = [
                ("real", tid, _make_real_robot_env(task_name, can_port, max_steps))
                for tid in task_ids
            ]
        else:
            # Simulation: make_env returns {suite_name: {task_id: vec_env}}
            envs_dict = make_env(cfg=env_cfg, n_envs=cfg.EVAL_BATCH_SIZE)
            envs_iter = [
                (suite, tid, vec_env)
                for suite, suite_envs in envs_dict.items()
                for tid, vec_env in suite_envs.items()
            ]

        all_successes = []
        for suite_name, task_id, vec_env in envs_iter:
            for ep_idx in range(n_episodes):
                mode_label = 'fag' if with_fag else 'baseline'
                logger.info(
                    f"Episode {ep_idx+1}/{n_episodes}  suite={suite_name}  "
                    f"task_id={task_id}  mode={mode_label}"
                )
                # Point TRACER to a per-episode directory so episodes don't mix
                ep_attn_dir = cfg.ATTN_DATA_DIR / mode_label / f"task{task_id}_ep{ep_idx}"
                ep_attn_dir.mkdir(parents=True, exist_ok=True)
                ATTENTION_TRACER.set_main_dir(str(ep_attn_dir))
                ATTENTION_TRACER.reset()
                policy.reset()
                if fag_wrapper is not None:
                    fag_wrapper.reset_episode()   # clear true_original so it's re-set from env
                obs, info = vec_env.reset(seed=[ep_idx])
                obs_orig  = obs.copy()

                step           = 0
                done           = np.array([False] * cfg.EVAL_BATCH_SIZE)
                max_steps      = vec_env.call("_max_episode_steps")[0]
                success        = False
                pending_rewrite: Optional[str] = None   # rewrite computed this step, applied next

                while not np.all(done) and step < max_steps:
                    ATTENTION_TRACER.update_step(step)

                    obs_tensor = preprocess_observation(obs)
                    obs_tensor = add_envs_task(vec_env, obs_tensor)

                    # Capture instruction text before preprocessors consume/rename the key.
                    current_instr_text = _get_obs_task(obs_tensor)

                    # Apply any rewrite that was computed at the previous inference step.
                    # (We can't apply it in the same step because the instruction must be
                    # ready before the policy preprocessor tokenises it.)
                    if fag_wrapper is not None and pending_rewrite is not None:
                        obs_tensor = _patch_task_instruction(obs_tensor, pending_rewrite)
                        current_instr_text = pending_rewrite
                        pending_rewrite = None

                    obs_tensor = env_preprocessor(obs_tensor)
                    obs_tensor = preprocessor(obs_tensor)

                    if ATTENTION_TRACER.is_collect_raw_images:
                        for k_env, k_tracer in [("image", "image1"), ("image2", "image2")]:
                            if "pixels" in obs_orig and k_env in obs_orig.get("pixels", {}):
                                img = obs_orig["pixels"][k_env][0]
                                import torch as _t
                                ATTENTION_TRACER.update_images(
                                    k_tracer,
                                    _t.from_numpy(img).permute(2, 0, 1).float() / 255.0
                                )
                    if "observation.language.tokens" in obs_tensor:
                        ATTENTION_TRACER.update_language_info({
                            "text_token_ids": obs_tensor["observation.language.tokens"],
                            "state": obs_tensor.get("observation.state"),
                        })

                    with torch.inference_mode():
                        action = policy.select_action(obs_tensor)
                    action     = postprocessor(action)
                    action_out = env_postprocessor({ACTION: action})[ACTION]
                    if hasattr(action_out, "detach"):
                        action_out = action_out.detach().cpu().numpy()

                    # Compute FAG rewrite NOW — attention for this step is still in memory
                    # before save_*() clears it.  The rewrite will be applied next iteration.
                    if fag_wrapper is not None:
                        pending_rewrite = fag_wrapper.maybe_rewrite(
                            step=step,
                            tracer=ATTENTION_TRACER,
                            current_instruction=current_instr_text,
                        )

                    obs, reward, done, truncated, info = vec_env.step(action_out)
                    obs_orig = obs.copy()

                    ATTENTION_TRACER.save_expert_attention()
                    ATTENTION_TRACER.save_language_attention()

                    # Correct key: LIBERO VectorEnv puts is_success in final_info
                    if "final_info" in info:
                        success = bool(np.any(info["final_info"].get("is_success", [False])))
                    else:
                        success = bool(np.any(info.get("is_success", [False])))
                    if success:
                        done = np.array([True] * cfg.EVAL_BATCH_SIZE)
                    step += 1

                ATTENTION_TRACER.save_language_info()

                # Real-robot: no simulator ground-truth — ask operator
                if getattr(self.args, "real_robot", False):
                    success = _query_human_success(ep_idx, step)

                ep_record = {
                    "ep":      ep_idx,
                    "task_id": task_id,
                    "success": success,
                    "steps":   step,
                    "mode":    "fag" if with_fag else "baseline",
                }
                if fag_wrapper is not None and fag_wrapper.step_results:
                    scores = [r.alignment_score for r in fag_wrapper.step_results]
                    vtis   = [r.diagnostics.get("vti_signal", 0.0) for r in fag_wrapper.step_results]
                    n = len(scores)
                    mid = n // 2
                    ep_record["lvci_mean"]    = float(sum(scores) / n)
                    ep_record["lvci_trend"]   = float(
                        sum(scores[mid:]) / max(n - mid, 1) - sum(scores[:max(mid,1)]) / max(mid, 1)
                    )
                    ep_record["vti_mean"]     = float(sum(vtis) / n)
                    ep_record["rewrite_count"] = sum(1 for r in fag_wrapper.step_results if r.was_rewritten)
                    ep_record["lvci_scores"]  = [round(s, 4) for s in scores]

                all_successes.append(ep_record)
                logger.info(f"  Episode {ep_idx+1} → success={success} in {step} steps")

            # Clean up real-robot env after each task
            if getattr(self.args, "real_robot", False):
                try:
                    vec_env.disconnect()
                except Exception:
                    pass

        success_rate = np.mean([r["success"] for r in all_successes])
        mode_label   = "FAG-VLA" if with_fag else "Baseline pi0.5"
        logger.info(f"\n{'='*50}")
        logger.info(f"{mode_label} success rate: {success_rate:.3f}")
        logger.info(f"{'='*50}\n")

        run_tag  = getattr(self.args, "run_tag", "")
        out_path = cfg.OUTPUT_DIR / f"eval_results_{self.args.mode}{run_tag}.json"
        with open(out_path, "w") as f:
            json.dump({
                "mode":         self.args.mode,
                "success_rate": success_rate,
                "episodes":     all_successes,
            }, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    # ------------------------------------------------------------------
    @staticmethod
    def _print_step_table(steps):
        print(f"\n{'Step':>6}  {'Alignment':>10}  {'Rewritten':>10}  Instruction")
        print("-" * 70)
        for s in steps:
            rw = "YES" if s.was_rewritten else "-"
            instr = (s.rewritten_instruction or s.instruction)[:40]
            print(f"{s.step:>6}  {s.alignment_score:>10.4f}  {rw:>10}  {instr}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_libero_normalization_stats(libero_suite: str, device: str = "cpu") -> dict:
    """
    Load per-feature normalization stats computed from LIBERO demos.

    Provides mean, std, q01, q99 for observation.state and action, satisfying
    both MEAN_STD and QUANTILE normalization modes used by pi0.5.

    The checkpoint's policy_preprocessor.json ships with features:{} (stats absent),
    which breaks the Pi05PrepareStateTokenizerProcessorStep that needs state in [-1,1].
    We compute stats from the raw HDF5 demos and return them in the format expected
    by NormalizerProcessorStep / UnnormalizerProcessorStep.
    """
    import json, torch
    from pathlib import Path

    # Pre-computed stats file
    stats_path = cfg._BASE / "data" / "libero10_normalization_stats.json"

    if stats_path.exists():
        with open(stats_path) as f:
            raw = json.load(f)
        # Check if q01/q99 are present; if not, recompute
        has_quantiles = all(
            "q01" in v and "q99" in v for v in raw.values()
        )
        if has_quantiles:
            dataset_stats = {}
            for key, val in raw.items():
                dataset_stats[key] = {
                    k: torch.tensor(v, dtype=torch.float32, device=device)
                    for k, v in val.items()
                }
            logger.info(f"Loaded normalization stats from {stats_path}")
            return dataset_stats
        else:
            logger.info("Cached stats missing q01/q99 — recomputing …")

    # Compute on-the-fly from LIBERO HDF5 demos
    logger.info("Computing normalization stats from LIBERO demos …")
    libero_root = cfg.LIBERO_DATA_ROOT
    suite_dir   = libero_root / libero_suite
    if not suite_dir.exists():
        suite_dir = libero_root / "libero_10"  # default fallback

    import h5py, numpy as np
    all_states, all_actions = [], []
    hdf5_files = sorted(suite_dir.glob("*.hdf5"))
    if not hdf5_files:
        logger.warning("No HDF5 files found; normalization stats will be empty.")
        return {}

    for fpath in hdf5_files:
        with h5py.File(fpath) as h:
            for demo_key in h["data"].keys():
                obs  = h["data"][demo_key]["obs"]
                acts = h["data"][demo_key]["actions"][:]
                ee_pos  = obs["ee_pos"][:]
                ee_ori  = obs["ee_ori"][:]          # already axis-angle (3D)
                gripper = obs["gripper_states"][:]
                state   = np.concatenate([ee_pos, ee_ori, gripper], axis=1)
                all_states.append(state)
                all_actions.append(acts)

    S = np.vstack(all_states)
    A = np.vstack(all_actions)

    def _to_torch(arr, device):
        return torch.tensor(arr.astype(np.float32), device=device)

    dataset_stats = {
        "observation.state": {
            "mean": _to_torch(S.mean(axis=0), device),
            "std":  _to_torch(S.std(axis=0).clip(min=1e-6), device),
            "q01":  _to_torch(np.percentile(S, 1, axis=0).astype(np.float32), device),
            "q99":  _to_torch(np.percentile(S, 99, axis=0).astype(np.float32), device),
        },
        "action": {
            "mean": _to_torch(A.mean(axis=0), device),
            "std":  _to_torch(A.std(axis=0).clip(min=1e-6), device),
            "q01":  _to_torch(np.percentile(A, 1, axis=0).astype(np.float32), device),
            "q99":  _to_torch(np.percentile(A, 99, axis=0).astype(np.float32), device),
        },
    }
    # Cache for future runs
    cfg._BASE.joinpath("data").mkdir(parents=True, exist_ok=True)
    cache = {k: {sk: sv.cpu().tolist() for sk, sv in v.items()}
             for k, v in dataset_stats.items()}
    with open(stats_path, "w") as f:
        json.dump(cache, f, indent=2)
    logger.info(f"Stats computed and saved to {stats_path}")
    return dataset_stats


def _make_pi05_processors_with_stats(policy_cfg, dataset_stats: dict, device: str):
    """
    Build pi0.5 pre/post-processor pipelines.

    Strategy (in order):
    1. If policy_preprocessor.json has embedded .safetensors normalization stats
       (fine-tuned checkpoint), load directly via make_pre_post_processors with
       local tokenizer override.
    2. Otherwise inject dataset_stats via make_pi05_pre_post_processors.

    The local PaliGemma tokenizer is always substituted for the gated HF repo.
    """
    _add_lerobot_to_path()
    local_tokenizer = str(cfg.TOKENIZER_PATH)
    pretrained_path = str(cfg.POLICY_PATH)

    # Tokenizer override: replace gated HF name with local path in every step config
    tokenizer_overrides = {
        "tokenizer_processor": {"tokenizer_name": local_tokenizer},
        "device_processor":    {"device": device},
        "rename_observations_processor": {"rename_map": {}},
    }

    # Check if the checkpoint ships its own normalization safetensors
    from pathlib import Path as _P
    has_norm_safetensors = any(
        _P(pretrained_path).glob("*normalizer_processor.safetensors")
    )

    if has_norm_safetensors:
        # Fine-tuned checkpoint: stats are embedded; just override tokenizer path
        logger.info("Fine-tuned checkpoint detected — loading preprocessor from checkpoint with local tokenizer.")
        try:
            from lerobot.policies.factory import make_pre_post_processors
            pre, post = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=pretrained_path,
                preprocessor_overrides=tokenizer_overrides,
            )
            logger.info("Built preprocessor from fine-tuned checkpoint stats.")
            return pre, post
        except Exception as e:
            logger.warning(f"Fine-tuned preprocessor load failed ({e}), falling back to injected stats.")

    # Base checkpoint (features:{}) or fallback: inject computed dataset_stats
    try:
        from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
        preprocessor, postprocessor = make_pi05_pre_post_processors(
            config=policy_cfg,
            dataset_stats=dataset_stats,
            tokenizer_name_override=local_tokenizer,
        )
        for step in preprocessor.steps:
            if hasattr(step, "device"):
                step.device = device
        logger.info("Built pi0.5 preprocessor with injected LIBERO normalization stats.")
        return preprocessor, postprocessor
    except TypeError:
        # make_pi05_pre_post_processors doesn't accept tokenizer_name_override —
        # patch the config object directly
        pass
    except Exception as e:
        logger.warning(f"make_pi05_pre_post_processors failed ({e})")

    # Last resort: patch tokenizer config on policy_cfg and call make_pre_post_processors
    try:
        _patch_tokenizer_in_policy_cfg(policy_cfg, local_tokenizer)
        from lerobot.policies.factory import make_pre_post_processors
        return make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=pretrained_path,
            preprocessor_overrides=tokenizer_overrides,
        )
    except Exception as e:
        raise RuntimeError(f"All preprocessor build paths failed. Last error: {e}") from e


def _patch_tokenizer_in_policy_cfg(policy_cfg, local_tokenizer_path: str):
    """Recursively replace gated paligemma tokenizer name with local path in policy config."""
    gated = "google/paligemma-3b-pt-224"
    def _replace(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "tokenizer_name" and v == gated:
                    obj[k] = local_tokenizer_path
                else:
                    _replace(v)
        elif isinstance(obj, list):
            for item in obj:
                _replace(item)
    # Try common config attributes
    for attr in ("preprocessor_config", "text_config", "model_config"):
        sub = getattr(policy_cfg, attr, None)
        if sub is not None:
            _replace(vars(sub) if hasattr(sub, "__dict__") else sub)
    _replace(vars(policy_cfg) if hasattr(policy_cfg, "__dict__") else {})


def _add_lerobot_to_path():
    """Make a locally-cloned LeRobot tree importable.

    Set the env var ``LEROBOT_SRC`` to the directory that contains LeRobot's
    ``src/`` (and optionally ``projects/lerobot/src/`` for ExplainVLA-style
    layouts). If unset, this is a no-op and LeRobot must already be on
    PYTHONPATH (e.g. installed via ``pip install lerobot``).
    """
    lerobot_root = os.getenv("LEROBOT_SRC")
    if not lerobot_root:
        return
    lerobot_src = Path(lerobot_root)
    for sub in ["projects/lerobot/src", "src", ""]:
        p = lerobot_src / sub if sub else lerobot_src
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _load_pi05_policy(policy_path: str):
    """Load the pre-trained pi0.5 policy.

    Bug fix: config.json inside fine-tuned checkpoints contains a `pretrained_path`
    field pointing to the original base model (e.g. "lerobot/pi05_libero").  If we
    pass that config directly to `make_policy`, it loads the BASE weights from HF
    instead of the local fine-tuned weights.  We override `pretrained_path` to the
    local directory so `from_pretrained` loads the actual fine-tuned `model.safetensors`.
    """
    _add_lerobot_to_path()
    import torch
    from pathlib import Path
    from lerobot.policies.factory import make_policy
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path, device=device)

    # Force make_policy to load weights from the LOCAL checkpoint directory,
    # not from whatever pretrained_path the config.json originally recorded.
    local_path = str(Path(policy_path).resolve())
    if getattr(policy_cfg, "pretrained_path", None) != local_path:
        logger.info(
            f"Overriding pretrained_path: {policy_cfg.pretrained_path!r} → {local_path!r} "
            f"(ensures fine-tuned weights are loaded, not base model)"
        )
        policy_cfg.pretrained_path = local_path

    env_cfg = LiberoEnvConfig(task=cfg.EVAL_LIBERO_SUITE, task_ids=[0])
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    return policy


def _make_libero_env(suite: str, task_id: int, batch_size: int = 1):
    """Instantiate a single LIBERO vec_env for one task."""
    _add_lerobot_to_path()
    from lerobot.envs.factory import make_env
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

    env_cfg = LiberoEnvConfig(task=suite, task_ids=[task_id])
    # make_env returns {suite_name: {task_id: vec_env}}
    envs_dict = make_env(cfg=env_cfg, n_envs=batch_size)
    # Extract the single vec_env for the requested task
    suite_envs = next(iter(envs_dict.values()))
    vec_env = next(iter(suite_envs.values()))
    return vec_env


def _get_obs_task(obs: dict) -> str:
    """Extract current task string from a preprocessed observation dict."""
    task_val = obs.get("observation.task_description", obs.get("task", ""))
    if isinstance(task_val, list):
        return str(task_val[0]) if task_val else ""
    return str(task_val)


def _query_human_success(ep_idx: int, steps: int) -> bool:
    """
    Prompt the operator to manually annotate episode success for real-robot runs.
    Accepted inputs: s/S/1/y/Y → success;  f/F/0/n/N → failure.
    """
    print(f"\n  ┌─ Episode {ep_idx+1} finished ({steps} steps) ──────────────────────")
    print(  "  │  Did the robot complete the task?")
    print(  "  │  [s] Success    [f] Failure    [?] Skip (mark as failure)")
    while True:
        try:
            ans = input("  └─ Result: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "f"
        if ans in ("s", "1", "y", "yes", "success"):
            print("  → Marked: SUCCESS")
            return True
        if ans in ("f", "0", "n", "no", "fail", "failure", "?", ""):
            print("  → Marked: FAILURE")
            return False
        print("  Input not recognised. Enter 's' or 'f'.")


def _patch_task_instruction(obs: dict, instruction: str) -> dict:
    """Inject rewritten instruction into observation dict (pre-preprocessor format)."""
    for key in ("observation.task_description", "task"):
        if key in obs:
            v = obs[key]
            if isinstance(v, list):
                obs[key] = [instruction] * len(v)
            else:
                obs[key] = instruction
    return obs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="FAG-VLA Evaluation")
    parser.add_argument("--mode", default="fag_offline",
                        choices=["baseline", "fag_offline", "fag_online"],
                        help="Evaluation mode")
    parser.add_argument("--tasks", default="libero_object",
                        help="LIBERO suite name")
    parser.add_argument("--task_ids", default="[0]",
                        help="Task IDs as JSON list, e.g. '[0,1,2]'")
    parser.add_argument("--n_episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--rewrite", action="store_true", default=True,
                        help="Enable GPT-4o instruction rewriting")
    parser.add_argument("--rewrite_threshold", type=float,
                        default=cfg.ALIGN_REWRITE_THRESHOLD,
                        help="Alignment score below which rewriting triggers")
    parser.add_argument("--attn_dir", default=None,
                        help="(fag_offline) Directory with saved attention .pkl files")
    parser.add_argument("--real_robot", action="store_true", default=False,
                        help="Use real robot interface instead of simulation")
    parser.add_argument("--real_robot_task", default="pick up the bowl",
                        help="Task description for real robot experiments")
    parser.add_argument("--can_port", default="can0",
                        help="CAN interface port for PIPER arm (default: can0)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max steps per episode (real robot only)")
    parser.add_argument("--run_tag", default="",
                        help="Optional tag appended to output filename, e.g. '_t015'")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg.ALIGN_REWRITE_THRESHOLD = args.rewrite_threshold
    evaluator = FAGEvaluator(args)
    evaluator.run()
