"""
FAG-VLA Full Pipeline

Orchestrates the complete Fused Attention Graph pipeline:

  1. Load attention data collected by AttentionTracer (or run live inference)
  2. Build FAG for each step
  3. Score instruction–vision alignment
  4. If score < threshold → rewrite instruction via GPT-4o
  5. (Optional) Re-run inference with rewritten instruction
  6. Persist graphs, scores, and rewrite logs
  7. Return structured results for downstream visualization / evaluation

This module is designed to work both:
  - Offline: loading pre-saved .pkl attention files from ExplainVLA runs
  - Online:  receiving live tracer data during evaluation (for robot deployment)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

@dataclass
class StepFAGResult:
    step: int
    instruction: str
    rewritten_instruction: Optional[str]
    alignment_score: float
    was_rewritten: bool
    graph_path: Optional[str]           # path to saved nx.DiGraph pickle
    token_weights: np.ndarray
    patch_weights_v1: np.ndarray
    patch_weights_v2: np.ndarray
    temporal_convergence: Optional[Dict]
    diagnostics: Dict


@dataclass
class EpisodeFAGResult:
    episode_id: int
    task_name: str
    steps: List[StepFAGResult] = field(default_factory=list)
    mean_alignment: float = 0.0
    rewrite_count: int = 0
    instruction_history: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class FAGPipeline:
    """
    Full FAG-VLA pipeline.

    Parameters
    ----------
    config_module: the fag_vla.settings module (or equivalent dict).
    tokenizer_path: path to PaliGemma tokenizer (for token decoding).
    save_graphs: whether to persist nx.DiGraph objects to disk.
    rewrite_enabled: whether to call GPT-4o for instruction rewriting.
    """

    def __init__(
        self,
        config_module=None,
        tokenizer_path: Optional[str] = None,
        save_graphs: bool = True,
        rewrite_enabled: bool = True,
    ):
        import fag_vla.settings as cfg
        self.cfg = cfg
        cfg.ensure_dirs()

        # Token-segment config from settings
        self._base_segments: Dict[str, Tuple[int, int]] = {
            "image1": cfg.IMAGE1_TOKENS,
            "image2": cfg.IMAGE2_TOKENS,
        }
        if not cfg.INCLUDE_IMAGE3:
            pass  # image3 excluded by default
        else:
            self._base_segments["image3"] = cfg.IMAGE3_TOKENS

        # Tokenizer for decoding instruction tokens
        tok_path = tokenizer_path or str(cfg.TOKENIZER_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)

        self.save_graphs  = save_graphs
        self.rewrite_enabled = rewrite_enabled

        # Lazy-init heavy components
        self._fag: Optional[object] = None
        self._scorer: Optional[object] = None
        self._rewriter: Optional[object] = None

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run_offline(
        self,
        attn_data_dir: Optional[Path] = None,
        episode_id: int = 0,
        task_name: str = "unknown",
        language_info: Optional[Dict] = None,
    ) -> EpisodeFAGResult:
        """
        Run FAG analysis on pre-saved attention data.

        attn_data_dir: directory with {step}_expert_attention.pkl and
                       {step}_language_attention.pkl files, plus language_info/.
        """
        cfg = self.cfg
        attn_dir = attn_data_dir or cfg.ATTN_DATA_DIR

        # Load attention data
        lang_attn  = self._load_language_attn(attn_dir)
        exp_attn   = self._load_expert_attn(attn_dir)
        lang_info  = language_info or self._load_language_info(attn_dir)

        steps = sorted(lang_attn.keys())
        logger.info(f"Loaded attention for {len(steps)} steps")

        episode = EpisodeFAGResult(episode_id=episode_id, task_name=task_name)
        current_instruction = self._decode_instruction(lang_info, steps[0] if steps else 0)

        # Resolve language info: if a step has no entry, fall back to the nearest available.
        # Within one episode the instruction rarely changes, so step-0 info is typically valid.
        _lang_info_steps = sorted(lang_info.keys()) if lang_info else []
        def _resolve_lang_info(step):
            if step in lang_info:
                return lang_info[step]
            if _lang_info_steps:
                nearest = min(_lang_info_steps, key=lambda s: abs(s - step))
                return lang_info[nearest]
            return {}

        for step in steps:
            vlm_attn  = lang_attn.get(step, {})
            e_attn_ts = exp_attn.get(step, {})

            token_texts, seg_override = self._get_token_texts_and_segments(
                {step: _resolve_lang_info(step)}, step
            )
            fag = self._get_fag(seg_override)
            scorer = self._get_scorer(fag)

            # Build FAG (all diffusion timesteps for temporal analysis)
            G = fag.build(vlm_attn, e_attn_ts.get(0, {}))

            # Temporal sequence
            temporal = None
            if len(e_attn_ts) > 1:
                from fag_vla.fused_attention_graph import analyse_temporal_convergence
                graphs_ts = fag.build_temporal_sequence(vlm_attn, e_attn_ts)
                temporal = {
                    "image1_text": asdict_array(
                        analyse_temporal_convergence(graphs_ts, "image1", "text", fag)
                    ),
                    "text_image1": asdict_array(
                        analyse_temporal_convergence(graphs_ts, "text", "image1", fag)
                    ),
                }

            # Alignment score
            result_score = scorer.score(G)

            # Instruction rewrite
            rewritten = None
            was_rewritten = False
            if (
                self.rewrite_enabled
                and result_score.score < self.cfg.ALIGN_REWRITE_THRESHOLD
            ):
                rewriter = self._get_rewriter()
                rw = rewriter.rewrite(
                    original=current_instruction,
                    token_texts=token_texts,
                    token_weights=result_score.token_weights,
                    alignment_score=result_score.score,
                )
                rewritten = rw.rewritten
                was_rewritten = (rewritten != current_instruction)
                current_instruction = rewritten
                # Log rewrite
                self._save_rewrite_log(step, episode_id, rw)

            # Save graph
            graph_path = None
            if self.save_graphs:
                graph_path = str(
                    self.cfg.GRAPH_DATA_DIR / f"ep{episode_id:04d}_step{step:04d}.pkl"
                )
                with open(graph_path, "wb") as f:
                    pickle.dump(G, f)

            step_result = StepFAGResult(
                step=step,
                instruction=current_instruction
                    if not was_rewritten
                    else episode.instruction_history[-1]
                    if episode.instruction_history else current_instruction,
                rewritten_instruction=rewritten,
                alignment_score=result_score.score,
                was_rewritten=was_rewritten,
                graph_path=graph_path,
                token_weights=result_score.token_weights,
                patch_weights_v1=result_score.patch_weights_view1,
                patch_weights_v2=result_score.patch_weights_view2,
                temporal_convergence=temporal,
                diagnostics=result_score.diagnostics,
            )
            episode.steps.append(step_result)
            episode.instruction_history.append(current_instruction)
            if was_rewritten:
                episode.rewrite_count += 1

        if episode.steps:
            episode.mean_alignment = float(
                np.mean([s.alignment_score for s in episode.steps])
            )

        self._save_episode_summary(episode)
        return episode

    def run_online_step(
        self,
        step: int,
        vlm_layer_attn: Dict,
        expert_attn_ts: Dict,
        current_instruction: str,
        lang_info: Optional[Dict] = None,
        true_original_instruction: Optional[str] = None,
        rewrite_override: bool = False,
        lvci_trend: float = 0.0,
        vti_signal_hint: float = 0.0,
        rewrite_strength: str = "medium",
    ) -> Tuple[str, StepFAGResult]:
        """
        Online single-step FAG analysis with optional instruction rewrite.
        Called during live robot evaluation.

        Parameters
        ----------
        true_original_instruction : str, optional
            The very first instruction issued at episode start (ground truth vocabulary).
        rewrite_override : bool
            If True, trigger rewrite regardless of score threshold (used by
            FAGOnlineWrapper's trend-based trigger). Takes precedence over
            the legacy absolute-threshold check.
        lvci_trend : float
            ΔLVCI computed by the wrapper (passed to rewriter prompt for context).
        vti_signal_hint : float
            VTI sub-score passed by the wrapper (determines rewrite strength).
        rewrite_strength : str
            "light" | "medium" | "strong" — controls prompt aggressiveness.

        Returns (instruction_to_use, StepFAGResult).
        """
        token_texts, seg_override = self._get_token_texts_and_segments(
            lang_info or {}, step
        )
        fag    = self._get_fag(seg_override)
        scorer = self._get_scorer(fag)

        G = fag.build(vlm_layer_attn, expert_attn_ts.get(0, {}))
        result_score = scorer.score(G)

        vti_signal = result_score.diagnostics.get("vti_signal", 0.0)
        # Use hint from wrapper if available, otherwise use computed value
        effective_vti = vti_signal_hint if vti_signal_hint > 0 else vti_signal

        # Trigger decision: rewrite_override (trend-based) takes priority;
        # legacy absolute threshold kept only for ablation mode.
        should_rewrite = self.rewrite_enabled and (
            rewrite_override
            or result_score.score < self.cfg.ALIGN_REWRITE_THRESHOLD
        )

        logger.info(
            f"Step {step:03d} LVCI={result_score.score:.3f} "
            f"VTI={vti_signal:.3f} trend={lvci_trend:+.3f} "
            f"{'→ REWRITE[%s]' % rewrite_strength if should_rewrite else '→ OK'}"
        )

        rewritten = None
        was_rewritten = False
        if should_rewrite:
            rewriter = self._get_rewriter()
            rw = rewriter.rewrite(
                original=current_instruction,
                token_texts=token_texts,
                token_weights=result_score.token_weights,
                alignment_score=result_score.score,
                true_original=true_original_instruction,
                lvci_trend=lvci_trend,
                vti_signal=effective_vti,
                rewrite_strength=rewrite_strength,
            )
            rewritten = rw.rewritten
            was_rewritten = (rewritten.strip() != current_instruction.strip())

        instruction_to_use = rewritten if was_rewritten else current_instruction

        step_result = StepFAGResult(
            step=step,
            instruction=current_instruction,
            rewritten_instruction=rewritten,
            alignment_score=result_score.score,
            was_rewritten=was_rewritten,
            graph_path=None,
            token_weights=result_score.token_weights,
            patch_weights_v1=result_score.patch_weights_view1,
            patch_weights_v2=result_score.patch_weights_view2,
            temporal_convergence=None,
            diagnostics=result_score.diagnostics,
        )
        return instruction_to_use, step_result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_fag(self, seg_override: Optional[Dict] = None):
        from fag_vla.fused_attention_graph import FusedAttentionGraph, FAGConfig
        segments = seg_override or self._base_segments
        cfg = FAGConfig(
            segments=segments,
            layer_agg=self.cfg.FAG_LAYER_AGG,
            head_merge=self.cfg.FAG_HEAD_MERGE,
            edge_threshold=self.cfg.FAG_EDGE_THRESHOLD,
            cross_model_bridge=self.cfg.FAG_CROSS_BRIDGE,
            include_image3=self.cfg.INCLUDE_IMAGE3,
        )
        return FusedAttentionGraph(cfg)

    def _get_scorer(self, fag):
        from fag_vla.alignment_scorer import AttentionAlignmentScorer
        return AttentionAlignmentScorer(
            fag,
            vti_weight=self.cfg.ALIGN_VTI_WEIGHT,
            tvi_weight=self.cfg.ALIGN_TVI_WEIGHT,
            entropy_penalty=self.cfg.ALIGN_ENTROPY_PENALTY,
            concentration_bonus=self.cfg.ALIGN_CONCENTRATION_BONUS,
        )

    def _get_rewriter(self):
        from fag_vla.instruction_rewriter import InstructionRewriter
        if self._rewriter is None:
            self._rewriter = InstructionRewriter(
                api_key=self.cfg.OPENAI_API_KEY,
                base_url=self.cfg.OPENAI_BASE_URL,
                model=self.cfg.OPENAI_LLM_MODEL,
                max_retries=self.cfg.REWRITE_MAX_RETRIES,
            )
        return self._rewriter

    def _get_token_texts_and_segments(
        self,
        lang_info: Dict,
        step: int,
    ) -> Tuple[List[str], Dict]:
        """
        Decode token texts for the given step and compute dynamic segment boundaries.
        Falls back to base image-only segments when language info unavailable.
        """
        token_texts = []
        segments = dict(self._base_segments)

        if not lang_info or step not in lang_info:
            return token_texts, segments

        info = lang_info[step]
        if not info:
            return token_texts, segments

        token_ids = info.get("text_token_ids", [[]])[0]
        decoded = self.tokenizer.convert_ids_to_tokens(token_ids)
        decoded = [t.replace("▁", " ").strip() for t in decoded]

        # Find Task: / State: / Action: boundaries
        task_start = task_end = state_start = state_end = None
        for i, tok in enumerate(decoded):
            if tok == "Task" and i + 1 < len(decoded) and decoded[i + 1] == ":":
                task_start = i
            if tok == "State" and i + 1 < len(decoded) and decoded[i + 1] == ":":
                task_end = i
                state_start = i
            if tok == "Action" and i + 1 < len(decoded) and decoded[i + 1] == ":":
                state_end = i

        image_end = 768  # 3 × 256 patches
        if task_start is not None and task_end is not None:
            t_start = task_start + image_end
            t_end   = task_end   + image_end
            segments["text"] = (t_start, t_end)
            token_texts = decoded[task_start:task_end]

        if state_start is not None and state_end is not None:
            s_start = state_start + image_end
            s_end   = state_end   + image_end
            segments["state"] = (s_start, s_end)

        return token_texts, segments

    def _decode_instruction(self, lang_info: Dict, step: int) -> str:
        texts, _ = self._get_token_texts_and_segments(lang_info, step)
        return " ".join(texts).strip()

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_language_attn(attn_dir: Path) -> Dict:
        result = {}
        for pkl in sorted(attn_dir.glob("*_language_attention.pkl")):
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            for step, layer_dict in data.items():
                if not layer_dict:
                    continue
                # New compact format: {"agg": tensor[seq,seq], "_n": N}
                if "agg" in layer_dict:
                    result[step] = {0: layer_dict["agg"]}
                else:
                    # Legacy format: {layer_idx: tensor[1,heads,seq,seq]}
                    result[step] = layer_dict
        return result

    @staticmethod
    def _load_expert_attn(attn_dir: Path) -> Dict:
        result = {}
        for pkl in sorted(attn_dir.glob("*_expert_attention.pkl")):
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            for step, ts_dict in data.items():
                if not ts_dict:
                    continue
                result[step] = {}
                for ts, layer_dict in ts_dict.items():
                    if not layer_dict:
                        continue
                    # New compact format: {"agg": tensor[seq,seq], "_n": N}
                    if "agg" in layer_dict:
                        result[step][ts] = {0: layer_dict["agg"]}
                    else:
                        result[step][ts] = layer_dict
        return result

    @staticmethod
    def _load_language_info(attn_dir: Path) -> Dict:
        info_path = attn_dir.parent / "language_info" / "language_info.pkl"
        if not info_path.exists():
            # Try same directory
            info_path = attn_dir / "language_info" / "language_info.pkl"
        if info_path.exists():
            with open(info_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_rewrite_log(self, step: int, episode_id: int, rw):
        log = {
            "step": step,
            "episode_id": episode_id,
            "original": rw.original,
            "rewritten": rw.rewritten,
            "alignment_before": rw.alignment_before,
            "model_used": rw.model_used,
            "token_analysis": rw.token_analysis,
        }
        log_path = (
            self.cfg.REWRITE_LOG_DIR
            / f"ep{episode_id:04d}_step{step:04d}_rewrite.json"
        )
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    def _save_episode_summary(self, episode: EpisodeFAGResult):
        summary = {
            "episode_id":     episode.episode_id,
            "task_name":      episode.task_name,
            "mean_alignment": episode.mean_alignment,
            "rewrite_count":  episode.rewrite_count,
            "n_steps":        len(episode.steps),
            "instruction_history": episode.instruction_history,
            "per_step": [
                {
                    "step": s.step,
                    "alignment_score": s.alignment_score,
                    "was_rewritten": s.was_rewritten,
                    "rewritten_instruction": s.rewritten_instruction,
                    "diagnostics": s.diagnostics,
                }
                for s in episode.steps
            ],
        }
        out_path = (
            self.cfg.OUTPUT_DIR
            / f"ep{episode.episode_id:04d}_{episode.task_name}_summary.json"
        )
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Episode summary saved → {out_path}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def asdict_array(d: Dict) -> Dict:
    """Convert numpy arrays to lists for JSON serialisation."""
    return {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in d.items()
    }
