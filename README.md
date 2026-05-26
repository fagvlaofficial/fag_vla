<h1 align="center">FAG-VLA</h1>

<p align="center">
  <b>Fused Attention Graph for Vision-Language-Action Models</b><br/>
  <i>An interpretability-driven, training-free intervention framework for <code>pi0.5</code>-style VLA policies.</i>
</p>

<p align="center">
  <img src="assets/system_architecture.png" alt="FAG-VLA system architecture" width="860"/>
</p>

<p align="center">
  <a href="#motivation">Motivation</a> ·
  <a href="#whats-in-the-framework">Framework</a> ·
  <a href="#results">Results</a> ·
  <a href="#installation">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#configuration">Config</a> ·
  <a href="#citation">Cite</a>
</p>

---

FAG-VLA observes the cross-modal attention flow inside a running VLA, scores how "visually grounded" the model currently is, and adaptively rewrites the language instruction *only when the model is drifting away from visual execution*. On LIBERO it lifts `pi0.5` from **93.3 % → 96.7 %** success rate without any fine-tuning.

> **TL;DR.** Language–visual cross-modal attention **decreases** over time when an episode succeeds, and **increases** when it fails. We turn that *trend signal* into a closed-loop "observe → score → intervene" controller around a frozen VLA.

A Chinese-language introduction is available at [`README_zh.md`](README_zh.md).

---

## Motivation

<p align="center">
  <img src="assets/attention_mechanism_in_VLA.png" alt="Cross-modal attention in a VLA model" width="720"/>
  <br/>
  <sub><i>Where the attention flows: image patches ↔ instruction tokens ↔ state tokens inside a dual-stream VLA.</i></sub>
</p>

We analysed 150 LIBERO baseline rollouts (140 success / 10 fail) and found a counter-intuitive pattern in cross-modal attention:

| Metric | Success (n=140) | Failure (n=10) | Cohen's *d* |
|---|---|---|---|
| Mean LVCI | 0.199 | 0.262 | 1.53 *** |
| **ΔLVCI** (late half − early half) | **−0.095** | **+0.049** | **3.96 ***  (strongest signal)** |
| VTI mean | 0.595 | 0.745 | 1.80 *** |
| % steps with LVCI < 0.20 | 62.5 % | 13.3 % | 3.56 *** |

When the model successfully **grounds** an instruction into the scene, it stops "re-reading" the language — visual–text attention naturally **decays**. When it fails to ground, it keeps re-querying the instruction, and cross-modal coupling **rises**. The trend, not the absolute level, is the strongest predictor.

This re-frames the intervention question: **don't rewrite based on how high the coupling is — rewrite based on whether the coupling is *rising*.**

---

## What's in the framework

<p align="center">
  <img src="assets/fag_vla_principle.png" alt="FAG-VLA closed-loop principle" width="860"/>
  <br/>
  <sub><i>FAG-VLA wraps a frozen VLA with a three-stage closed loop: build the graph, score the coupling, intervene only when the trend turns.</i></sub>
</p>

FAG-VLA is built around three modules that wrap a frozen VLA:

```
              attention tensors (VLM + Action-Expert)
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  FusedAttentionGraph                │ ← cross-layer aggregation
        │   nodes  = image / text / state     │   + cross-stream bridging
        │   edges  = attention weights        │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  LV-Coupling Scorer                 │ ← LVCI / VGS, trend slope
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  Adaptive Instruction Rewriter      │ ← 3-layer trigger,
        │   ① VTI prior   ② ΔLVCI trend       │   "revert-first" strategy
        │   ③ VTI confirmation                │
        └─────────────────────────────────────┘
                              │
                              ▼
                       updated instruction
```

### 1. Fused Attention Graph (FAG)

Build a directed weighted graph G = (V, E) per inference step:
- **Nodes**: tokens grouped by modality — image patches, instruction tokens, state tokens.
- **Edges**: attention weights, aggregated across heads (mean) and layers (mean).
- **Two streams fused**: 18-layer PaliGemma VLM attention + 10-layer Gemma-Expert attention, bridged via hidden-state cosine similarity to avoid dimensional mismatch.

### 2. LV-Coupling Scorer (LVCI / VGS)

The **Language-Visual Coupling Intensity** for one step is:

```
LVCI = w₁ · S_VTI  +  w₂ · S_TVI  −  w₃ · H_tok  +  w₄ · M_tok
```

with `S_VTI` = visual→text flow, `S_TVI` = text→visual flow, `H_tok` = token-weight entropy, `M_tok` = max token weight. The complementary **Visual Grounding Score** `VGS = 1 − LVCI` is the "execution health" indicator (high VGS = grounded).

### 3. Adaptive Instruction Rewriter

Triggers and actions:

```
Layer 1 (step 0, prior check):     VTI_0 ≥ 0.75  AND  LVCI_0 ≥ 0.28
                                   → pre-emptive light rewrite

Layer 2 (every 50 steps):          ΔLVCI > 0.02  AND  LVCI ≥ 0.28
                                   AND  ≥ 2 checkpoints since last rewrite
                                   → trigger rewrite

Layer 3 (action selection):        VTI ≥ 0.85  → LLM front-loading rewrite
                                   VTI < 0.85  → revert to original instruction
```

The key empirical insight is the **revert-first** policy: when an LLM-rewritten instruction has drifted, the most reliable repair is to restore the original instruction, not to keep stacking LLM rewrites. LLM rewriting is only invoked when the model is severely off-track (high VTI).

---

## Results

LIBERO, `pi0.5` checkpoint, 50 episodes × 3 tasks:

| Method | Success rate | Δ vs baseline |
|---|---|---|
| Baseline `pi0.5` | 93.3 % | — |
| **FAG-VLA (ours)** | **96.7 %** | **+3.4 pp** |

---

## Repository layout

```
FAG-VLA/
├── src/fag_vla/                    # flat package — every module is one file
│   ├── fused_attention_graph.py    # FAG construction + temporal analysis
│   ├── alignment_scorer.py         # LVCI / VGS / VTI scoring
│   ├── instruction_rewriter.py     # LLM front-loading + revert strategy
│   ├── fag_pipeline.py             # end-to-end pipeline (offline + online)
│   ├── fag_eval.py                 # LIBERO + real-robot evaluation loop
│   ├── piper_d435i_interface.py    # PIPER 6-DOF + RealSense D435i driver
│   ├── paper_plots.py              # 900 DPI publication figures
│   └── settings.py                 # all paths + thresholds (env-driven)
├── assets/                         # architecture & principle figures
├── requirements.txt
├── .env.example
├── .gitignore
└── LICENSE
```

---

## Installation

### 1. Python dependencies

```bash
# create a fresh virtualenv (recommended), then:
pip install -r requirements.txt
```

`requirements.txt` lists every package FAG-VLA itself uses (`torch`, `numpy`, `networkx`, `scipy`, `transformers`, `openai`, `matplotlib`, `seaborn`, `pillow`, `tqdm`, `einops`, `lerobot`, …). Optional subsystems — LIBERO, the PIPER SDK, RealSense — are listed at the bottom of that file and may be enabled when you need them.

### 2. External assets (download separately)

| Asset | Source | Expected path (default) |
|---|---|---|
| `pi0.5` LIBERO checkpoint | LeRobot HuggingFace hub | `./checkpoints/lerobot_pi05_libero` |
| PaliGemma tokenizer | HuggingFace | `./checkpoints/paligemma_tokenizer` |
| LIBERO dataset | [LIBERO release](https://github.com/Lifelong-Robot-Learning/LIBERO) | `./datasets/LIBERO_DATASETS` |
| LeRobot source *(if not pip-installed)* | [LeRobot repo](https://github.com/huggingface/lerobot) | set via `LEROBOT_SRC` env var |

You can override any default location with the matching environment variable (see [`.env.example`](.env.example)).

### 3. Environment

Copy the template and fill in your own credentials:

```bash
cp .env.example .env
# edit .env, set at minimum OPENAI_API_KEY for fag_online mode
```

The OpenAI API key is **only** required when running `fag_online` evaluation with LLM-based instruction rewriting. `baseline`, `fag_offline` (analysis on pre-saved attention pickles) and revert-only `fag_online` traces do not consume any LLM tokens.

---

## Quickstart

### Run the LIBERO evaluation directly

```bash
# export your key first (or rely on .env via direnv / dotenv)
export OPENAI_API_KEY="sk-…"

# Baseline pi0.5 (no FAG intervention)
python -m fag_vla.fag_eval --mode baseline \
    --tasks libero_object --task_ids "[0,1,2]" --n_episodes 5

# FAG-VLA online — adaptive instruction rewriting
python -m fag_vla.fag_eval --mode fag_online --rewrite \
    --tasks libero_object --task_ids "[0,1,2]" --n_episodes 5

# Post-hoc analysis on previously saved attention pickles
python -m fag_vla.fag_eval --mode fag_offline \
    --tasks libero_object --attn_dir ./data/attention_data
```

Per-episode JSON results land in `$FAG_BASE_DIR/data/outputs/`, attention pickles in `$FAG_BASE_DIR/data/attention_data/`. Both directories are created automatically at first run.

### Use the pipeline as a library

```python
from fag_vla.fag_pipeline import FAGPipeline

pipeline = FAGPipeline(
    tokenizer_path="./checkpoints/paligemma_tokenizer",
    save_graphs=True,
    rewrite_enabled=True,
)

result = pipeline.run_offline(
    attn_data_dir="./data/attention_data",
    episode_id=0,
    task_name="libero_object",
)
print(result.mean_alignment, result.rewrite_count)
```

### Real-robot rollout (PIPER 6-DOF + RealSense D435i)

```python
from fag_vla.piper_d435i_interface import PiperD435iInterface

env = PiperD435iInterface(task_name="pick_metal_bowl",
                          can_port="can0", max_steps=200)
env.connect()
# … then plug `env` into your evaluation loop (see fag_eval.py)
```

Safety checklists are printed by the interface before motion starts.

---

## Configuration

Every tunable lives in [`src/fag_vla/settings.py`](src/fag_vla/settings.py) and is environment-overridable. Highlights:

| Variable | Default | Purpose |
|---|---|---|
| `FAG_BASE_DIR` | repo root | Where `data/`, `visualizations/`, etc. resolve |
| `POLICY_PATH` | `<repo>/checkpoints/lerobot_pi05_libero` | `pi0.5` weights |
| `PALIGEMMA_TOKENIZER_PATH` | `<repo>/checkpoints/paligemma_tokenizer` | Tokenizer for token-segment math |
| `LIBERO_DATA_ROOT` | `<repo>/datasets/LIBERO_DATASETS` | LIBERO HDF5 datasets |
| `LEROBOT_SRC` | *(unset)* | Optional: path to a local LeRobot checkout |
| `OPENAI_API_KEY` | *(empty)* | Required for `fag_online` LLM rewriting |
| `OPENAI_BASE_URL` | *(empty)* | Optional: custom OpenAI-compatible endpoint |
| `OPENAI_LLM_MODEL` | `gpt-4o-mini` | Rewriter model |
| `LVCI_TREND_THRESHOLD` | `0.02` | Δτ — Layer-2 trigger |
| `LVCI_FLOOR` | `0.28` | Minimum absolute LVCI before trend trigger fires |
| `VTI_STRONG_THRESHOLD` | `0.85` | VTI threshold above which LLM rewrite is used |
| `REWRITE_COOLDOWN_STEPS` | `2` | Checkpoints to skip after a rewrite |

---

## Citation

If you use FAG-VLA in your research, please cite:

```bibtex
@misc{fagvla2026,
  title   = {FAG-VLA: Fused Attention Graphs for Adaptive Intervention in Vision-Language-Action Models},
  author  = {Lei Gao},
  year    = {2026},
  note    = {Huazhong University of Science and Technology},
  howpublished = {\url{https://github.com/<your-org>/FAG-VLA}}
}
```

(Update the BibTeX once the paper is published.)

---

## Acknowledgments

This project builds directly on top of:

- **LeRobot** (HuggingFace) — `pi0.5` policy implementation and runtime.
- **PaliGemma** — VLM backbone whose attention is the primary signal.
- **LIBERO** — simulation benchmark for VLA generalisation.

---

## License

Released under the [MIT License](LICENSE).

Copyright © 2026 Lei Gao, Huazhong University of Science and Technology.
