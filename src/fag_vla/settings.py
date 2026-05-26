"""
FAG-VLA Settings
Inherits token-segment conventions from ExplainVLA and extends with FAG-specific config.

All filesystem paths are resolved from environment variables (with sensible
defaults relative to the repository root). To run on your own machine, export
the variables shown in `.env.example` before launching any script.
"""

import os
from pathlib import Path
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Directory layout (env-driven; defaults are relative to the repo root)
# ---------------------------------------------------------------------------

# Repo root resolves to <repo>/  when this file lives at src/fag_vla/
_REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[2]
_BASE = Path(os.getenv("FAG_BASE_DIR", str(_REPO_ROOT_DEFAULT)))

# Model assets — point these at your local HuggingFace cache or downloaded checkpoints.
POLICY_PATH      = Path(os.getenv("POLICY_PATH",
                         str(_BASE / "checkpoints" / "lerobot_pi05_libero")))
TOKENIZER_PATH   = Path(os.getenv("PALIGEMMA_TOKENIZER_PATH",
                         str(_BASE / "checkpoints" / "paligemma_tokenizer")))
HF_CACHE         = Path(os.getenv("HF_HOME",
                         str(_BASE / "checkpoints" / "hf_cache")))

# LIBERO dataset root (download separately from the official LIBERO release)
LIBERO_DATA_ROOT = Path(os.getenv("LIBERO_DATA_ROOT",
                         str(_BASE / "datasets" / "LIBERO_DATASETS")))

# FAG analysis data (intermediate storage)
DATA_DIR         = _BASE / "data"
ATTN_DATA_DIR    = DATA_DIR / "attention_data"
IMAGE_DATA_DIR   = DATA_DIR / "raw_images"
GRAPH_DATA_DIR   = DATA_DIR / "graphs"
LANGUAGE_INFO_DIR = DATA_DIR / "language_info"
OUTPUT_DIR       = DATA_DIR / "outputs"
REWRITE_LOG_DIR  = DATA_DIR / "rewrite_logs"

# Visualization output
VIZ_DIR          = _BASE / "visualizations"
PAPER_FIGURE_DIR = VIZ_DIR / "paper_figures"   # 900 DPI PNG outputs


# ---------------------------------------------------------------------------
# Token-sequence segmentation (matches ExplainVLA convention for pi0.5)
# ---------------------------------------------------------------------------
# pi0.5 token layout per inference step:
#   [  0 –  256)  image1 patches (SigLIP 256 patches)
#   [256 –  512)  image2 patches
#   [512 –  768)  image3 patches (wrist view, often disabled)
#   [768 –  768+T) instruction tokens  (T varies per task)
#   [768+T – end)  state tokens

IMAGE1_TOKENS:  Tuple[int, int] = (0,   256)
IMAGE2_TOKENS:  Tuple[int, int] = (256, 512)
IMAGE3_TOKENS:  Tuple[int, int] = (512, 768)  # optional / disabled
PATCH_GRID_SIZE: int = 16          # 16×16 = 256 patches per view

# Text / state ranges are computed dynamically from tokenised input.


# ---------------------------------------------------------------------------
# FAG construction parameters
# ---------------------------------------------------------------------------

FAG_LAYER_AGG:      str   = "mean"     # across-layer aggregation strategy
FAG_HEAD_MERGE:     str   = "mean"     # multi-head merging
FAG_EDGE_THRESHOLD: float = 1e-4       # prune edges below this weight
FAG_CROSS_BRIDGE:   bool  = True       # use hidden-state cosine-sim bridge
INCLUDE_IMAGE3:     bool  = False      # disable wrist-camera view in graphs

VLM_LAYER_COUNT:    int   = 18         # number of VLM transformer layers
EXPERT_LAYER_COUNT: int   = 10         # number of action-expert layers
ACTION_NUM:         int   = 50         # diffusion time steps

# ---------------------------------------------------------------------------
# Alignment scorer parameters
# ---------------------------------------------------------------------------

ALIGN_VTI_WEIGHT:           float = 0.4
ALIGN_TVI_WEIGHT:           float = 0.4
ALIGN_ENTROPY_PENALTY:      float = 0.1
ALIGN_CONCENTRATION_BONUS:  float = 0.1
ALIGN_REWRITE_THRESHOLD:    float = 0.25   # legacy absolute-threshold (kept for ablation Abl-5)

# Trend-based trigger parameters
LVCI_TREND_THRESHOLD:       float = 0.02   # ΔLVCI above this triggers rewrite (main signal)
VTI_PRIOR_THRESHOLD:        float = 0.75   # Layer 1: pre-emptive rewrite if VTI_0 exceeds this
VTI_CONFIRM_THRESHOLD:      float = 0.65   # Layer 3: strong vs light rewrite decision
LVCI_HISTORY_WINDOW:        int   = 2      # number of past checkpoints for trend (early vs late)

# Dual-condition trigger refinements
LVCI_FLOOR:                 float = 0.28   # minimum absolute LVCI before trend trigger fires
                                           # prevents false-positive rewrites when model is grounded
REWRITE_COOLDOWN_STEPS:     int   = 2      # checkpoints to skip after a rewrite (prevents cascade)
VTI_STRONG_THRESHOLD:       float = 0.85   # VTI above this uses LLM front-loading; below uses revert

# ---------------------------------------------------------------------------
# Instruction rewriter (LLM API)
# ---------------------------------------------------------------------------

# Provide OPENAI_API_KEY via your shell environment or a .env file (see .env.example).
# OPENAI_BASE_URL is optional — leave unset to use the official OpenAI endpoint, or
# set it to a compatible relay (e.g. Azure, an in-house gateway).
OPENAI_API_KEY:   str = os.getenv("OPENAI_API_KEY",   "")
OPENAI_BASE_URL:  str = os.getenv("OPENAI_BASE_URL",  "")
OPENAI_LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
REWRITE_MAX_RETRIES: int = 3

# ---------------------------------------------------------------------------
# Evaluation parameters
# ---------------------------------------------------------------------------

EVAL_TASK_IDS:        str = os.getenv("EVAL_TASK_IDS", "[0,1,2]")
EVAL_N_EPISODES:      int = int(os.getenv("EVAL_N_EPISODES", "5"))
EVAL_N_ACTION_STEPS:  int = int(os.getenv("EVAL_N_ACTION_STEPS", "10"))
EVAL_BATCH_SIZE:      int = 1
EVAL_LIBERO_SUITE:    str = os.getenv("EVAL_LIBERO_SUITE", "libero_object")

# ---------------------------------------------------------------------------
# Storage budget (reference only)
# ---------------------------------------------------------------------------
# Attention tensors per episode:  ~200 MB  (bfloat16, 18+10 layers × 50 steps)
# After FAG processing (graphs):  ~  5 MB  (sparse adjacency, pickled networkx)
# Paper figures (PNG 900 DPI):    ~  2 MB  each
# Total for 50 episodes:          ~10 GB   (attention) + negligible for graphs/figures


# ---------------------------------------------------------------------------
# Utility: ensure all output directories exist
# ---------------------------------------------------------------------------

def ensure_dirs():
    for d in [
        DATA_DIR, ATTN_DATA_DIR, IMAGE_DATA_DIR, GRAPH_DATA_DIR,
        LANGUAGE_INFO_DIR, OUTPUT_DIR, REWRITE_LOG_DIR,
        VIZ_DIR, PAPER_FIGURE_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
