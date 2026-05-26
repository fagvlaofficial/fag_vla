"""
Attention Alignment Scorer

Quantifies how well the current instruction aligns with visual information
by analyzing text-to-vision and vision-to-text attention patterns in the FAG.

Alignment score = weighted combination of:
  1. VTI (Vision→Text Intensity): mean attention from image patches to text tokens
  2. TVI (Text→Vision Intensity): mean attention from text tokens to image patches
  3. Entropy penalty: high entropy → diffuse attention → lower alignment
  4. Concentration bonus: high max-weight patches → focused attention → higher alignment

Output
------
AlignmentResult dataclass with:
  - score:          float in [0, 1] — higher is better aligned
  - token_weights:  per text-token importance (for rewriter context)
  - patch_weights:  per image-patch importance (for visualization)
  - diagnostics:    dict of sub-scores for analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from fag_vla.fused_attention_graph import FusedAttentionGraph

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    score: float                          # overall alignment score [0,1]
    token_weights: np.ndarray             # (T,) per text-token weight
    patch_weights_view1: np.ndarray       # (256,) per image1-patch weight
    patch_weights_view2: np.ndarray       # (256,) per image2-patch weight
    diagnostics: Dict[str, float] = field(default_factory=dict)


class AttentionAlignmentScorer:
    """
    Computes instruction–vision alignment scores from a built FusedAttentionGraph.

    Parameters
    ----------
    fag : FusedAttentionGraph
        Configured FAG instance (holds segment metadata).
    vti_weight : float
        Weight of vision-to-text attention in final score.
    tvi_weight : float
        Weight of text-to-vision attention in final score.
    entropy_penalty : float
        Penalty factor for diffuse (high-entropy) attention.
    concentration_bonus : float
        Bonus factor for focused (high max-weight) attention.
    """

    def __init__(
        self,
        fag: FusedAttentionGraph,
        vti_weight: float = 0.4,
        tvi_weight: float = 0.4,
        entropy_penalty: float = 0.1,
        concentration_bonus: float = 0.1,
    ):
        self.fag = fag
        self.vti_weight = vti_weight
        self.tvi_weight = tvi_weight
        self.entropy_penalty = entropy_penalty
        self.concentration_bonus = concentration_bonus

    # ------------------------------------------------------------------
    def score(self, G) -> AlignmentResult:
        """
        Compute alignment score from a built FAG (nx.DiGraph).

        Scoring uses raw (non-max-normalised) adjacency values, so that row sums
        represent the true fraction of softmax attention directed at a modality.
        For a sequence of length S and T text tokens, each image patch's expected
        attention to text is T/S ≈ 14/968 ≈ 0.014.  We compare actual vs. expected
        to get a meaningful signal.

        Returns AlignmentResult.
        """
        # ---- Raw (unnormalised) adjacency matrices ----
        vti_1 = self.fag.to_adjacency_matrix(G, "image1", "text", normalize=False)
        vti_2 = self.fag.to_adjacency_matrix(G, "image2", "text", normalize=False)
        tvi_1 = self.fag.to_adjacency_matrix(G, "text", "image1", normalize=False)
        tvi_2 = self.fag.to_adjacency_matrix(G, "text", "image2", normalize=False)

        # ---- Fraction of attention directed to the other modality (row-sum) ----
        # vti: each image patch's total attention going to text tokens  → mean over patches
        vti_frac_1 = vti_1.sum(1).mean() if vti_1.size > 0 else 0.0
        vti_frac_2 = vti_2.sum(1).mean() if vti_2.size > 0 else 0.0
        vti_score  = float((vti_frac_1 + vti_frac_2) / 2.0)

        # tvi: each text token's total attention going to image patches → mean over tokens
        tvi_frac_1 = tvi_1.sum(1).mean() if tvi_1.size > 0 else 0.0
        tvi_frac_2 = tvi_2.sum(1).mean() if tvi_2.size > 0 else 0.0
        tvi_score  = float((tvi_frac_1 + tvi_frac_2) / 2.0)

        # ---- Per-token importance: how much vision attends to each text token ----
        # Column-sum of vti: how much total image attention lands on each text token
        tok_w_vti = vti_1.sum(0) + vti_2.sum(0)   # (T,)
        tok_w_tvi = tvi_1.sum(1) + tvi_2.sum(1)   # (T,)  text→image row sums
        token_weights = self._safe_normalize(tok_w_vti + tok_w_tvi)

        # ---- Per-patch importance ----
        # How much each patch is attended to (from text) + how much it attends to text
        patch_w1 = self._safe_normalize(tvi_1.sum(0) + vti_1.sum(1))  # (P,)
        patch_w2 = self._safe_normalize(tvi_2.sum(0) + vti_2.sum(1))

        # ---- Composite score ----
        # Scale vti/tvi by expected uniform baseline: T/S ≈ 14/968.
        # Expected ≈ 0.014; ratio > 1 means above-chance alignment.
        n_seq = 968.0  # total VLM sequence length
        n_text = float(max(vti_1.shape[1], 1))
        n_img  = float(max(tvi_1.shape[1], 1))
        baseline_vti = n_text / n_seq
        baseline_tvi = n_img  / n_seq

        # Normalized above-chance signal, clipped to [0, 1]
        vti_signal = float(np.clip(vti_score / (baseline_vti + 1e-12) / 10.0, 0, 1))
        tvi_signal = float(np.clip(tvi_score / (baseline_tvi + 1e-12) / 10.0, 0, 1))

        # Concentration: entropy of token weight distribution
        n_tok = max(len(token_weights), 1)
        tok_entropy_norm = self._entropy(token_weights) / max(np.log(n_tok), 1e-12)
        max_normed = float(token_weights.max()) if token_weights.size > 0 else 0.0

        base = self.vti_weight * vti_signal + self.tvi_weight * tvi_signal
        score = base \
              - self.entropy_penalty * tok_entropy_norm \
              + self.concentration_bonus * max_normed
        score = float(np.clip(score, 0.0, 1.0))

        diagnostics = {
            "vti_score":          vti_score,         # raw fraction (~0.001–0.05)
            "tvi_score":          tvi_score,
            "vti_signal":         vti_signal,         # above-chance normalised
            "tvi_signal":         tvi_signal,
            "token_entropy_norm": float(tok_entropy_norm),
            "max_token_w":        max_normed,
            "composite":          score,
        }

        return AlignmentResult(
            score=score,
            token_weights=token_weights,
            patch_weights_view1=patch_w1,
            patch_weights_view2=patch_w2,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_normalize(arr: np.ndarray) -> np.ndarray:
        s = arr.sum()
        return arr / (s + 1e-12) if s > 0 else arr

    @staticmethod
    def _entropy(prob: np.ndarray) -> float:
        p = prob + 1e-12
        p = p / p.sum()
        return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# Batch scorer for episode-level analysis
# ---------------------------------------------------------------------------

def score_episode(
    fag: FusedAttentionGraph,
    tracer,                    # AttentionTracer instance
    steps: Optional[List[int]] = None,
    timestep: int = 0,         # which diffusion timestep to use
) -> Dict[int, AlignmentResult]:
    """
    Score all steps of an episode.

    Returns {step: AlignmentResult}.
    """
    scorer = AttentionAlignmentScorer(fag)
    results: Dict[int, AlignmentResult] = {}

    if steps is None:
        steps = sorted(tracer.language_attn.keys())

    for step in steps:
        vlm_attn  = tracer.language_attn.get(step, {})
        exp_attn  = tracer.expert_attn.get(step, {}).get(timestep, {})
        if not vlm_attn:
            continue
        G = fag.build(vlm_attn, exp_attn)
        results[step] = scorer.score(G)

    return results
