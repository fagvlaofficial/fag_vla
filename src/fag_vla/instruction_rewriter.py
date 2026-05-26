"""
Instruction Rewriter

Uses GPT-4o to generate attention-informed instruction rewrites.

Given:
  - original_instruction: str
  - token_weights: np.ndarray  — per-token alignment importance
  - token_texts: List[str]     — decoded token strings
  - diagnostics: dict          — alignment score breakdown
  - scene_description: str     — optional scene caption

The rewriter constructs a structured prompt that highlights which parts of
the instruction receive high vs. low visual attention, then asks GPT-4o to
produce a clearer, more grounded instruction that should improve alignment.

API credentials are read from environment variables (see settings.py).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RewriteResult:
    original: str
    rewritten: str
    alignment_before: float
    model_used: str
    token_analysis: str    # human-readable token weight summary passed to LLM
    raw_response: str      # full LLM response for logging
    rewrite_strength: str = "light"   # "light" | "medium" | "strong"
    lvci_trend: float = 0.0           # ΔLVCI that triggered this rewrite
    vti_signal: float = 0.0           # VTI value at trigger time


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _extract_allowed_vocab(instruction: str) -> set:
    """
    Extract the set of lowercase alphabetic words from the original instruction.
    These are the ONLY content words allowed in any rewrite.
    """
    import re
    return set(re.findall(r"[a-z]+", instruction.lower()))


# Safe function words that may appear in rewrites even if not in the original.
_SAFE_FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "it", "its",
    "in", "on", "at", "by", "for", "up", "now", "then",
}


def validate_rewrite(rewrite: str, true_original: str) -> bool:
    """
    Return True if the rewrite does not introduce out-of-distribution words.
    OOD = words not in the true_original AND not in the safe function word list.
    """
    import re
    allowed = _extract_allowed_vocab(true_original) | _SAFE_FUNCTION_WORDS
    rewrite_words = set(re.findall(r"[a-z]+", rewrite.lower()))
    ood = rewrite_words - allowed
    if ood:
        logger.debug(f"Rewrite rejected — OOD words: {ood}")
        return False
    return True


def _build_rewrite_prompt(
    true_original: str,
    current_instruction: str,
    token_texts: List[str],
    token_weights: np.ndarray,
    alignment_score: float,
    lvci_trend: float = 0.0,
    vti_signal: float = 0.0,
    rewrite_strength: str = "medium",
) -> str:
    """
    Build a prompt targeting reduced Language-Visual Coupling (LV-Coupling).

    Goal: help the robot model ground the instruction into visual features
    so it stops cross-referencing language and acts on visual cues.

    rewrite_strength controls how much visual specificity to add:
      "light"  — word-order adjustment only (low VTI, model mostly grounded)
      "medium" — front-load key object, add one concrete descriptor
      "strong" — add object color/shape/spatial position (high VTI, model lost)
    """
    k = min(5, len(token_weights))
    if len(token_weights) > 0:
        top_idx    = np.argsort(token_weights)[-k:][::-1]
        bottom_idx = np.argsort(token_weights)[:k]
        high_focus = ", ".join(
            f'"{token_texts[i]}"'
            for i in top_idx if i < len(token_texts) and token_texts[i].strip()
        )
        low_focus = ", ".join(
            f'"{token_texts[i]}"'
            for i in bottom_idx if i < len(token_texts) and token_texts[i].strip()
        )
    else:
        high_focus = low_focus = "(none)"

    allowed_words = sorted(_extract_allowed_vocab(true_original))
    vocab_line = ", ".join(f'"{w}"' for w in allowed_words)

    drift_note = ""
    if current_instruction.strip().lower() != true_original.strip().lower():
        drift_note = (
            f'\nNote: the robot is currently following a modified version:\n'
            f'    CURRENT: "{current_instruction}"\n'
            f'Prefer to return the ORIGINAL unless you can make a strictly better rephrasing.'
        )

    # Strength-specific instructions
    if rewrite_strength == "strong":
        goal_line = (
            "The model keeps re-reading the instruction instead of acting on visual cues "
            f"(language-visual coupling is HIGH: trend={lvci_trend:+.3f}, VTI={vti_signal:.3f}). "
            "Front-load the TARGET OBJECT as the very first word. "
            "Make the action as direct and unambiguous as possible."
        )
    elif rewrite_strength == "medium":
        goal_line = (
            "The model is still referencing language too much during execution "
            f"(coupling trend={lvci_trend:+.3f}). "
            "Reorder so the most visually identifiable element comes first. "
            "Make the instruction more action-direct."
        )
    else:  # light
        goal_line = (
            "Minor adjustment: the model needs slight refocusing on the key action. "
            "Adjust word order only — no content changes."
        )

    prompt = f"""A robot manipulation model is executing an instruction but keeps cross-referencing language instead of acting on visual cues.

ORIGINAL instruction (never add words beyond this):
    "{true_original}"{drift_note}

Attention analysis:
  - Words the model attends to MOST (likely already grounded): {high_focus}
  - Words the model attends to LEAST (visually un-grounded, need to be more prominent): {low_focus}

{goal_line}

STRICT RULES — violating any rule means you must return the ORIGINAL unchanged:
  1. Use ONLY words from this allowed vocabulary: {vocab_line}
  2. Do NOT add spatial information not in the original
  3. Do NOT add objects not mentioned in the original
  4. Do NOT add adverbs
  5. Do NOT change the core action verb
  6. The rewrite must be the same length or shorter than the ORIGINAL
  7. If no valid improvement exists, return the ORIGINAL exactly

Respond with ONLY the instruction text, nothing else."""

    return prompt


# ---------------------------------------------------------------------------
# Rewriter class
# ---------------------------------------------------------------------------

class InstructionRewriter:
    """
    LLM-based instruction rewriter powered by GPT-4o.

    Parameters
    ----------
    api_key:      OpenAI-compatible API key (defaults to env OPENAI_API_KEY)
    base_url:     API base URL (defaults to env OPENAI_BASE_URL)
    model:        LLM model name (defaults to env OPENAI_LLM_MODEL or 'gpt-4o')
    max_retries:  Number of API call retries on failure
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
    ):
        self.api_key    = api_key    or os.getenv("OPENAI_API_KEY",   "")
        self.base_url   = base_url   or os.getenv("OPENAI_BASE_URL",  "https://api.openai.com/v1")
        self.model      = model      or os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        """Lazy-init OpenAI client to avoid import-time side effects."""
        if self._client is None:
            try:
                from openai import OpenAI
                import os
                # Remove SOCKS proxy which httpx doesn't support natively
                for _k in ["ALL_PROXY", "all_proxy"]:
                    os.environ.pop(_k, None)
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError(
                    "openai package required for instruction rewriting. "
                    "Install with: pip install openai"
                )
        return self._client

    def rewrite(
        self,
        original: str,
        token_texts: List[str],
        token_weights: np.ndarray,
        alignment_score: float,
        true_original: Optional[str] = None,
        temperature: float = 0.3,
        lvci_trend: float = 0.0,
        vti_signal: float = 0.0,
        rewrite_strength: str = "medium",
    ) -> RewriteResult:
        """
        Generate a visually-grounding instruction rewrite via LLM.

        Parameters
        ----------
        original:          current instruction the robot is following
        token_texts:       decoded token strings
        token_weights:     per-token LVCI importance scores
        alignment_score:   overall LVCI from scorer (for logging)
        true_original:     episode's first instruction (ground truth vocabulary)
        lvci_trend:        ΔLVCI that triggered this rewrite (for prompt context)
        vti_signal:        VTI sub-score (determines rewrite strength)
        rewrite_strength:  "light" | "medium" | "strong" (controls prompt aggressiveness)
        """
        ground_truth = true_original if true_original else original

        prompt = _build_rewrite_prompt(
            true_original=ground_truth,
            current_instruction=original,
            token_texts=token_texts,
            token_weights=token_weights,
            alignment_score=alignment_score,
            lvci_trend=lvci_trend,
            vti_signal=vti_signal,
            rewrite_strength=rewrite_strength,
        )

        client = self._get_client()
        last_error = None
        raw_response = ""

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a strict robot instruction editor. "
                                "You only rearrange or emphasize existing words — "
                                "you NEVER introduce new information."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=60,
                )
                raw_response = response.choices[0].message.content or ""
                candidate = raw_response.strip().strip('"').strip("'").rstrip(".")

                if not candidate:
                    continue

                # Post-hoc OOD validation: reject if new content words appear
                if validate_rewrite(candidate, ground_truth):
                    rewritten = candidate
                else:
                    logger.info(
                        f"Rewrite rejected (OOD): '{candidate}' → falling back to true_original"
                    )
                    rewritten = ground_truth  # reset to true original, not current

                return RewriteResult(
                    original=original,
                    rewritten=rewritten,
                    alignment_before=alignment_score,
                    model_used=self.model,
                    token_analysis=_build_token_summary(token_texts, token_weights),
                    raw_response=raw_response,
                    rewrite_strength=rewrite_strength,
                    lvci_trend=lvci_trend,
                    vti_signal=vti_signal,
                )
            except Exception as e:
                last_error = e
                logger.warning(f"Rewrite attempt {attempt+1} failed: {e}")

        logger.error(f"All rewrite attempts failed. Last error: {last_error}")
        return RewriteResult(
            original=original,
            rewritten=ground_truth,
            alignment_before=alignment_score,
            model_used=self.model,
            token_analysis="",
            raw_response=str(last_error),
            rewrite_strength=rewrite_strength,
            lvci_trend=lvci_trend,
            vti_signal=vti_signal,
        )

    def rewrite_batch(
        self,
        instructions: List[Tuple[str, List[str], np.ndarray, float]],
        true_originals: Optional[List[Optional[str]]] = None,
    ) -> List[RewriteResult]:
        """
        Rewrite a batch of instructions sequentially.

        instructions:   list of (current_instruction, token_texts, token_weights, alignment_score)
        true_originals: list of episode true-original instructions (same length).
                        Passed to the rewriter to prevent OOD drift.
        """
        if true_originals is None:
            true_originals = [None] * len(instructions)

        results = []
        for (orig, tokens, weights, score), true_orig in zip(instructions, true_originals):
            result = self.rewrite(orig, tokens, weights, score, true_original=true_orig)
            results.append(result)
            logger.info(
                f"Rewrite: '{orig}' → '{result.rewritten}' (alignment: {score:.3f})"
            )
        return results


def _build_token_summary(token_texts: List[str], token_weights: np.ndarray) -> str:
    """Build a readable token weight summary string."""
    pairs = list(zip(token_texts, token_weights.tolist()))
    pairs_sorted = sorted(pairs, key=lambda x: -x[1])
    return "; ".join(f"{t}:{w:.3f}" for t, w in pairs_sorted[:10])
