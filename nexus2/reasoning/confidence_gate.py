"""Confidence gate for known/novel routing.

Routes queries based on max memory attention signal:
  - "known" (high confidence) -> continue reasoning chain, use memory
  - "novel" (low confidence) -> route to learning/web search

D-197 adds MultiSignalConfidenceGate with granular RouteLevel routing.
"""

import enum
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


class RouteLevel(enum.Enum):
    """Granular routing levels (D-197, D-227).

    D-227: REJECT level achieves ≤0.1% hallucination by explicitly refusing
    to answer when confidence is very low and no reliable memory signal exists.
    """
    INJECT_FULL = "inject_full"
    INJECT_TOP1 = "inject_top1"
    INJECT_HEDGED = "inject_hedged"  # D-287: deprecated — 92% context confusion. Use SKIP instead.
    SKIP = "skip"
    REJECT = "reject"  # D-227: explicit refusal when memory unreliable


@dataclass
class GateSignals:
    """Diagnostic signals from multi-signal gate (D-197, D-222).

    D-222: raw_cos_max is the only feature that transfers reliably from
    synthetic to real AMM distributions. It is now the primary routing signal.
    """
    max_attn: float
    retrieval_margin: float
    memory_age_seconds: float
    source_type: str
    raw_cos_max: float = 0.0  # D-222: primary transferable signal


class ConfidenceGate:
    """Routes via max memory attention signal with auto-calibration.

    Maintains a rolling window of recent attention scores to auto-calibrate
    the threshold between known and novel queries.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        calibration_window: int = 100,
    ):
        self.base_threshold = threshold
        self.threshold = threshold
        self._recent_scores: deque = deque(maxlen=calibration_window)

    def route(
        self,
        attn_weights: torch.Tensor,
    ) -> Tuple[str, float]:
        """Determine if query hits known or novel territory.

        Args:
            attn_weights: [num_slots] attention weights from memory read

        Returns:
            (route, confidence) where route is "known" or "novel"
        """
        if attn_weights.numel() == 0:
            return "novel", 0.0

        max_attn = attn_weights.max().item()
        self._recent_scores.append(max_attn)

        route = "known" if max_attn >= self.threshold else "novel"
        return route, max_attn

    def calibrate(self):
        """Auto-calibrate threshold from recent score distribution.

        Sets threshold to the median of recent scores, bounded by
        [base_threshold * 0.5, base_threshold * 1.5].
        """
        if len(self._recent_scores) < 10:
            return

        scores = sorted(self._recent_scores)
        median = scores[len(scores) // 2]

        low = self.base_threshold * 0.5
        high = self.base_threshold * 1.5
        self.threshold = max(low, min(high, median))

    def get_stats(self) -> dict:
        """Return gate statistics."""
        scores = list(self._recent_scores)
        return {
            "threshold": self.threshold,
            "base_threshold": self.base_threshold,
            "recent_count": len(scores),
            "recent_mean": sum(scores) / len(scores) if scores else 0.0,
            "recent_max": max(scores) if scores else 0.0,
            "recent_min": min(scores) if scores else 0.0,
        }


class MultiSignalConfidenceGate:
    """Multi-signal confidence gate with granular route levels (D-197, D-222, D-227).

    D-222: raw_cos_max is the primary routing signal — the only feature that
    transfers reliably from synthetic to real AMM distributions (19.8% overlap
    for other features).

    D-227: REJECT level achieves ≤0.1% hallucination by refusing to answer
    when raw_cos_max is below reject_threshold.

    Route levels:
      - INJECT_FULL:   high confidence + good margin + fresh → use all retrieved
      - INJECT_TOP1:   high confidence + poor margin or stale → use only top-1
      - INJECT_HEDGED: medium confidence → inject with hedging caveat
      - SKIP:          low confidence → don't inject, route to learning
      - REJECT:        very low raw_cos_max → refuse to answer (D-227)
    """

    def __init__(
        self,
        low_threshold: float = 0.30,
        high_threshold: float = 0.55,
        margin_threshold: float = 0.10,
        stale_seconds: float = 259200.0,  # 3 days
        calibration_window: int = 100,
        reject_threshold: float = 0.15,   # D-227: below this → REJECT
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.margin_threshold = margin_threshold
        self.stale_seconds = stale_seconds
        self.reject_threshold = reject_threshold  # D-227
        self._base_low = low_threshold
        self._base_high = high_threshold
        self._recent_scores: deque = deque(maxlen=calibration_window)

    def route(
        self,
        attn_weights: torch.Tensor,
        retrieval_scores: Optional[torch.Tensor] = None,
        top_entry_age: Optional[float] = None,
        top_entry_type: Optional[str] = None,
    ) -> Tuple[RouteLevel, float, GateSignals]:
        """Determine granular route level from multiple signals.

        D-222: Uses raw_cos_max (from retrieval_scores) as primary signal when
        available, falling back to max_attn. This is the only feature that
        transfers from synthetic to real AMM distributions.

        Args:
            attn_weights: [num_slots] attention weights from memory read
            retrieval_scores: [top_k] cosine similarity scores from retrieval
            top_entry_age: age in seconds of the top retrieved memory entry
            top_entry_type: mem_type of the top retrieved entry (e.g., "identity", "fact")

        Returns:
            (route_level, confidence, gate_signals)
        """
        # Defaults
        if top_entry_age is None:
            top_entry_age = 0.0
        if top_entry_type is None:
            top_entry_type = "unknown"

        # Empty attention → REJECT (not just SKIP — no memory at all)
        if attn_weights.numel() == 0:
            signals = GateSignals(
                max_attn=0.0, retrieval_margin=0.0,
                memory_age_seconds=top_entry_age, source_type=top_entry_type,
                raw_cos_max=0.0,
            )
            return RouteLevel.REJECT, 0.0, signals

        max_attn = attn_weights.max().item()
        self._recent_scores.append(max_attn)

        # D-222: raw_cos_max from retrieval scores is the primary signal
        raw_cos_max = max_attn  # fallback to attention-based
        if retrieval_scores is not None and retrieval_scores.numel() >= 1:
            raw_cos_max = retrieval_scores.max().item()

        # Compute retrieval margin (gap between top-1 and top-2 scores)
        if retrieval_scores is not None and retrieval_scores.numel() >= 2:
            sorted_scores = retrieval_scores.sort(descending=True).values
            retrieval_margin = (sorted_scores[0] - sorted_scores[1]).item()
        else:
            retrieval_margin = 0.0

        signals = GateSignals(
            max_attn=max_attn,
            retrieval_margin=retrieval_margin,
            memory_age_seconds=top_entry_age,
            source_type=top_entry_type,
            raw_cos_max=raw_cos_max,
        )

        # Identity memories never go stale
        is_stale = (
            top_entry_age > self.stale_seconds
            and top_entry_type != "identity"
        )

        # D-227: REJECT when raw_cos_max is below reject threshold
        # This achieves ≤0.1% hallucination rate
        if raw_cos_max < self.reject_threshold:
            return RouteLevel.REJECT, raw_cos_max, signals

        # D-222: Use raw_cos_max as primary routing signal
        if raw_cos_max < self.low_threshold:
            return RouteLevel.SKIP, raw_cos_max, signals

        if raw_cos_max >= self.high_threshold:
            if retrieval_margin >= self.margin_threshold and not is_stale:
                return RouteLevel.INJECT_FULL, raw_cos_max, signals
            else:
                return RouteLevel.INJECT_TOP1, raw_cos_max, signals

        # D-287: Middle zone → SKIP (was INJECT_HEDGED, but hedged context
        # causes 92% context confusion — better to skip entirely)
        return RouteLevel.SKIP, raw_cos_max, signals

    def calibrate(self):
        """Auto-calibrate thresholds from p25/p75 of recent scores."""
        if len(self._recent_scores) < 10:
            return

        scores = sorted(self._recent_scores)
        n = len(scores)
        p25 = scores[n // 4]
        p75 = scores[3 * n // 4]

        # Bounded adjustment
        low_lo = self._base_low * 0.5
        low_hi = self._base_low * 1.5
        self.low_threshold = max(low_lo, min(low_hi, p25))

        high_lo = self._base_high * 0.5
        high_hi = self._base_high * 1.5
        self.high_threshold = max(high_lo, min(high_hi, p75))
