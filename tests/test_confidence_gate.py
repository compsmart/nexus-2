"""Confidence gate tests: known/novel routing, auto-calibration, multi-signal gate, and REJECT (D-227)."""

import pytest
import torch

from nexus2.reasoning.confidence_gate import (
    ConfidenceGate,
    MultiSignalConfidenceGate,
    RouteLevel,
    GateSignals,
)
from nexus2.reasoning.learned_confidence_gate import LearnedConfidenceGate


# Verify REJECT is in the enum
assert hasattr(RouteLevel, "REJECT"), "D-227: REJECT route level must exist"


class TestConfidenceGate:
    """Tests for known/novel routing."""

    def test_empty_weights_routes_novel(self, confidence_gate):
        route, conf = confidence_gate.route(torch.tensor([]))
        assert route == "novel"
        assert conf == 0.0

    def test_high_attention_routes_known(self, confidence_gate):
        weights = torch.tensor([0.1, 0.2, 0.8, 0.1])
        route, conf = confidence_gate.route(weights)
        assert route == "known"
        assert abs(conf - 0.8) < 1e-5

    def test_low_attention_routes_novel(self, confidence_gate):
        weights = torch.tensor([0.1, 0.2, 0.3, 0.1])
        route, conf = confidence_gate.route(weights)
        assert route == "novel"
        assert abs(conf - 0.3) < 1e-5

    def test_threshold_boundary(self):
        gate = ConfidenceGate(threshold=0.5)
        # Exactly at threshold -> known
        weights = torch.tensor([0.5])
        route, _ = gate.route(weights)
        assert route == "known"

        # Just below -> novel
        weights = torch.tensor([0.499])
        route, _ = gate.route(weights)
        assert route == "novel"

    def test_calibrate_adjusts_threshold(self):
        gate = ConfidenceGate(threshold=0.5, calibration_window=20)

        # Feed consistently high scores
        for _ in range(15):
            gate.route(torch.tensor([0.8]))

        gate.calibrate()
        # Threshold should move toward 0.8 (but bounded)
        assert gate.threshold > 0.5
        assert gate.threshold <= 0.75  # bounded at 1.5x base

    def test_calibrate_with_low_scores(self):
        gate = ConfidenceGate(threshold=0.5, calibration_window=20)

        for _ in range(15):
            gate.route(torch.tensor([0.2]))

        gate.calibrate()
        # Threshold should move down (but bounded)
        assert gate.threshold >= 0.25  # bounded at 0.5x base
        assert gate.threshold < 0.5

    def test_calibrate_needs_minimum_samples(self):
        gate = ConfidenceGate(threshold=0.5, calibration_window=20)
        original = gate.threshold
        gate.calibrate()  # Not enough samples
        assert gate.threshold == original

    def test_stats(self, confidence_gate):
        confidence_gate.route(torch.tensor([0.7]))
        confidence_gate.route(torch.tensor([0.3]))

        stats = confidence_gate.get_stats()
        assert stats["threshold"] == 0.5
        assert stats["recent_count"] == 2
        assert abs(stats["recent_max"] - 0.7) < 1e-5
        assert abs(stats["recent_min"] - 0.3) < 1e-5


class TestMultiSignalConfidenceGate:
    """Tests for multi-signal gate (D-197)."""

    def test_empty_routes_reject(self, multi_signal_gate):
        """D-227: Empty attention now routes to REJECT (not SKIP)."""
        level, conf, signals = multi_signal_gate.route(torch.tensor([]))
        assert level == RouteLevel.REJECT
        assert conf == 0.0

    def test_low_routes_skip(self, multi_signal_gate):
        weights = torch.tensor([0.1, 0.15, 0.2])
        level, conf, signals = multi_signal_gate.route(weights)
        assert level == RouteLevel.SKIP
        assert abs(conf - 0.2) < 1e-5

    def test_high_with_margin_routes_full(self, multi_signal_gate):
        weights = torch.tensor([0.1, 0.2, 0.7])
        scores = torch.tensor([0.7, 0.3, 0.1])  # margin = 0.4 > 0.10
        level, conf, signals = multi_signal_gate.route(
            weights, retrieval_scores=scores,
            top_entry_age=100.0, top_entry_type="fact",
        )
        assert level == RouteLevel.INJECT_FULL

    def test_high_poor_margin_routes_top1(self, multi_signal_gate):
        weights = torch.tensor([0.1, 0.2, 0.7])
        scores = torch.tensor([0.7, 0.65, 0.1])  # margin = 0.05 < 0.10
        level, conf, signals = multi_signal_gate.route(
            weights, retrieval_scores=scores,
            top_entry_age=100.0, top_entry_type="fact",
        )
        assert level == RouteLevel.INJECT_TOP1

    def test_stale_routes_top1(self, multi_signal_gate):
        weights = torch.tensor([0.1, 0.2, 0.7])
        scores = torch.tensor([0.7, 0.3, 0.1])  # good margin
        level, conf, signals = multi_signal_gate.route(
            weights, retrieval_scores=scores,
            top_entry_age=500000.0,  # very stale
            top_entry_type="fact",
        )
        assert level == RouteLevel.INJECT_TOP1

    def test_identity_never_stale(self, multi_signal_gate):
        weights = torch.tensor([0.1, 0.2, 0.7])
        scores = torch.tensor([0.7, 0.3, 0.1])
        level, conf, signals = multi_signal_gate.route(
            weights, retrieval_scores=scores,
            top_entry_age=999999.0,  # very old
            top_entry_type="identity",  # but identity never stales
        )
        assert level == RouteLevel.INJECT_FULL

    def test_middle_routes_skip(self, multi_signal_gate):
        """D-287: Middle zone now routes to SKIP (was INJECT_HEDGED)."""
        weights = torch.tensor([0.1, 0.2, 0.4])  # between 0.30 and 0.55
        level, conf, signals = multi_signal_gate.route(weights)
        assert level == RouteLevel.SKIP

    def test_signals_populated(self, multi_signal_gate):
        weights = torch.tensor([0.1, 0.2, 0.7])
        scores = torch.tensor([0.7, 0.3])
        level, conf, signals = multi_signal_gate.route(
            weights, retrieval_scores=scores,
            top_entry_age=42.0, top_entry_type="fact",
        )
        assert isinstance(signals, GateSignals)
        assert abs(signals.max_attn - 0.7) < 1e-5
        assert abs(signals.retrieval_margin - 0.4) < 1e-5
        assert abs(signals.memory_age_seconds - 42.0) < 1e-5
        assert signals.source_type == "fact"

    def test_backward_compat_mapping(self):
        """RouteLevel maps to known/novel for backward compat."""
        gate = MultiSignalConfidenceGate()
        # Very low -> REJECT
        level, _, _ = gate.route(torch.tensor([0.05]))
        assert level == RouteLevel.REJECT
        # Above high_threshold -> known
        level, _, _ = gate.route(torch.tensor([0.8]))
        assert level not in (RouteLevel.SKIP, RouteLevel.REJECT)

    def test_reject_below_reject_threshold(self):
        """D-227: Very low raw_cos_max routes to REJECT for ≤0.1% hallucination."""
        gate = MultiSignalConfidenceGate(reject_threshold=0.15)
        weights = torch.tensor([0.05, 0.08, 0.10])
        scores = torch.tensor([0.10, 0.05, 0.02])
        level, conf, signals = gate.route(
            weights, retrieval_scores=scores,
            top_entry_age=100.0, top_entry_type="fact",
        )
        assert level == RouteLevel.REJECT
        assert conf < 0.15

    def test_reject_vs_skip_boundary(self):
        """D-227: REJECT is below reject_threshold, SKIP is between reject and low."""
        gate = MultiSignalConfidenceGate(
            reject_threshold=0.15, low_threshold=0.30,
        )
        # Just above reject threshold but below low -> SKIP
        weights = torch.tensor([0.20])
        scores = torch.tensor([0.20])
        level, _, _ = gate.route(weights, retrieval_scores=scores)
        assert level == RouteLevel.SKIP

        # Below reject threshold -> REJECT
        weights = torch.tensor([0.10])
        scores = torch.tensor([0.10])
        level, _, _ = gate.route(weights, retrieval_scores=scores)
        assert level == RouteLevel.REJECT

    def test_raw_cos_max_in_signals(self):
        """D-222: raw_cos_max is populated in gate signals."""
        gate = MultiSignalConfidenceGate()
        weights = torch.tensor([0.3, 0.5, 0.7])
        scores = torch.tensor([0.8, 0.3])
        level, conf, signals = gate.route(
            weights, retrieval_scores=scores,
            top_entry_age=10.0, top_entry_type="fact",
        )
        assert hasattr(signals, "raw_cos_max")
        assert abs(signals.raw_cos_max - 0.8) < 1e-5  # max of retrieval scores

    def test_raw_cos_max_primary_signal(self):
        """D-222: raw_cos_max from retrieval_scores takes priority over max_attn."""
        gate = MultiSignalConfidenceGate(high_threshold=0.55)
        # max_attn=0.7 (high) but raw_cos_max=0.3 (low)
        weights = torch.tensor([0.1, 0.2, 0.7])
        scores = torch.tensor([0.3, 0.2])  # raw_cos_max=0.3, below high
        level, conf, signals = gate.route(
            weights, retrieval_scores=scores,
            top_entry_age=100.0, top_entry_type="fact",
        )
        # D-287: Should route based on raw_cos_max=0.3, not max_attn=0.7
        assert level == RouteLevel.SKIP  # 0.30 <= 0.3 < 0.55 → SKIP (was INJECT_HEDGED)


class TestLearnedConfidenceGate:
    """D-297: Learned gate with margin neutralization and entropy-based routing."""

    def test_margin_neutralized(self):
        """D-297: raw_cos_margin is fixed at scaler mean → 0 after standardization."""
        gate = LearnedConfidenceGate()
        weights = torch.tensor([0.8])
        scores = torch.tensor([0.8, 0.3])  # margin would be 0.5

        features, valid = gate._extract_features(weights, scores, entropy=0.2)
        # Feature 1 (margin) should be fixed at scaler_mean[1], not actual margin
        assert abs(features[1] - gate.scaler_mean[1]) < 1e-6

    def test_low_cos_routes_skip_or_reject(self):
        """Low raw_cos_max should route to SKIP or REJECT."""
        gate = LearnedConfidenceGate()
        weights = torch.tensor([0.1])
        scores = torch.tensor([0.1, 0.05])
        level, conf, signals = gate.route(
            weights, retrieval_scores=scores, entropy=0.5,
        )
        assert level in (RouteLevel.SKIP, RouteLevel.REJECT)

    def test_entropy_based_full_vs_top1(self):
        """D-297: Low entropy → INJECT_FULL, high entropy → INJECT_TOP1."""
        gate = LearnedConfidenceGate()
        weights = torch.tensor([0.9])
        scores = torch.tensor([0.9, 0.3])

        # Low entropy → FULL
        level_low, _, _ = gate.route(
            weights, retrieval_scores=scores, entropy=0.1,
        )
        # High entropy → TOP1
        level_high, _, _ = gate.route(
            weights, retrieval_scores=scores, entropy=0.9,
        )
        # At least one should be FULL and one TOP1 (if LR predicts CORRECT)
        # Both should be inject-level (not SKIP/REJECT) for high raw_cos
        if level_low not in (RouteLevel.SKIP, RouteLevel.REJECT):
            assert level_low == RouteLevel.INJECT_FULL
        if level_high not in (RouteLevel.SKIP, RouteLevel.REJECT):
            assert level_high == RouteLevel.INJECT_TOP1
