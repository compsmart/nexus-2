"""Learned confidence gate from Phase 7 (real-data trained 3-feature LR).

Production-ready gate with 0.9% hallucination rate on real AMM episodes (D-223).
D-297: raw_cos_max + entropy_norm achieves 0.83% hallucination vs 0.92% with
all 3 features. raw_cos_margin is neutralized (fixed at scaler mean → 0 after
standardization).
"""

import numpy as np
import torch
from typing import Tuple, Optional

from .confidence_gate import RouteLevel, GateSignals


class LearnedConfidenceGate:
    """Real-data trained logistic regression gate (Phase 6c/7).

    Features: raw_cos_max, raw_cos_margin, entropy_norm
    Model: Logistic regression with class weighting
    Performance: 99.6% accuracy, 0.9% hallucination on 4,199 real episodes (D-223)
    """

    def __init__(self):
        """Initialize with pre-trained model parameters."""
        # StandardScaler parameters
        self.scaler_mean = np.array([0.7976748453833342, 0.06499770272520503, 0.3565955985493313], dtype=np.float32)
        self.scaler_scale = np.array([0.1781662134658833, 0.03247747648939904, 0.185840505173339], dtype=np.float32)

        # Logistic regression parameters
        self.lr_coeff = np.array([[-9.088564061086377, -0.398790855890882, 2.0944064191159213]], dtype=np.float32)  # [3, 2] for binary classification
        self.lr_intercept = np.array([0.385689659189148], dtype=np.float32)

    def _extract_features(self,
                         attn_weights: torch.Tensor,
                         retrieval_scores: Optional[torch.Tensor] = None,
                         entropy: float = 0.0) -> Tuple[np.ndarray, bool]:
        """Extract the 3 required features for the LR gate.

        Args:
            attn_weights: [num_slots] attention weights
            retrieval_scores: [top_k] raw cosine scores (for raw_cos_max, raw_cos_margin)
            entropy: entropy value from attention distribution

        Returns:
            (features_3d, valid) where features_3d is shape (3,) and valid indicates
            if all features could be extracted.
        """
        # Feature 1: raw_cos_max (from retrieval_scores if available, else attention)
        if retrieval_scores is not None and retrieval_scores.numel() >= 1:
            raw_cos_max = retrieval_scores.max().item()
        else:
            raw_cos_max = attn_weights.max().item() if attn_weights.numel() > 0 else 0.0

        # Feature 2: raw_cos_margin — D-297: neutralized by fixing at scaler mean.
        # After standardization this becomes 0.0, contributing nothing to the LR
        # decision. raw_cos_max + entropy_norm alone achieve 0.83% hallucination.
        raw_cos_margin = self.scaler_mean[1]  # fixed at training mean

        # Feature 3: entropy_norm (normalized entropy)
        entropy_norm = entropy  # already normalized in caller

        features = np.array([raw_cos_max, raw_cos_margin, entropy_norm], dtype=np.float32)
        valid = retrieval_scores is not None or attn_weights.numel() > 0

        return features, valid

    def route(self,
              attn_weights: torch.Tensor,
              retrieval_scores: Optional[torch.Tensor] = None,
              entropy: float = 0.0,
              top_entry_age: Optional[float] = None,
              top_entry_type: Optional[str] = None) -> Tuple[RouteLevel, float, GateSignals]:
        """Determine route using learned 3-feature LR gate.

        Args:
            attn_weights: [num_slots] attention weights from memory read
            retrieval_scores: [top_k] raw cosine similarity scores
            entropy: entropy of attention distribution (pre-normalized)
            top_entry_age: age in seconds of top entry
            top_entry_type: type of top entry

        Returns:
            (route_level, confidence, gate_signals)
        """
        # Defaults
        if top_entry_age is None:
            top_entry_age = 0.0
        if top_entry_type is None:
            top_entry_type = "unknown"

        # Extract 3 features
        features, valid = self._extract_features(attn_weights, retrieval_scores, entropy)

        # If no valid features, REJECT
        if not valid:
            signals = GateSignals(
                max_attn=0.0, retrieval_margin=0.0,
                memory_age_seconds=top_entry_age, source_type=top_entry_type,
                raw_cos_max=0.0,
            )
            return RouteLevel.REJECT, 0.0, signals

        # Standardize features
        features_scaled = (features - self.scaler_mean) / self.scaler_scale

        # Logistic regression: softmax over 2 classes
        # logits = X @ coeff.T + intercept
        logits = np.dot(features_scaled, self.lr_coeff.T) + self.lr_intercept

        # Softmax to get probabilities
        logits_exp = np.exp(logits - logits.max())  # numerical stability
        probs = logits_exp / logits_exp.sum()

        # Predict: 0 = CORRECT, 1 = NO_MEMORY
        pred = np.argmax(logits)
        confidence = probs.max()

        # Create signals (using features directly)
        signals = GateSignals(
            max_attn=features[0],  # raw_cos_max
            retrieval_margin=features[1],
            memory_age_seconds=top_entry_age,
            source_type=top_entry_type,
            raw_cos_max=features[0],
        )

        # Decision logic:
        # - If NO_MEMORY (pred=1): REJECT (0% hallucination)
        # - If CORRECT (pred=0): use raw_cos_max to decide inject level
        if pred == 1:
            # NO_MEMORY -> REJECT (per D-227)
            return RouteLevel.REJECT, confidence, signals

        # CORRECT: decide inject level based on raw_cos_max confidence
        raw_cos = features[0]
        entropy_norm = features[2]
        if raw_cos < 0.30:
            return RouteLevel.SKIP, confidence, signals
        elif raw_cos >= 0.55:
            # D-297: entropy-based FULL vs TOP1 (replaces margin-based)
            # Low entropy = confident retrieval → use all context
            # High entropy = ambiguous retrieval → use only top-1
            if entropy_norm < 0.5:
                return RouteLevel.INJECT_FULL, confidence, signals
            else:
                return RouteLevel.INJECT_TOP1, confidence, signals
        else:
            # D-287: SKIP instead of INJECT_HEDGED (92% context confusion)
            return RouteLevel.SKIP, confidence, signals
