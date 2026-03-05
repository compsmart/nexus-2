"""Curriculum engine for k-scaling, mixed-K, and hop-depth progression.

Manages the training curriculum:
  - k-schedule: [5, 10, 20, 50, 100, 200, 350, 500]
  - mixed-K: random k sampling for cross-scale regularization (D-183)
  - hop-schedule: [2, 3, 4, 5]
  - Advance when accuracy >= convergence_threshold (99%)

D-183: Mixed-K training achieves 15/15 ≥95% — universal cross-scale adapter.
Training on randomly sampled k values after sequential k-scaling provides
robust generalization across memory bank sizes.

ANTI-PATTERN: NEVER skip curriculum for k > 15.
ANTI-PATTERN: NEVER train LSTM at k > 500.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CurriculumState:
    """Tracks current position in the curriculum."""
    k_stage: int = 0           # index into k_schedule
    hop_stage: int = 0         # index into hop_schedule
    epoch: int = 0             # current epoch within stage
    best_accuracy: float = 0.0
    phase: str = "k_scaling"   # "k_scaling" | "mixed_k" | "hop_depth" | "done"
    mixed_k_epoch: int = 0     # D-183: epoch counter for mixed-K phase


class CurriculumEngine:
    """Manages training curriculum progression.

    D-183: After sequential k-scaling, a mixed-K phase trains on randomly
    sampled k values from the full schedule. This universal cross-scale
    regularizer achieves 15/15 ≥95% across all k values.
    """

    def __init__(
        self,
        k_schedule: Optional[List[int]] = None,
        hop_schedule: Optional[List[int]] = None,
        convergence_threshold: float = 0.99,
        max_epochs_per_stage: int = 500,
        encoder_type: str = "lstm",
        mixed_k_epochs: int = 100,  # D-183: epochs for mixed-K phase
        mixed_k_enabled: bool = True,  # D-183: enable mixed-K regularization
    ):
        self.encoder_type = encoder_type
        k_sched = k_schedule or [5, 10, 20, 50, 100, 200, 350, 500, 750, 1000]
        # D-188: LSTM must be capped at k=500; Mamba can go higher
        if encoder_type == "lstm":
            k_sched = [k for k in k_sched if k <= 500]
        self.k_schedule = k_sched
        self.hop_schedule = hop_schedule or [2, 3, 4, 5]
        self.convergence_threshold = convergence_threshold
        self.max_epochs_per_stage = max_epochs_per_stage
        self.mixed_k_epochs = mixed_k_epochs
        self.mixed_k_enabled = mixed_k_enabled
        self._rng = random.Random(42)
        self.state = CurriculumState()

    @property
    def current_k(self) -> int:
        if self.state.k_stage < len(self.k_schedule):
            return self.k_schedule[self.state.k_stage]
        return self.k_schedule[-1]

    @property
    def current_hops(self) -> int:
        if self.state.hop_stage < len(self.hop_schedule):
            return self.hop_schedule[self.state.hop_stage]
        return self.hop_schedule[-1]

    @property
    def is_done(self) -> bool:
        return self.state.phase == "done"

    @property
    def in_mixed_k(self) -> bool:
        """True when in the mixed-K regularization phase (D-183)."""
        return self.state.phase == "mixed_k"

    def sample_mixed_k(self) -> int:
        """Sample a random k from the schedule (D-183).

        Used during the mixed-K phase to train on randomly sampled k values,
        providing cross-scale regularization.
        """
        return self._rng.choice(self.k_schedule)

    def step(self, accuracy: float) -> bool:
        """Report accuracy for current stage. Returns True if stage advanced.

        Args:
            accuracy: accuracy on current stage's eval set

        Returns:
            True if curriculum advanced to next stage.
        """
        self.state.epoch += 1
        self.state.best_accuracy = max(self.state.best_accuracy, accuracy)

        # D-183: Mixed-K phase advances on epoch count, not convergence
        if self.state.phase == "mixed_k":
            self.state.mixed_k_epoch += 1
            if self.state.mixed_k_epoch >= self.mixed_k_epochs:
                return self._advance()
            return False

        # Check convergence
        converged = accuracy >= self.convergence_threshold
        max_reached = self.state.epoch >= self.max_epochs_per_stage

        if not converged and not max_reached:
            return False

        # Advance to next stage
        return self._advance()

    def _advance(self) -> bool:
        """Advance to next curriculum stage. Returns True if advanced."""
        if self.state.phase == "k_scaling":
            self.state.k_stage += 1
            if self.state.k_stage >= len(self.k_schedule):
                # D-183: Insert mixed-K phase between k-scaling and hop depth
                if self.mixed_k_enabled:
                    self.state.phase = "mixed_k"
                    self.state.mixed_k_epoch = 0
                else:
                    self.state.phase = "hop_depth"
                    self.state.hop_stage = 0
            self.state.epoch = 0
            self.state.best_accuracy = 0.0
            return True

        elif self.state.phase == "mixed_k":
            # D-183: Mixed-K complete, move to hop depth
            self.state.phase = "hop_depth"
            self.state.hop_stage = 0
            self.state.epoch = 0
            self.state.best_accuracy = 0.0
            return True

        elif self.state.phase == "hop_depth":
            self.state.hop_stage += 1
            if self.state.hop_stage >= len(self.hop_schedule):
                self.state.phase = "done"
            self.state.epoch = 0
            self.state.best_accuracy = 0.0
            return True

        return False

    def get_status(self) -> str:
        """Human-readable status string."""
        if self.state.phase == "k_scaling":
            return (
                f"k-scaling: k={self.current_k} "
                f"(stage {self.state.k_stage + 1}/{len(self.k_schedule)}) "
                f"epoch={self.state.epoch} best={self.state.best_accuracy:.4f}"
            )
        elif self.state.phase == "mixed_k":
            return (
                f"mixed-K (D-183): epoch={self.state.mixed_k_epoch}/{self.mixed_k_epochs} "
                f"best={self.state.best_accuracy:.4f}"
            )
        elif self.state.phase == "hop_depth":
            return (
                f"hop-depth: hops={self.current_hops} "
                f"(stage {self.state.hop_stage + 1}/{len(self.hop_schedule)}) "
                f"epoch={self.state.epoch} best={self.state.best_accuracy:.4f}"
            )
        else:
            return "Curriculum complete."

    def reset(self):
        """Reset curriculum to beginning."""
        self.state = CurriculumState()
