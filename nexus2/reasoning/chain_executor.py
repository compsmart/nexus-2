"""Chain executor -- orchestrates multi-hop reasoning with confidence gating.

Combines the N-hop reader with the confidence gate to produce
a complete reasoning result from a query and memory state.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import NexusConfig
from .confidence_gate import ConfidenceGate, MultiSignalConfidenceGate
from .learned_confidence_gate import LearnedConfidenceGate
from .nhop_reader import ExplicitNReadChain, SharedNReadChain


@dataclass
class ReasoningResult:
    """Output of the reasoning chain."""
    retrieval_vectors: torch.Tensor  # [d_val] final state from chain
    route: str                       # "known" or "novel"
    confidence: float                # max attention score
    intermediate_logits: List[torch.Tensor] = field(default_factory=list)
    attention_weights: List[torch.Tensor] = field(default_factory=list)
    n_hops_used: int = 0
    route_level: str = "skip"        # D-197: granular route level
    gate_signals: object = None      # D-197: GateSignals dataclass


class ChainExecutor(nn.Module):
    """Orchestrates query encoding, hop chain, and confidence gating.

    Encodes a query vector, runs it through the N-hop reader against
    the memory bank's keys/values, and routes via confidence gate.
    """

    def __init__(self, config: NexusConfig, n_entities: int = 2000):
        super().__init__()
        self.config = config
        self.n_entities = n_entities

        # Query projection from value space
        self.query_proj = nn.Linear(config.d_val, config.d_val)

        # N-hop reader (explicit per-hop projections)
        self.reader = ExplicitNReadChain(
            d_key=config.d_key,
            d_val=config.d_val,
            n_hops=config.max_hops,
            n_entities=n_entities,
            intermediate_supervision=True,
        )

        # Optional shared reader for variable-depth
        self.shared_reader = SharedNReadChain(
            d_key=config.d_key,
            d_val=config.d_val,
            max_hops=config.max_hops,
            n_entities=n_entities,
        )

        # Legacy single-signal gate (backward compat)
        self.gate = ConfidenceGate(
            threshold=config.confidence_threshold,
            calibration_window=config.gate_calibration_window,
        )

        # D-197/D-222/D-227: Multi-signal confidence gate with REJECT level
        self.multi_gate = MultiSignalConfidenceGate(
            low_threshold=config.gate_low_threshold,
            high_threshold=config.gate_high_threshold,
            margin_threshold=config.gate_margin_threshold,
            stale_seconds=config.gate_stale_seconds,
            reject_threshold=getattr(config, 'gate_reject_threshold', 0.15),
        )

        # D-228: Phase 8 Production Deployment - Learned confidence gate
        self.learned_gate = LearnedConfidenceGate()
        self._use_learned_gate = getattr(config, 'use_learned_gate', True)

        self._use_shared = False

    def forward(
        self,
        query_vec: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        n_hops: Optional[int] = None,
    ) -> ReasoningResult:
        """Execute reasoning chain.

        Args:
            query_vec:     [batch, d_val] encoded query
            memory_keys:   [batch, num_slots, d_key]
            memory_values: [batch, num_slots, d_val]
            n_hops:        override number of hops

        Returns:
            ReasoningResult with retrieval vectors, route, confidence, etc.
        """
        if n_hops is None:
            n_hops = self.config.max_hops

        # Project query
        q = self.query_proj(query_vec)

        # Handle empty memory
        if memory_keys.shape[1] == 0:
            return ReasoningResult(
                retrieval_vectors=q.squeeze(0) if q.dim() > 1 else q,
                route="novel",
                confidence=0.0,
                n_hops_used=0,
            )

        # Choose reader
        if self._use_shared:
            final_state, all_logits, all_attn = self.shared_reader(
                q, memory_keys, memory_values, n_hops=n_hops,
            )
        else:
            final_state, all_logits, all_attn = self.reader(
                q, memory_keys, memory_values,
            )
            # Truncate to requested hops
            all_logits = all_logits[:n_hops]
            all_attn = all_attn[:n_hops]

        # Confidence gating on first-hop attention
        first_attn = all_attn[0] if all_attn else torch.tensor([])
        route, confidence = self.gate.route(first_attn)

        # D-228: Phase 8 Production Deployment - Use learned confidence gate
        from .confidence_gate import RouteLevel

        if self._use_learned_gate and first_attn.numel() > 0:
            # Compute raw cosine scores (before softmax) for learned gate features
            # Recompute first-hop cosine logits: q_proj @ keys (normalized)
            q_first = self.reader.query_projs[0](q)  # [batch, d_key]
            q_norm = F.normalize(q_first, dim=-1)  # [batch, d_key]
            k_norm = F.normalize(memory_keys, dim=-1)  # [batch, num_slots, d_key]

            # Raw cosine logits (unnormalized)
            raw_logits = (q_norm.unsqueeze(1) @ k_norm.transpose(-2, -1)).squeeze(1)  # [batch, num_slots]

            # Squeeze batch dim if needed for single-batch case
            if raw_logits.dim() == 2 and raw_logits.shape[0] == 1:
                raw_logits = raw_logits.squeeze(0)  # [num_slots]

            # Detach from computation graph (gate is not trainable)
            raw_logits = raw_logits.detach()
            first_attn_detached = first_attn.detach()

            # Compute entropy of attention distribution (D-195)
            # .item() converts CUDA tensor to Python float (fix: np.array fails on CUDA tensors)
            entropy_val = self._attention_entropy(first_attn_detached).item()

            # Call learned gate
            route_level_enum, lg_confidence, gate_signals = self.learned_gate.route(
                first_attn_detached,
                retrieval_scores=raw_logits,
                entropy=entropy_val,
            )
        else:
            # Fallback to rule-based gate (backward compatibility)
            route_level_enum, lg_confidence, gate_signals = self.multi_gate.route(
                first_attn,
            )

        route_level = route_level_enum.value

        # Backward-compatible mapping: INJECT_* -> "known", SKIP/REJECT -> "novel"
        if route_level_enum in (RouteLevel.SKIP, RouteLevel.REJECT):
            route = "novel"
        else:
            route = "known"

        # Squeeze batch dim for single-query case
        if final_state.dim() == 2 and final_state.shape[0] == 1:
            retrieval_vec = final_state.squeeze(0)
        else:
            retrieval_vec = final_state

        return ReasoningResult(
            retrieval_vectors=retrieval_vec,
            route=route,
            confidence=confidence,
            intermediate_logits=all_logits,
            attention_weights=all_attn,
            n_hops_used=n_hops,
            route_level=route_level,
            gate_signals=gate_signals,
        )

    @staticmethod
    def _attention_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention distribution (D-195).

        Args:
            attn_weights: [..., num_slots] softmax attention weights

        Returns:
            Scalar mean entropy.
        """
        # Guard against log(0)
        H = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean()
        return H

    def compute_loss(
        self,
        result: ReasoningResult,
        target_entities: List[int],
        supervision_weight: float = 1.0,
    ) -> torch.Tensor:
        """Compute training loss with intermediate supervision and entropy penalty.

        Args:
            result: ReasoningResult from forward()
            target_entities: list of target entity indices per hop
            supervision_weight: weight for intermediate hop losses (>= 0.5 REQUIRED)

        Returns:
            Total loss tensor.
        """
        assert supervision_weight >= 0.5, (
            "ANTI-PATTERN: intermediate supervision weight must be >= 0.5"
        )

        total_loss = torch.tensor(0.0, device=result.intermediate_logits[0].device)
        n_hops = min(len(result.intermediate_logits), len(target_entities))

        for hop in range(n_hops):
            logits = result.intermediate_logits[hop]
            target = torch.tensor(
                [target_entities[hop]],
                device=logits.device,
                dtype=torch.long,
            )
            if logits.dim() == 2:
                # [batch, n_entities]
                hop_loss = F.cross_entropy(logits, target.expand(logits.shape[0]))
            else:
                hop_loss = F.cross_entropy(logits.unsqueeze(0), target)

            # Final hop gets weight 1.0, intermediates get supervision_weight
            if hop < n_hops - 1:
                total_loss = total_loss + supervision_weight * hop_loss
            else:
                total_loss = total_loss + hop_loss

        # D-195: Attention entropy penalty — encourages sharper attention
        entropy_lambda = getattr(self.config, 'entropy_lambda', 0.0)
        if entropy_lambda > 0.0 and result.attention_weights:
            avg_entropy = torch.stack([
                self._attention_entropy(aw) for aw in result.attention_weights
            ]).mean()
            total_loss = total_loss - entropy_lambda * avg_entropy

        return total_loss
