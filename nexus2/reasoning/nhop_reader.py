"""N-hop memory reading chains.

ANTI-PATTERNS:
  - ALWAYS use per-hop read projections for N >= 3 (L-036)
  - NEVER use separate head for supervision vs chain decoder (D-024)
  - NEVER use sparse attention (entmax15, sparsemax) -- destroys grokking
  - NEVER use intermediate supervision weight < 0.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ExplicitNReadChain(nn.Module):
    """Explicit N-hop read chain with per-hop projections and intermediate decoders.

    Each hop has its own query projection (REQUIRED for N>=3 per L-036).
    Intermediate entity decoders serve BOTH supervision AND chain control (D-024).

    Architecture per hop i:
        query_i = project_i(state)
        attn_weights_i = softmax(query_i @ keys^T / sqrt(d))
        read_i = attn_weights_i @ values
        state = state + read_i
        entity_logits_i = decoder_i(state)  # used for both supervision and next query
    """

    def __init__(
        self,
        d_key: int = 256,
        d_val: int = 256,
        n_hops: int = 5,
        n_entities: int = 2000,
        intermediate_supervision: bool = True,
    ):
        super().__init__()
        self.d_key = d_key
        self.d_val = d_val
        self.n_hops = n_hops
        self.intermediate_supervision = intermediate_supervision

        # Per-hop query projections (REQUIRED for N >= 3)
        self.query_projs = nn.ModuleList([
            nn.Linear(d_val, d_key) for _ in range(n_hops)
        ])

        # Entity decoders -- used for BOTH supervision AND chain control (D-024)
        # One decoder per intermediate hop + final decoder
        self.entity_decoders = nn.ModuleList([
            nn.Linear(d_val, n_entities) for _ in range(n_hops)
        ])

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Execute N-hop reading.

        Args:
            query:  [batch, d_val] initial query state
            keys:   [batch, num_slots, d_key] memory keys
            values: [batch, num_slots, d_val] memory values

        Returns:
            final_state:        [batch, d_val]
            all_logits:         list of [batch, n_entities] per hop
            all_attn_weights:   list of [batch, num_slots] per hop
        """
        state = query  # [batch, d_val]
        all_logits = []
        all_attn_weights = []

        for hop in range(self.n_hops):
            # Project state to query space
            q = self.query_projs[hop](state)  # [batch, d_key]

            # Cosine attention (softmax, NEVER sparse)
            q_norm = F.normalize(q.unsqueeze(1), dim=-1)       # [batch, 1, d_key]
            k_norm = F.normalize(keys, dim=-1)                  # [batch, slots, d_key]
            attn_logits = (q_norm @ k_norm.transpose(-1, -2)).squeeze(1)  # [batch, slots]
            attn_weights = F.softmax(attn_logits, dim=-1)       # NEVER entmax/sparsemax
            all_attn_weights.append(attn_weights)

            # Read from memory
            read_vec = (attn_weights.unsqueeze(-1) * values).sum(dim=1)  # [batch, d_val]
            state = state + read_vec

            # Entity decode (same head for supervision AND chain control)
            logits = self.entity_decoders[hop](state)  # [batch, n_entities]
            all_logits.append(logits)

        return state, all_logits, all_attn_weights


class SharedNReadChain(nn.Module):
    """Shared-parameter N-hop reader for mixed-depth operation.

    Same architecture as ExplicitNReadChain but with weight sharing across hops.
    Smaller footprint (~554K params) but slightly less accurate at high hop counts.
    Suitable for deployment where max_hops varies dynamically.
    """

    def __init__(
        self,
        d_key: int = 256,
        d_val: int = 256,
        max_hops: int = 5,
        n_entities: int = 2000,
    ):
        super().__init__()
        self.d_key = d_key
        self.d_val = d_val
        self.max_hops = max_hops

        # Shared projections
        self.query_proj = nn.Linear(d_val, d_key)
        self.entity_decoder = nn.Linear(d_val, n_entities)

        # Hop embedding to differentiate positions
        self.hop_embed = nn.Embedding(max_hops, d_val)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        n_hops: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Execute N-hop reading with shared weights.

        Args:
            query:  [batch, d_val]
            keys:   [batch, num_slots, d_key]
            values: [batch, num_slots, d_val]
            n_hops: number of hops (defaults to max_hops)

        Returns:
            final_state, all_logits, all_attn_weights
        """
        if n_hops is None:
            n_hops = self.max_hops

        state = query
        all_logits = []
        all_attn_weights = []
        device = query.device

        for hop in range(n_hops):
            # Add hop embedding to differentiate positions
            hop_idx = torch.tensor([hop], device=device).expand(query.shape[0])
            state = state + self.hop_embed(hop_idx)

            # Shared query projection
            q = self.query_proj(state)

            # Cosine attention
            q_norm = F.normalize(q.unsqueeze(1), dim=-1)
            k_norm = F.normalize(keys, dim=-1)
            attn_logits = (q_norm @ k_norm.transpose(-1, -2)).squeeze(1)
            attn_weights = F.softmax(attn_logits, dim=-1)
            all_attn_weights.append(attn_weights)

            # Read
            read_vec = (attn_weights.unsqueeze(-1) * values).sum(dim=1)
            state = state + read_vec

            # Shared entity decode
            logits = self.entity_decoder(state)
            all_logits.append(logits)

        return state, all_logits, all_attn_weights
