"""Mamba SSM encoder for memory key/value computation.

Drop-in replacement for LSTMEncoder using Mamba selective state-space blocks.
Requires mamba-ssm package (optional, CUDA-only). Falls back gracefully if unavailable.

Finding D-184: Mamba encoder matches LSTM accuracy with better sequence scaling.
ANTI-PATTERN: NEVER train LSTM at k > 500 — use Mamba for extended k-schedules.
"""

import torch
import torch.nn as nn

# Graceful import: mamba-ssm requires CUDA
MAMBA_AVAILABLE = False
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    pass


class NativeMambaBlock(nn.Module):
    """Single Mamba block: LayerNorm -> Mamba -> residual."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for NativeMambaBlock. "
                "Install with: pip install mamba-ssm>=1.2.0 (requires CUDA)"
            )
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm residual.

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model]
        """
        return x + self.mamba(self.norm(x))


class MambaEncoder(nn.Module):
    """Mamba-based encoder — drop-in replacement for LSTMEncoder.

    Takes token embeddings and produces (keys, values) via Mamba blocks
    followed by linear projection, matching the LSTMEncoder interface.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        d_key: int = 256,
        d_val: int = 256,
        n_layers: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for MambaEncoder. "
                "Install with: pip install mamba-ssm>=1.2.0 (requires CUDA)"
            )

        # Project from embedding dim to Mamba operating dim
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # Stack of Mamba blocks
        self.blocks = nn.ModuleList([
            NativeMambaBlock(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])

        # Key/value projections (same interface as LSTMEncoder)
        self.key_proj = nn.Linear(hidden_dim, d_key)
        self.val_proj = nn.Linear(hidden_dim, d_val)

        self.hidden_dim = hidden_dim
        self.d_key = d_key
        self.d_val = d_val

    def forward(self, x: torch.Tensor) -> tuple:
        """Encode input embeddings to key/value pairs.

        Args:
            x: [batch, seq_len, embed_dim] input embeddings

        Returns:
            keys:   [batch, seq_len, d_key]
            values: [batch, seq_len, d_val]
        """
        h = self.input_proj(x)  # [batch, seq_len, hidden_dim]
        for block in self.blocks:
            h = block(h)
        keys = self.key_proj(h)    # [batch, seq_len, d_key]
        values = self.val_proj(h)  # [batch, seq_len, d_val]
        return keys, values

    def encode_single(self, x: torch.Tensor) -> tuple:
        """Encode and return the last-timestep key/value (for single-fact storage).

        Args:
            x: [batch, seq_len, embed_dim]

        Returns:
            key:   [batch, d_key]
            value: [batch, d_val]
        """
        keys, values = self.forward(x)
        return keys[:, -1, :], values[:, -1, :]
