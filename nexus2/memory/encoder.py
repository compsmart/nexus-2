"""Neural encoders for memory key/value computation.

ANTI-PATTERNS:
  - NEVER use BiLSTM for key computation -- forward-only required (L-025)
  - NEVER increase LSTM depth > 1 without memory-forcing mechanism
  - NEVER use sparse attention (entmax15, sparsemax) -- destroys grokking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    """Unidirectional LSTM encoder producing key/value vectors.

    Takes token embeddings and produces (keys, values) via linear projection
    of LSTM hidden states. This is the primary encoder used during training;
    at inference time it may be replaced by the distilled Conv1DEncoder.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        d_key: int = 256,
        d_val: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        # ANTI-PATTERN GUARD: bidirectional=False is mandatory
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # NEVER change to True
            dropout=dropout if num_layers > 1 else 0.0,
        )
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
        # LSTM forward pass (unidirectional only)
        hidden_states, _ = self.lstm(x)  # [batch, seq, hidden_dim]
        keys = self.key_proj(hidden_states)    # [batch, seq, d_key]
        values = self.val_proj(hidden_states)  # [batch, seq, d_val]
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


class Conv1DEncoder(nn.Module):
    """Causal Conv1D encoder -- drop-in replacement for LSTMEncoder.

    Distilled from the LSTM encoder for 28x faster inference.
    Uses causal (left-only) padding to maintain autoregressive property.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        channels: int = 512,
        kernel_size: int = 7,
        d_key: int = 256,
        d_val: int = 256,
    ):
        super().__init__()
        self.causal_pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=0,  # manual causal padding
        )
        self.key_proj = nn.Linear(channels, d_key)
        self.val_proj = nn.Linear(channels, d_val)

        self.d_key = d_key
        self.d_val = d_val

    def forward(self, x: torch.Tensor) -> tuple:
        """Encode input embeddings to key/value pairs.

        Args:
            x: [batch, seq_len, embed_dim]

        Returns:
            keys:   [batch, seq_len, d_key]
            values: [batch, seq_len, d_val]
        """
        # Transpose for Conv1d: [batch, embed_dim, seq_len]
        h = x.transpose(1, 2)
        # Causal (left-only) padding
        h = F.pad(h, (self.causal_pad, 0))
        h = self.conv(h)  # [batch, channels, seq_len]
        h = F.gelu(h)
        # Transpose back: [batch, seq_len, channels]
        h = h.transpose(1, 2)
        keys = self.key_proj(h)
        values = self.val_proj(h)
        return keys, values

    def encode_single(self, x: torch.Tensor) -> tuple:
        """Encode and return the last-timestep key/value."""
        keys, values = self.forward(x)
        return keys[:, -1, :], values[:, -1, :]
