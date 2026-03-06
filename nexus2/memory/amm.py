"""Adaptive Modular Memory -- high-level text-to-neural bridge.

Wraps encoder + memory bank + persistence into a single interface.
Accepts text, tokenizes, encodes, stores/retrieves from neural memory.

D-250/D-262: SentenceTransformer mode replaces hash+LSTM for inference.
The hash(c)%2000 tokenizer produces degenerate embeddings (all cosine ~0.98).
MiniLM gives real semantic embeddings; a Linear(384,512) projection preserves
the d_key/d_val=512 interface for chain/adapter compatibility.
"""

import logging
import threading
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import NexusConfig
from .encoder import Conv1DEncoder, LSTMEncoder
from .memory_bank import MemoryBank, MemoryEntry
from .persistence import load_memory, save_memory


def _create_encoder(config: NexusConfig) -> torch.nn.Module:
    """Factory: build encoder based on config.encoder_type."""
    if config.encoder_type == "mamba":
        try:
            from .mamba_encoder import MambaEncoder
            return MambaEncoder(
                embed_dim=config.embed_dim,
                hidden_dim=config.lstm_hidden,
                d_key=config.d_key,
                d_val=config.d_val,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
            )
        except ImportError:
            pass  # Fall through to LSTM
    return LSTMEncoder(
        embed_dim=config.embed_dim,
        hidden_dim=config.lstm_hidden,
        d_key=config.d_key,
        d_val=config.d_val,
        num_layers=config.lstm_layers,
        dropout=config.encoder_dropout,
    )


class AdaptiveModularMemory:
    """High-level AMM wrapping encoder + bank + persistence.

    Provides a text-level interface for the agent while internally
    using neural key/value encoding for storage and retrieval.

    D-250/D-262: When use_sentence_transformer=True (default), uses
    all-MiniLM-L6-v2 for encoding instead of hash+LSTM. A learned
    Linear(384, d_key) projection maps to the chain/adapter interface.
    """

    def __init__(self, config: NexusConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self._use_st = config.use_sentence_transformer

        if self._use_st:
            # D-250/D-262: SentenceTransformer replaces hash+LSTM
            self._st_model = None  # Lazy-loaded on first encode
            # Projection: SentenceTransformer dim -> d_key/d_val
            st_dim = config.sentence_transformer_dim
            self.st_key_proj = nn.Linear(st_dim, config.d_key).to(device)
            self.st_val_proj = nn.Linear(st_dim, config.d_val).to(device)
            # Initialize projections with small random weights
            nn.init.xavier_uniform_(self.st_key_proj.weight)
            nn.init.xavier_uniform_(self.st_val_proj.weight)

        # Keep LSTM pipeline for training (not used at inference when _use_st=True)
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim).to(device)
        self.encoder = _create_encoder(config).to(device)

        # Optional Conv1D encoder (loaded after distillation)
        self.conv_encoder: Optional[Conv1DEncoder] = None
        self._use_conv = False

        # Memory bank
        self.bank = MemoryBank(
            d_key=config.d_key,
            d_val=config.d_val,
            max_slots=config.max_slots,
            novelty_threshold=config.novelty_threshold,
            decay_enabled=config.memory_decay_enabled,
            decay_half_lives=config.decay_half_lives,
            dedup_enabled=config.memory_dedup_enabled,
            dedup_scope=config.memory_dedup_scope,
            type_boosts=config.retrieval_type_boosts,
        )

        self._lock = threading.Lock()

    def _get_st_model(self):
        """Lazy-load SentenceTransformer to avoid import overhead."""
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(
                self.config.sentence_transformer_model,
            )
            # Keep on CPU — encode() handles device transfer
            logging.info("Loaded SentenceTransformer: %s", self.config.sentence_transformer_model)
        return self._st_model

    def _tokenize(self, text: str) -> torch.Tensor:
        """Simple hash-based tokenizer mapping chars to vocab indices.
        Used only when _use_st=False (training pipeline).
        """
        text = text.lower()
        tokens = [hash(c) % self.config.vocab_size for c in text]
        if not tokens:
            tokens = [0]
        return torch.tensor([tokens], dtype=torch.long, device=self.device)

    def _get_encoder(self) -> nn.Module:
        """Return active encoder (Conv1D if distilled, else LSTM)."""
        if self._use_conv and self.conv_encoder is not None:
            return self.conv_encoder
        return self.encoder

    @torch.no_grad()
    def encode_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text to (key, value) vectors.

        D-250/D-262: Uses SentenceTransformer when enabled, falling back
        to hash+LSTM for training compatibility.

        Returns:
            key:   [1, d_key]
            value: [1, d_val]
        """
        if self._use_st:
            return self._encode_text_st(text)

        # Legacy path: hash tokenizer + LSTM
        tokens = self._tokenize(text)
        embeds = self.embedding(tokens)
        encoder = self._get_encoder()
        encoder.eval()
        key, value = encoder.encode_single(embeds)
        return key, value

    @torch.no_grad()
    def _encode_text_st(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text via SentenceTransformer + linear projection.

        text → SentenceTransformer → [384] → Linear → key [d_key], value [d_val]
        """
        st = self._get_st_model()
        # SentenceTransformer.encode returns numpy array
        embedding = st.encode(text, convert_to_tensor=True)  # [384]
        embedding = embedding.to(self.device).float()
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)  # [1, 384]

        key = self.st_key_proj(embedding)    # [1, d_key]
        value = self.st_val_proj(embedding)  # [1, d_val]
        return key, value

    def store(
        self,
        text: str,
        mem_type: str = "fact",
        subject: str = "",
        extra: Optional[Dict] = None,
    ) -> bool:
        """Encode and store a text fact in memory.

        Returns True if stored, False if rejected (duplicate).
        """
        key, value = self.encode_text(text)
        return self.bank.write(
            key.squeeze(0), value.squeeze(0),
            text=text, mem_type=mem_type, subject=subject, extra=extra,
        )

    def _compute_retrieval_entropy(self, query_key: torch.Tensor) -> float:
        """D-263: Compute normalized entropy of cosine similarity distribution.

        High entropy = uncertain retrieval (scores spread evenly).
        Low entropy = confident retrieval (one dominant match).

        Returns:
            Normalized entropy in [0, 1]. Returns 0.0 for empty banks.
        """
        import math
        query_key = query_key.detach().cpu()
        if query_key.dim() == 2:
            query_key = query_key.squeeze(0)

        with self.bank._lock:
            if len(self.bank._keys) == 0:
                return 0.0
            keys_snapshot = torch.stack(self.bank._keys)

        query_norm = torch.nn.functional.normalize(query_key.unsqueeze(0), dim=-1)
        keys_norm = torch.nn.functional.normalize(keys_snapshot, dim=-1)
        cos_sim = (query_norm @ keys_norm.T).squeeze(0)  # [N]

        # Convert to probability distribution via softmax
        probs = torch.softmax(cos_sim * 10.0, dim=0)  # temperature-scaled
        # Entropy
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum().item()
        # Normalize by max entropy (log N)
        max_entropy = math.log(len(keys_snapshot)) if len(keys_snapshot) > 1 else 1.0
        return min(entropy / max_entropy, 1.0)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float, MemoryEntry]]:
        """Retrieve memories most relevant to query text.

        D-263: Computes retrieval entropy and passes it to the memory bank
        for adaptive decay — high entropy boosts recent facts.

        Returns list of (text, score, metadata) tuples.
        """
        if top_k is None:
            top_k = self.config.retrieval_top_k
        key, _ = self.encode_text(query)
        query_key = key.squeeze(0)
        entropy = self._compute_retrieval_entropy(query_key)
        results = self.bank.read_with_metadata(query_key, top_k, entropy=entropy)
        return [(entry.text, score, entry) for _, score, entry in results]

    def retrieve_vectors(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Retrieve raw value vectors for reasoning chain input.

        Returns:
            values:  [top_k, d_val]
            weights: [top_k]
            indices: list of int
        """
        if top_k is None:
            top_k = self.config.retrieval_top_k
        key, _ = self.encode_text(query)
        return self.bank.read(key.squeeze(0), top_k)

    def is_novel(self, text: str) -> bool:
        """Check if text is novel (below novelty threshold)."""
        key, _ = self.encode_text(text)
        return self.bank.should_grow(key.squeeze(0))

    def delete_matching(self, pattern: str, only_types=None) -> int:
        """Delete entries matching text pattern."""
        return self.bank.delete_matching(pattern, only_types=only_types)

    def save(self):
        """Persist memory to disk."""
        save_memory(
            self.bank,
            self.config.memory_json_path,
            self.config.memory_pt_path,
        )

    def load(self) -> bool:
        """Load memory from disk. Returns True if loaded."""
        loaded = load_memory(
            self.bank,
            self.config.memory_json_path,
            self.config.memory_pt_path,
        )
        if (
            loaded
            and self._use_st
            and self.config.sentence_transformer_reencode_on_load
        ):
            self._reencode_snapshot_from_text()
        return loaded

    def _reencode_snapshot_from_text(self) -> None:
        """Rebuild persisted vectors from text in SentenceTransformer mode.

        Keys/values in .pt are projection-dependent. Re-encoding from text on load
        keeps retrieval stable even if process-local projection initialization differs.
        """
        _, _, metadata = self.bank.get_snapshot()
        if not metadata:
            return

        new_keys: list[torch.Tensor] = []
        new_values: list[torch.Tensor] = []
        for entry in metadata:
            text = (entry.text or "").strip()
            if not text:
                text = (entry.subject or "").strip() or "<empty>"
            key, value = self.encode_text(text)
            new_keys.append(key.squeeze(0).detach().cpu())
            new_values.append(value.squeeze(0).detach().cpu())

        self.bank.load_snapshot(new_keys, new_values, metadata)

    def use_conv_encoder(self, enable: bool = True):
        """Switch to Conv1D encoder (after distillation)."""
        if enable and self.conv_encoder is None:
            self.conv_encoder = Conv1DEncoder(
                embed_dim=self.config.embed_dim,
                channels=self.config.conv1d_channels,
                kernel_size=self.config.conv1d_kernel,
                d_key=self.config.d_key,
                d_val=self.config.d_val,
            ).to(self.device)
        self._use_conv = enable

    @property
    def size(self) -> int:
        return self.bank.size

    def get_stats(self) -> Dict:
        """Return memory statistics."""
        if self._use_st:
            encoder_name = f"SentenceTransformer({self.config.sentence_transformer_model})"
        elif self._use_conv:
            encoder_name = "conv1d"
        else:
            encoder_name = type(self.encoder).__name__
        return {
            "size": self.bank.size,
            "max_slots": self.config.max_slots,
            "d_key": self.config.d_key,
            "d_val": self.config.d_val,
            "encoder": encoder_name,
            "dirty": self.bank.dirty,
        }
