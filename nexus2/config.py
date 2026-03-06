"""NEXUS-2 configuration dataclass with all hyperparameters.

Anti-pattern guards are documented inline. Violating any WILL break the system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class NexusConfig:
    """Central configuration for all NEXUS-2 components."""

    # --- LLM ---
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    use_4bit: bool = True
    device: str = "auto"  # "auto", "cuda", "cpu"
    max_new_tokens: int = 128
    temperature: float = 0.6
    repetition_penalty: float = 1.2
    context_fallback: int = 4096

    # --- Encoder ---
    # ANTI-PATTERN: NEVER use BiLSTM (L-025). Forward-only required.
    # ANTI-PATTERN: NEVER use LSTM depth > 1 without memory-forcing mechanism.
    # ANTI-PATTERN: NEVER train LSTM at k > 500 — use Mamba for extended k-schedules.
    # ANTI-PATTERN: NEVER use d_key/d_val < 256 (D-199).
    encoder_type: str = "lstm"  # "mamba" or "lstm" (D-184) — lstm has trained checkpoints
    sentence_transformer_model: str = "all-MiniLM-L6-v2"  # D-250/D-262: real embeddings for inference
    use_sentence_transformer: bool = True  # D-250: use SentenceTransformer instead of hash+LSTM
    sentence_transformer_dim: int = 384  # output dim of all-MiniLM-L6-v2
    # Rebuild key/value tensors from persisted text on load when ST mode is enabled.
    # This avoids retrieval drift across process restarts when projection weights differ.
    sentence_transformer_reencode_on_load: bool = True
    lstm_hidden: int = 512
    lstm_layers: int = 1
    lstm_bidirectional: bool = False  # MUST stay False
    d_key: int = 512  # D-199: scaled from 256
    d_val: int = 512  # D-199: scaled from 256
    encoder_dropout: float = 0.0
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    # --- Conv1D Encoder (distilled replacement) ---
    conv1d_kernel: int = 7
    conv1d_channels: int = 512

    # --- Tokenizer / Embedding ---
    vocab_size: int = 2000  # synthetic vocabulary size for training
    embed_dim: int = 128  # input embedding dimension for encoder

    # --- Memory Bank ---
    max_slots: int = 10000
    novelty_threshold: float = 0.5  # grow slot when max_cos < this
    retrieval_top_k: int = 3  # D-261: selective top-3 beats full top-10 (93.1% vs 90.8%)
    memory_dedup_enabled: bool = True
    memory_dedup_scope: str = "exact_text"  # exact_text | normalized_text | off
    memory_decay_enabled: bool = True
    # D-228: Hybrid neural+text retrieval breaks 65% constraint ceiling to 85%
    hybrid_retrieval_enabled: bool = True  # D-228: merge neural + text search
    hybrid_text_weight: float = 0.3  # weight for text search in hybrid merge
    # Per-type half-lives in seconds (None = never decays)
    decay_half_lives: Dict[str, Optional[float]] = field(default_factory=lambda: {
        "identity": None,
        "correction": None,         # never decays — corrections always outrank stale facts
        "skill": 2_592_000.0,      # 30 days
        "fact": 1_209_600.0,        # 14 days
        "web_fact": 604_800.0,      # D-275: 7 days (shorter than fact's 14 — web content is less reliable)
        "user_input": 259_200.0,    # 3 days
        "agent_response": 259_200.0,  # 3 days
        "default": 604_800.0,       # 7 days
    })
    # Per-type retrieval boost: multiplied into cosine score so facts/corrections
    # rank above conversation entries at similar similarity distances.
    retrieval_type_boosts: Dict[str, float] = field(default_factory=lambda: {
        "identity": 1.25,
        "correction": 1.20,
        "fact": 1.15,
        "document": 1.10,
        "skill": 1.05,
        "web_fact": 1.0,            # D-275: no boost (unvalidated web content)
        "user_input": 0.70,
        "agent_response": 0.70,
    })

    # --- Reasoning ---
    # ANTI-PATTERN: ALWAYS use per-hop read projections for N >= 3 (L-036).
    # ANTI-PATTERN: NEVER use separate head for supervision vs chain decoder (D-024).
    # ANTI-PATTERN: NEVER use intermediate supervision weight < 0.5.
    max_hops: int = 5
    per_hop_projections: bool = True  # MUST be True for N >= 3
    intermediate_supervision_weight: float = 1.0  # MUST be >= 0.5
    reasoning_hidden: int = 256
    entropy_lambda: float = 0.01  # D-195: attention entropy regularization

    # --- Confidence Gate ---
    # D-222: raw_cos_max is primary routing signal (only synthetic→real transferable feature)
    # D-227: REJECT level achieves ≤0.1% hallucination when raw_cos_max < reject_threshold
    confidence_threshold: float = 0.5  # above = known, below = novel
    gate_calibration_window: int = 100  # recent scores for auto-calibration
    gate_low_threshold: float = 0.30   # D-197: multi-signal gate low
    gate_high_threshold: float = 0.55  # D-197: multi-signal gate high
    gate_margin_threshold: float = 0.10  # D-197: retrieval margin for FULL vs TOP1
    gate_stale_seconds: float = 259200.0  # D-197: 3 days staleness cutoff
    gate_reject_threshold: float = 0.15  # D-227: below this → REJECT (≤0.1% hallucination)

    # --- Soft-Prompt Adapter ---
    # ANTI-PATTERN: ALWAYS match adapter output dtype to LLM dtype (L-034).
    # ANTI-PATTERN: ALWAYS use LLM's native lm_head (not custom classifiers).
    num_soft_tokens: int = 8  # L-246: 8 tokens is validated sweet spot (was 4)
    adapter_hidden: int = 512
    adapter_dropout: float = 0.1

    # --- Training ---
    # ANTI-PATTERN: NEVER skip curriculum for k > 15.
    # ANTI-PATTERN: NEVER train LSTM at k > 500.
    # ANTI-PATTERN: NEVER use sparse attention (entmax15, sparsemax) -- destroys grokking.
    # ANTI-PATTERN: NEVER use sigmoid gating between different encoders.
    # ANTI-PATTERN: NEVER use shared-prefix entity names (use diverse adjective+noun).
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 32
    grad_clip: float = 1.0
    k_schedule: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200, 350, 500, 750, 1000])  # D-188: extended for Mamba
    hop_schedule: List[int] = field(default_factory=lambda: [2, 3, 4, 5])
    curriculum_convergence_threshold: float = 0.97  # 97% to advance
    max_epochs_per_stage: int = 500
    mixed_k_epochs: int = 100  # D-183: epochs for mixed-K regularization phase
    mixed_k_enabled: bool = True  # D-183: enable mixed-K cross-scale training
    distillation_epochs: int = 50
    adapter_train_epochs: int = 100

    # --- Perception ---
    name_patterns: List[str] = field(default_factory=lambda: [
        r"\bmy name is\s+([A-Za-z][A-Za-z0-9'\- ]{0,40})",
        r"\bi am\s+([A-Za-z][A-Za-z0-9'\- ]{0,40})",
        r"\bi'm\s+([A-Za-z][A-Za-z0-9'\- ]{0,40})",
        r"\bcall me\s+([A-Za-z][A-Za-z0-9'\- ]{0,40})",
    ])
    blocked_names: set = field(default_factory=lambda: {
        "nexus", "assistant", "ai", "bot", "creator", "your creator",
        "a human", "a person", "nobody", "someone", "something",
    })
    negation_cues: set = field(default_factory=lambda: {
        "don't", "not", "never", "no longer", "isn't", "aren't",
        "wasn't", "weren't", "won't", "can't", "couldn't",
    })

    # --- Action / Tools ---
    max_tool_calls_per_turn: int = 3
    # Permissive regex: LLM sometimes generates [TOOLCallCheck:...], [Tool_Call:...] etc.
    tool_call_pattern: str = r'\[\s*(?:TOOL_?CALL|TOOLCall\w*|Tool_?Call)\s*:\s*(\w+)\s*\|\s*(.+?)\s*\]'
    autonomous_learning: bool = True
    uncertainty_cues: List[str] = field(default_factory=lambda: [
        "don't know", "not sure", "cannot find", "no information",
        "i'm uncertain", "i don't have", "unable to find",
    ])

    # --- Skills ---
    skills_dir: str = "data/skills"
    skills_index: str = "data/skills/index.json"

    # --- Persistence ---
    memory_json_path: str = "data/memory/nexus2_memory.json"
    memory_pt_path: str = "data/memory/nexus2_memory.json.pt"
    checkpoint_dir: str = "nexus2/models/checkpoints"

    # --- Background Consolidation ---
    think_interval_secs: float = 30.0
    flush_interval_secs: float = 60.0

    # --- Benchmarks ---
    benchmark_k_values: List[int] = field(default_factory=lambda: [10, 25, 50, 100])
    benchmark_hop_values: List[int] = field(default_factory=lambda: [2, 3, 4, 5])
