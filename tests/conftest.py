"""Shared test fixtures for NEXUS-2 tests."""

import os
import sys
import tempfile
from unittest.mock import MagicMock

import pytest
import torch

# Ensure nexus2 is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus2.config import NexusConfig
from nexus2.memory.amm import AdaptiveModularMemory
from nexus2.memory.memory_bank import MemoryBank
from nexus2.memory.encoder import LSTMEncoder
from nexus2.reasoning.chain_executor import ChainExecutor
from nexus2.reasoning.confidence_gate import ConfidenceGate, MultiSignalConfidenceGate


@pytest.fixture
def config():
    """Test configuration with small dimensions for speed."""
    cfg = NexusConfig()
    cfg.encoder_type = "lstm"  # Tests run on CPU; Mamba requires CUDA
    cfg.use_sentence_transformer = False  # Use LSTM for unit tests (fast, no model download)
    cfg.vocab_size = 200
    cfg.embed_dim = 32
    cfg.lstm_hidden = 64
    cfg.d_key = 32
    cfg.d_val = 32
    cfg.max_slots = 100
    cfg.max_hops = 3
    cfg.reasoning_hidden = 32
    cfg.adapter_hidden = 64
    cfg.num_soft_tokens = 2
    cfg.max_epochs_per_stage = 5
    cfg.memory_json_path = ""  # Will be overridden per test
    cfg.memory_pt_path = ""
    return cfg


@pytest.fixture
def memory_bank():
    """Fresh memory bank with small dimensions."""
    return MemoryBank(d_key=32, d_val=32, max_slots=100)


@pytest.fixture
def isolated_memory(config, tmp_path):
    """Fresh AMM with isolated temp directory."""
    config.memory_json_path = str(tmp_path / "test_memory.json")
    config.memory_pt_path = str(tmp_path / "test_memory.json.pt")
    config.checkpoint_dir = str(tmp_path / "checkpoints")
    config.skills_dir = str(tmp_path / "skills")
    config.skills_index = str(tmp_path / "skills" / "index.json")
    return AdaptiveModularMemory(config, device="cpu")


@pytest.fixture
def encoder(config):
    """Small LSTM encoder for testing."""
    return LSTMEncoder(
        embed_dim=config.embed_dim,
        hidden_dim=config.lstm_hidden,
        d_key=config.d_key,
        d_val=config.d_val,
    )


@pytest.fixture
def chain_executor(config):
    """Chain executor with small dimensions."""
    return ChainExecutor(config, n_entities=config.vocab_size)


@pytest.fixture
def confidence_gate():
    """Fresh confidence gate."""
    return ConfidenceGate(threshold=0.5, calibration_window=50)


@pytest.fixture
def multi_signal_gate():
    """Fresh multi-signal confidence gate."""
    return MultiSignalConfidenceGate(
        low_threshold=0.30,
        high_threshold=0.55,
        margin_threshold=0.10,
        stale_seconds=259200.0,
    )


@pytest.fixture
def mock_llm():
    """Deterministic mock LLM returning canned responses."""
    llm = MagicMock()
    llm.chat.return_value = "I understand. I'll remember that."
    llm.generate.return_value = "Here is my response."
    llm.generate_with_embeds.return_value = "Soft-prompt response."
    llm.get_embedding_dim.return_value = 64
    llm.get_embedding_dtype.return_value = torch.float32
    llm.device = "cpu"
    llm.tokenizer = MagicMock()
    llm.tokenizer.eos_token_id = 0
    llm.tokenizer.apply_chat_template.return_value = "formatted text"
    return llm


@pytest.fixture
def test_agent(config, tmp_path, mock_llm):
    """Fully wired agent with mocks for testing."""
    config.memory_json_path = str(tmp_path / "test_memory.json")
    config.memory_pt_path = str(tmp_path / "test_memory.json.pt")
    config.checkpoint_dir = str(tmp_path / "checkpoints")
    config.skills_dir = str(tmp_path / "skills")
    config.skills_index = str(tmp_path / "skills" / "index.json")

    from nexus2.agent import Nexus2Agent
    agent = Nexus2Agent(config=config, device="cpu", load_llm=False, load_checkpoints=False)
    # Inject mock LLM
    agent.llm = mock_llm
    from nexus2.generation.response_generator import ResponseGenerator
    agent.generator = ResponseGenerator(config, mock_llm, adapter=None)
    return agent
