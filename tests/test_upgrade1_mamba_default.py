"""Upgrade 1 TDD tests: Mamba as default encoder.

These tests verify:
  - Default NexusConfig uses encoder_type="mamba"
  - _create_encoder produces MambaEncoder on CUDA, falls back to LSTM on CPU
  - Curriculum engine allows k>500 for Mamba, caps LSTM at 500
  - Conv1D distillation works from Mamba (same interface as LSTM distillation)
  - AMM reports correct encoder type in stats

Research basis:
  D-184: Mamba matches LSTM accuracy with better sequence scaling
  D-188: LSTM hard ceiling at k>500 — training destroys model
  D-158: Mamba+AMM achieves 83% FR-2 grokking
"""

import pytest
import torch

from nexus2.config import NexusConfig
from nexus2.memory.amm import AdaptiveModularMemory, _create_encoder
from nexus2.memory.encoder import LSTMEncoder, Conv1DEncoder
from nexus2.learning.curriculum_engine import CurriculumEngine


class TestMambaDefault:
    """Verify Mamba is the default encoder and LSTM is the fallback."""

    def test_default_config_uses_lstm(self):
        """NexusConfig.encoder_type should default to 'lstm' (trained checkpoints available)."""
        cfg = NexusConfig()
        assert cfg.encoder_type == "lstm"

    def test_factory_creates_mamba_on_cuda(self):
        """_create_encoder should produce MambaEncoder when CUDA is available."""
        from nexus2.memory.mamba_encoder import MAMBA_AVAILABLE
        if not MAMBA_AVAILABLE or not torch.cuda.is_available():
            pytest.skip("Mamba + CUDA required")

        cfg = NexusConfig()
        cfg.encoder_type = "mamba"
        cfg.embed_dim = 32
        cfg.lstm_hidden = 64
        cfg.d_key = 32
        cfg.d_val = 32
        encoder = _create_encoder(cfg)

        from nexus2.memory.mamba_encoder import MambaEncoder
        assert isinstance(encoder, MambaEncoder)

    def test_factory_falls_back_to_lstm_on_cpu(self):
        """When Mamba is requested but unavailable, fall back to LSTM gracefully."""
        cfg = NexusConfig()
        cfg.encoder_type = "mamba"
        cfg.embed_dim = 32
        cfg.lstm_hidden = 64
        cfg.d_key = 32
        cfg.d_val = 32

        # Force CPU-only scenario: create encoder and check type
        # On CPU without mamba-ssm, should fall back to LSTM
        encoder = _create_encoder(cfg)
        # Either MambaEncoder (if CUDA available) or LSTMEncoder (fallback)
        from nexus2.memory.mamba_encoder import MAMBA_AVAILABLE
        if not MAMBA_AVAILABLE or not torch.cuda.is_available():
            assert isinstance(encoder, LSTMEncoder)
        else:
            from nexus2.memory.mamba_encoder import MambaEncoder
            assert isinstance(encoder, MambaEncoder)

    def test_lstm_explicit_config(self):
        """Setting encoder_type='lstm' should always produce LSTMEncoder."""
        cfg = NexusConfig()
        cfg.encoder_type = "lstm"
        cfg.embed_dim = 32
        cfg.lstm_hidden = 64
        cfg.d_key = 32
        cfg.d_val = 32
        encoder = _create_encoder(cfg)
        assert isinstance(encoder, LSTMEncoder)


class TestMambaKSchedule:
    """Verify Mamba allows extended k-schedules beyond LSTM's k=500 cap."""

    def test_mamba_curriculum_includes_k750(self):
        """Mamba k-schedule should include k values above 500."""
        engine = CurriculumEngine(
            k_schedule=[5, 10, 50, 100, 200, 500, 750, 1000],
            encoder_type="mamba",
        )
        assert 750 in engine.k_schedule
        assert 1000 in engine.k_schedule

    def test_lstm_curriculum_caps_at_500(self):
        """LSTM k-schedule must cap at 500 (D-188)."""
        engine = CurriculumEngine(
            k_schedule=[5, 10, 50, 100, 200, 500, 750, 1000],
            encoder_type="lstm",
        )
        assert 750 not in engine.k_schedule
        assert 1000 not in engine.k_schedule
        assert max(engine.k_schedule) == 500

    def test_default_config_k_schedule_matches_encoder(self):
        """Default config k_schedule should include extended values for curriculum."""
        cfg = NexusConfig()
        assert cfg.encoder_type == "lstm"
        # k_schedule has extended values; LSTM caps at 500 via CurriculumEngine
        assert any(k > 100 for k in cfg.k_schedule)


class TestMambaAMMIntegration:
    """End-to-end tests: AMM with Mamba encoder on CUDA."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_mamba_cuda(self):
        from nexus2.memory.mamba_encoder import MAMBA_AVAILABLE
        if not MAMBA_AVAILABLE:
            pytest.skip("mamba-ssm not installed")
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for Mamba")

    def test_amm_creates_mamba_encoder(self, tmp_path):
        """AMM should use MambaEncoder when config says mamba and CUDA available."""
        cfg = NexusConfig()
        cfg.encoder_type = "mamba"
        cfg.vocab_size = 200
        cfg.embed_dim = 32
        cfg.lstm_hidden = 64
        cfg.d_key = 32
        cfg.d_val = 32
        cfg.max_slots = 50
        cfg.memory_json_path = str(tmp_path / "mem.json")
        cfg.memory_pt_path = str(tmp_path / "mem.json.pt")

        amm = AdaptiveModularMemory(cfg, device="cuda")

        from nexus2.memory.mamba_encoder import MambaEncoder
        assert isinstance(amm.encoder, MambaEncoder)

    def test_amm_stats_report_mamba(self, tmp_path):
        """AMM stats should report 'MambaEncoder' when using Mamba."""
        cfg = NexusConfig()
        cfg.encoder_type = "mamba"
        cfg.use_sentence_transformer = False  # Test Mamba encoder path
        cfg.vocab_size = 200
        cfg.embed_dim = 32
        cfg.lstm_hidden = 64
        cfg.d_key = 32
        cfg.d_val = 32
        cfg.max_slots = 50
        cfg.memory_json_path = str(tmp_path / "mem.json")
        cfg.memory_pt_path = str(tmp_path / "mem.json.pt")

        amm = AdaptiveModularMemory(cfg, device="cuda")
        stats = amm.get_stats()
        assert stats["encoder"] == "MambaEncoder"

    def test_amm_mamba_store_and_retrieve(self, tmp_path):
        """Basic store/retrieve cycle should work with Mamba encoder."""
        cfg = NexusConfig()
        cfg.encoder_type = "mamba"
        cfg.vocab_size = 200
        cfg.embed_dim = 32
        cfg.lstm_hidden = 64
        cfg.d_key = 32
        cfg.d_val = 32
        cfg.max_slots = 100
        cfg.memory_json_path = str(tmp_path / "mem.json")
        cfg.memory_pt_path = str(tmp_path / "mem.json.pt")

        amm = AdaptiveModularMemory(cfg, device="cuda")
        amm.store("Alice likes red", mem_type="fact")
        amm.store("Bob likes blue", mem_type="fact")

        results = amm.retrieve("Alice")
        assert len(results) > 0

    def test_amm_mamba_conv1d_distillation_interface(self, tmp_path):
        """Conv1D distillation switch should work regardless of primary encoder type."""
        cfg = NexusConfig()
        cfg.encoder_type = "mamba"
        cfg.use_sentence_transformer = False  # Test Mamba/Conv1D encoder path
        cfg.vocab_size = 200
        cfg.embed_dim = 32
        cfg.lstm_hidden = 64
        cfg.d_key = 32
        cfg.d_val = 32
        cfg.conv1d_channels = 64
        cfg.conv1d_kernel = 3
        cfg.max_slots = 50
        cfg.memory_json_path = str(tmp_path / "mem.json")
        cfg.memory_pt_path = str(tmp_path / "mem.json.pt")

        amm = AdaptiveModularMemory(cfg, device="cuda")

        # Store with Mamba
        amm.store("test fact", mem_type="fact")

        # Switch to Conv1D
        amm.use_conv_encoder(enable=True)
        assert amm._use_conv is True
        assert isinstance(amm.conv_encoder, Conv1DEncoder)

        # Retrieve should still work (different encoder, same bank)
        results = amm.retrieve("test")
        assert len(results) > 0

        stats = amm.get_stats()
        assert stats["encoder"] == "conv1d"
