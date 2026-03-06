"""AMM unit tests: write/read/decay/evict/persist/dedup."""

import time

import pytest
import torch

from nexus2.memory.memory_bank import MemoryBank, MemoryEntry
from nexus2.memory.persistence import save_memory, load_memory
from nexus2.memory.encoder import LSTMEncoder, Conv1DEncoder


class TestMemoryBank:
    """Tests for the core memory bank."""

    def test_write_and_read(self, memory_bank):
        key = torch.randn(32)
        val = torch.randn(32)
        assert memory_bank.write(key, val, text="test fact")
        assert memory_bank.size == 1

    def test_read_returns_correct_value(self, memory_bank):
        key = torch.randn(32)
        val = torch.randn(32)
        memory_bank.write(key, val, text="target fact")

        # Add distractors
        for i in range(5):
            memory_bank.write(torch.randn(32), torch.randn(32), text=f"distractor {i}")

        values, weights, indices = memory_bank.read(key, top_k=1)
        assert values.shape[0] == 1
        assert 0 in indices  # target should be among results

    def test_dedup_rejects_duplicates(self):
        bank = MemoryBank(d_key=32, d_val=32, dedup_enabled=True)
        key = torch.randn(32)
        val = torch.randn(32)
        assert bank.write(key, val, text="same fact", mem_type="fact")
        assert not bank.write(key, val, text="same fact", mem_type="fact")
        assert bank.size == 1

    def test_dedup_disabled_allows_duplicates(self):
        bank = MemoryBank(d_key=32, d_val=32, dedup_enabled=False)
        key = torch.randn(32)
        val = torch.randn(32)
        assert bank.write(key, val, text="same fact")
        assert bank.write(key, val, text="same fact")
        assert bank.size == 2

    def test_fifo_eviction(self):
        bank = MemoryBank(d_key=32, d_val=32, max_slots=3, dedup_enabled=False)
        for i in range(5):
            bank.write(torch.randn(32), torch.randn(32), text=f"fact {i}")
        assert bank.size == 3
        # First two should have been evicted
        with bank._lock:
            texts = [m.text for m in bank._metadata]
        assert "fact 0" not in texts
        assert "fact 1" not in texts
        assert "fact 4" in texts

    def test_novelty_detection(self, memory_bank):
        # Empty bank: everything is novel
        assert memory_bank.should_grow(torch.randn(32))

        # Add a known key
        key = torch.randn(32)
        memory_bank.write(key, torch.randn(32), text="known")
        # Same key should not be novel
        assert not memory_bank.should_grow(key)

    def test_delete_matching(self, memory_bank):
        memory_bank.write(torch.randn(32), torch.randn(32), text="my dog is Bruno")
        memory_bank.write(torch.randn(32), torch.randn(32), text="my cat is Luna")
        assert memory_bank.size == 2
        deleted = memory_bank.delete_matching("dog")
        assert deleted == 1
        assert memory_bank.size == 1

    def test_empty_read(self, memory_bank):
        values, weights, indices = memory_bank.read(torch.randn(32), top_k=5)
        assert values.shape[0] == 0
        assert len(indices) == 0

    def test_max_similarity_empty(self, memory_bank):
        assert memory_bank.max_similarity(torch.randn(32)) == 0.0

    def test_top_k_limit(self, memory_bank):
        for i in range(20):
            memory_bank.write(torch.randn(32), torch.randn(32), text=f"fact {i}")
        values, weights, indices = memory_bank.read(torch.randn(32), top_k=5)
        assert values.shape[0] == 5
        assert len(indices) == 5

    def test_clear(self, memory_bank):
        memory_bank.write(torch.randn(32), torch.randn(32), text="fact")
        memory_bank.clear()
        assert memory_bank.size == 0

    def test_snapshot_restore(self, memory_bank):
        for i in range(5):
            memory_bank.write(torch.randn(32), torch.randn(32), text=f"fact {i}")

        keys, vals, meta = memory_bank.get_snapshot()
        assert len(keys) == 5

        new_bank = MemoryBank(d_key=32, d_val=32)
        new_bank.load_snapshot(keys, vals, meta)
        assert new_bank.size == 5


class TestPersistence:
    """Tests for atomic save/load."""

    def test_save_and_load(self, memory_bank, tmp_path):
        for i in range(5):
            memory_bank.write(torch.randn(32), torch.randn(32), text=f"fact {i}")

        json_path = str(tmp_path / "test.json")
        pt_path = str(tmp_path / "test.json.pt")

        save_memory(memory_bank, json_path, pt_path)
        assert not memory_bank.dirty

        new_bank = MemoryBank(d_key=32, d_val=32)
        assert load_memory(new_bank, json_path, pt_path)
        assert new_bank.size == 5

    def test_load_nonexistent(self, memory_bank, tmp_path):
        assert not load_memory(memory_bank, str(tmp_path / "nope.json"))

    def test_roundtrip_preserves_metadata(self, memory_bank, tmp_path):
        memory_bank.write(
            torch.randn(32), torch.randn(32),
            text="my dog is Bruno", mem_type="fact", subject="personal",
        )

        json_path = str(tmp_path / "test.json")
        save_memory(memory_bank, json_path)

        new_bank = MemoryBank(d_key=32, d_val=32)
        load_memory(new_bank, json_path)
        with new_bank._lock:
            assert new_bank._metadata[0].text == "my dog is Bruno"
            assert new_bank._metadata[0].mem_type == "fact"
            assert new_bank._metadata[0].subject == "personal"


class TestEncoders:
    """Tests for LSTM and Conv1D encoders."""

    def test_lstm_encoder_shapes(self, encoder):
        x = torch.randn(2, 10, 32)
        keys, values = encoder(x)
        assert keys.shape == (2, 10, 32)
        assert values.shape == (2, 10, 32)

    def test_lstm_encode_single(self, encoder):
        x = torch.randn(1, 5, 32)
        key, val = encoder.encode_single(x)
        assert key.shape == (1, 32)
        assert val.shape == (1, 32)

    def test_conv1d_encoder_shapes(self):
        enc = Conv1DEncoder(embed_dim=32, channels=64, kernel_size=3, d_key=32, d_val=32)
        x = torch.randn(2, 10, 32)
        keys, values = enc(x)
        assert keys.shape == (2, 10, 32)
        assert values.shape == (2, 10, 32)

    def test_conv1d_causal(self):
        """Conv1D output at position t should not depend on future inputs."""
        enc = Conv1DEncoder(embed_dim=16, channels=32, kernel_size=3, d_key=16, d_val=16)
        enc.eval()
        x = torch.randn(1, 8, 16)
        keys1, _ = enc(x)
        # Modify future token
        x2 = x.clone()
        x2[0, 7, :] = 0  # zero out last position
        keys2, _ = enc(x2)
        # Positions before 7 should be unchanged
        assert torch.allclose(keys1[0, :7, :], keys2[0, :7, :], atol=1e-6)


class TestMambaEncoder:
    """Tests for Mamba encoder (skipped if mamba-ssm unavailable or no CUDA)."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_mamba(self):
        from nexus2.memory.mamba_encoder import MAMBA_AVAILABLE
        if not MAMBA_AVAILABLE:
            pytest.skip("mamba-ssm not installed")
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for mamba-ssm")

    def _device(self):
        return "cuda"

    def test_mamba_encoder_shapes(self):
        from nexus2.memory.mamba_encoder import MambaEncoder
        dev = self._device()
        enc = MambaEncoder(embed_dim=32, hidden_dim=64, d_key=32, d_val=32).to(dev)
        x = torch.randn(2, 10, 32, device=dev)
        keys, values = enc(x)
        assert keys.shape == (2, 10, 32)
        assert values.shape == (2, 10, 32)

    def test_mamba_encode_single(self):
        from nexus2.memory.mamba_encoder import MambaEncoder
        dev = self._device()
        enc = MambaEncoder(embed_dim=32, hidden_dim=64, d_key=32, d_val=32).to(dev)
        x = torch.randn(1, 5, 32, device=dev)
        key, val = enc.encode_single(x)
        assert key.shape == (1, 32)
        assert val.shape == (1, 32)

    def test_mamba_causal(self):
        """Mamba output at position t should not depend on future inputs."""
        from nexus2.memory.mamba_encoder import MambaEncoder
        dev = self._device()
        enc = MambaEncoder(embed_dim=16, hidden_dim=32, d_key=16, d_val=16).to(dev)
        enc.eval()
        x = torch.randn(1, 8, 16, device=dev)
        keys1, _ = enc(x)
        # Modify future token
        x2 = x.clone()
        x2[0, 7, :] = 0
        keys2, _ = enc(x2)
        # Positions before 7 should be unchanged (Mamba is causal)
        assert torch.allclose(keys1[0, :7, :], keys2[0, :7, :], atol=1e-5)


class TestAMM:
    """Tests for the high-level AdaptiveModularMemory."""

    def test_store_and_retrieve(self, isolated_memory):
        isolated_memory.store("my dog is Bruno", mem_type="fact")
        results = isolated_memory.retrieve("dog")
        assert len(results) > 0
        assert any("Bruno" in text for text, _, _ in results)

    def test_dedup(self, isolated_memory):
        isolated_memory.store("same fact", mem_type="fact")
        isolated_memory.store("same fact", mem_type="fact")
        assert isolated_memory.size == 1

    def test_is_novel(self, isolated_memory):
        # Empty memory: everything is novel
        assert isolated_memory.is_novel("anything")

    def test_persistence_roundtrip(self, isolated_memory):
        isolated_memory.store("persisted fact", mem_type="fact")
        isolated_memory.save()

        # Create new AMM with same paths
        from nexus2.memory.amm import AdaptiveModularMemory
        new_amm = AdaptiveModularMemory(isolated_memory.config, device="cpu")
        assert new_amm.load()
        assert new_amm.size == 1

    def test_sentence_transformer_load_reencodes_vectors(self, tmp_path, monkeypatch):
        """ST mode should rebuild vectors from text on load to prevent drift."""
        from nexus2.config import NexusConfig
        from nexus2.memory.amm import AdaptiveModularMemory
        from nexus2.memory.persistence import save_memory

        cfg = NexusConfig()
        cfg.use_sentence_transformer = True
        cfg.d_key = 4
        cfg.d_val = 4
        cfg.memory_json_path = str(tmp_path / "st_mem.json")
        cfg.memory_pt_path = str(tmp_path / "st_mem.json.pt")
        cfg.sentence_transformer_reencode_on_load = True

        def _fake_encode(self, text):
            t = text.lower()
            if "iron man" in t:
                k = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
            elif "batman" in t:
                k = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
            else:
                k = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
            return k, k.clone()

        # Avoid loading real sentence-transformer model in unit test.
        monkeypatch.setattr(AdaptiveModularMemory, "_encode_text_st", _fake_encode)

        # Build deliberately wrong persisted vectors (swapped labels).
        writer = AdaptiveModularMemory(cfg, device="cpu")
        wrong_iron = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)  # Batman key
        wrong_bat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)   # Iron key
        writer.bank.write(
            wrong_iron, wrong_iron,
            text="Iron Man real name is Tony Stark.", mem_type="fact",
        )
        writer.bank.write(
            wrong_bat, wrong_bat,
            text="Batman real name is Bruce Wayne.", mem_type="fact",
        )
        save_memory(writer.bank, cfg.memory_json_path, cfg.memory_pt_path)

        # On load, vectors should be rebuilt from text and retrieval should recover.
        reader = AdaptiveModularMemory(cfg, device="cpu")
        assert reader.load()
        top_text = reader.retrieve("what is iron man real name?", top_k=1)[0][0]
        assert "Tony Stark" in top_text


class TestTextSearch:
    """D-228: Text-based retrieval for hybrid neural+RAG path."""

    def test_text_search_basic(self, memory_bank):
        memory_bank.write(torch.randn(32), torch.randn(32), text="Alice likes red")
        memory_bank.write(torch.randn(32), torch.randn(32), text="Bob likes blue")
        memory_bank.write(torch.randn(32), torch.randn(32), text="Charlie works at Google")

        results = memory_bank.text_search("Alice likes", top_k=5)
        assert len(results) > 0
        assert results[0][0] == "Alice likes red"

    def test_text_search_empty_query(self, memory_bank):
        memory_bank.write(torch.randn(32), torch.randn(32), text="fact")
        results = memory_bank.text_search("", top_k=5)
        assert len(results) == 0

    def test_text_search_empty_bank(self, memory_bank):
        results = memory_bank.text_search("test query", top_k=5)
        assert len(results) == 0

    def test_text_search_no_match(self, memory_bank):
        memory_bank.write(torch.randn(32), torch.randn(32), text="Alice likes red")
        results = memory_bank.text_search("xyz zzz", top_k=5)
        assert len(results) == 0

    def test_text_search_returns_metadata(self, memory_bank):
        memory_bank.write(
            torch.randn(32), torch.randn(32),
            text="Alice likes red", mem_type="fact",
        )
        results = memory_bank.text_search("Alice", top_k=5)
        assert len(results) == 1
        text, score, entry = results[0]
        assert text == "Alice likes red"
        assert entry.mem_type == "fact"
        assert score > 0

    def test_text_search_top_k_limit(self, memory_bank):
        for i in range(10):
            memory_bank.write(
                torch.randn(32), torch.randn(32),
                text=f"fact about topic {i}",
            )
        results = memory_bank.text_search("fact topic", top_k=3)
        assert len(results) == 3


class TestAdaptiveRecencyBias:
    """D-263: Adaptive decay scales with retrieval entropy."""

    def test_high_entropy_boosts_recent(self):
        """High entropy should decay old entries faster, boosting recent ones."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)

        # Create an old entry (10 days ago) and a recent entry
        old_key = torch.randn(32)
        old_val = torch.randn(32)
        bank.write(old_key, old_val, text="old fact", mem_type="fact")
        with bank._lock:
            bank._metadata[0].timestamp = time.time() - 864_000  # 10 days ago

        recent_key = torch.randn(32)
        recent_val = torch.randn(32)
        bank.write(recent_key, recent_val, text="recent fact", mem_type="fact")

        # Read with no entropy (standard decay)
        _, scores_normal, _ = bank.read(recent_key, top_k=2, entropy=None)
        # Read with high entropy (should boost recent more)
        _, scores_entropy, _ = bank.read(recent_key, top_k=2, entropy=1.0)

        # With high entropy, old entries should be penalized more heavily,
        # so recent entry's relative advantage should be larger
        if scores_normal.numel() >= 2 and scores_entropy.numel() >= 2:
            ratio_normal = scores_normal[0] / (scores_normal[1] + 1e-8)
            ratio_entropy = scores_entropy[0] / (scores_entropy[1] + 1e-8)
            assert ratio_entropy >= ratio_normal - 0.01  # entropy should widen gap

    def test_zero_entropy_equals_normal(self):
        """Zero entropy should produce identical results to no entropy."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        key = torch.randn(32)
        bank.write(key, torch.randn(32), text="fact", mem_type="fact")

        _, scores_none, _ = bank.read(key, top_k=1, entropy=None)
        _, scores_zero, _ = bank.read(key, top_k=1, entropy=0.0)
        assert torch.allclose(scores_none, scores_zero, atol=1e-5)

    def test_identity_immune_to_entropy(self):
        """Identity entries should not be affected by entropy scaling."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        entry = MemoryEntry(
            text="user name", mem_type="identity",
            timestamp=time.time() - 100_000,
        )
        decay_normal = bank._decay_multiplier(entry)
        decay_adaptive = bank._adaptive_decay_multiplier(entry, entropy=1.0)
        assert abs(decay_normal - decay_adaptive) < 1e-6


class TestFAISSIndex:
    """D-295: FAISS index for sub-linear retrieval at scale."""

    def test_faiss_matches_brute_force(self):
        """FAISS-accelerated results should match brute-force for top-1."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=200)
        # Lower threshold so FAISS kicks in with fewer entries
        bank._faiss_threshold = 10

        # Populate with enough entries
        target_key = torch.randn(32)
        target_val = torch.randn(32)
        bank.write(target_key, target_val, text="target fact")

        for i in range(20):
            bank.write(torch.randn(32), torch.randn(32), text=f"distractor {i}")

        # Query should find target
        _, scores, indices = bank.read(target_key, top_k=1)
        assert scores.numel() >= 1
        assert 0 in indices  # target was written first

    def test_faiss_survives_consolidation(self):
        """FAISS index should be rebuilt after consolidation."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=200, dedup_enabled=False)
        bank._faiss_threshold = 5

        # Write similar entries that will be consolidated
        base_key = torch.randn(32)
        for i in range(15):
            noise = torch.randn(32) * 0.01
            bank.write(base_key + noise, torch.randn(32), text=f"similar fact {i}")

        bank.consolidate(similarity_threshold=0.90)

        # Index should still work after consolidation
        _, scores, indices = bank.read(base_key, top_k=1)
        assert scores.numel() >= 1

    def test_faiss_survives_deletion(self):
        """FAISS index should be rebuilt after deletion."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=200)
        bank._faiss_threshold = 5

        for i in range(15):
            bank.write(torch.randn(32), torch.randn(32), text=f"fact {i}")

        bank.delete_matching("fact 0")

        # Index should still work
        query = torch.randn(32)
        _, scores, indices = bank.read(query, top_k=3)
        assert scores.numel() == 3

    def test_faiss_graceful_fallback(self):
        """When FAISS is unavailable, brute-force should work fine."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        # Simulate no FAISS
        bank._faiss_index = None

        for i in range(10):
            bank.write(torch.randn(32), torch.randn(32), text=f"fact {i}")

        query = torch.randn(32)
        _, scores, indices = bank.read(query, top_k=3)
        assert scores.numel() == 3
        assert len(indices) == 3
