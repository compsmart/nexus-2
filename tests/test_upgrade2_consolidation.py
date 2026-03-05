"""Upgrade 2 TDD tests: Memory consolidation.

These tests MUST FAIL before implementation — they define the target behavior.

Memory consolidation clusters similar memories and compresses them into
summaries, freeing slots and enabling unbounded scaling. It triggers
before FIFO eviction so important memories aren't lost.

Research basis:
  AMM_SCALING_IDEAS.md #1: Memory consolidation — high impact, low effort
  Current limitation: 10K max_slots with FIFO eviction loses old memories
"""

import time

import pytest
import torch

from nexus2.memory.memory_bank import MemoryBank, MemoryEntry


class TestConsolidation:
    """Tests for memory consolidation in MemoryBank."""

    def test_consolidate_method_exists(self):
        """MemoryBank should have a consolidate() method."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        assert hasattr(bank, "consolidate"), "MemoryBank.consolidate() not implemented"
        assert callable(bank.consolidate)

    def test_consolidate_clusters_similar_memories(self):
        """Consolidation should merge memories with high cosine similarity."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)

        # Create a cluster of similar memories (same key direction, small perturbation)
        base_key = torch.randn(32)
        base_key = base_key / base_key.norm()  # normalize

        for i in range(5):
            # Add small noise to create near-duplicates
            key = base_key + torch.randn(32) * 0.01
            val = torch.randn(32)
            bank.write(key, val, text=f"Alice likes the color red variant {i}", mem_type="fact")

        # Add a distinct memory
        distinct_key = torch.randn(32)
        bank.write(distinct_key, torch.randn(32), text="Bob works at Google", mem_type="fact")

        assert bank.size == 6

        # Consolidate with high similarity threshold
        merged = bank.consolidate(similarity_threshold=0.95)

        # Should have merged the 5 similar entries into fewer
        assert bank.size < 6, f"Expected consolidation to reduce slots, got {bank.size}"
        assert merged > 0, "Expected at least one merge"

        # The distinct memory should survive
        with bank._lock:
            texts = [m.text for m in bank._metadata]
        assert any("Bob" in t for t in texts), "Distinct memory should survive consolidation"

    def test_consolidation_preserves_distinct_memories(self):
        """Memories with low similarity should not be merged."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)

        # Store completely different memories
        for i, fact in enumerate(["Alice likes red", "Bob likes blue", "Charlie works at Google"]):
            key = torch.randn(32)  # random = uncorrelated
            bank.write(key, torch.randn(32), text=fact, mem_type="fact")

        assert bank.size == 3
        merged = bank.consolidate(similarity_threshold=0.95)

        # Nothing should merge — they're unrelated
        assert merged == 0
        assert bank.size == 3

    def test_consolidation_frees_slots(self):
        """After consolidation, freed slots should allow new writes without FIFO eviction."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=10, dedup_enabled=False)

        # Fill to capacity with similar memories
        base_key = torch.randn(32)
        base_key = base_key / base_key.norm()
        for i in range(10):
            key = base_key + torch.randn(32) * 0.01
            bank.write(key, torch.randn(32), text=f"similar fact {i}", mem_type="fact")

        assert bank.size == 10

        # Consolidate — should free slots
        merged = bank.consolidate(similarity_threshold=0.90)
        assert merged > 0
        slots_after = bank.size
        assert slots_after < 10

        # Now we should be able to write new entries without evicting
        bank.write(torch.randn(32), torch.randn(32), text="brand new fact", mem_type="fact")
        with bank._lock:
            texts = [m.text for m in bank._metadata]
        assert "brand new fact" in texts

    def test_consolidated_entry_has_summary_text(self):
        """Merged entries should produce a readable summary, not raw concatenation."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)

        base_key = torch.randn(32)
        base_key = base_key / base_key.norm()
        for i in range(5):
            key = base_key + torch.randn(32) * 0.01
            bank.write(key, torch.randn(32), text=f"Alice likes red variant {i}", mem_type="fact")

        bank.consolidate(similarity_threshold=0.90)

        # Find the consolidated entry
        with bank._lock:
            consolidated = [m for m in bank._metadata if m.extra.get("consolidated")]

        assert len(consolidated) > 0, "Should have at least one consolidated entry"
        entry = consolidated[0]
        # Summary should reference the original content
        assert len(entry.text) > 0, "Consolidated entry should have non-empty text"
        # Should track how many entries were merged
        assert entry.extra.get("merged_count", 0) >= 2, "Should track merge count"

    def test_consolidation_preserves_memory_types(self):
        """Consolidation should only merge within the same mem_type."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)

        # Same key direction but different types
        base_key = torch.randn(32)
        base_key = base_key / base_key.norm()

        for i in range(3):
            key = base_key + torch.randn(32) * 0.01
            bank.write(key, torch.randn(32), text=f"fact about Alice {i}", mem_type="fact")

        for i in range(3):
            key = base_key + torch.randn(32) * 0.01
            bank.write(key, torch.randn(32), text=f"user said about Alice {i}", mem_type="user_input")

        assert bank.size == 6
        bank.consolidate(similarity_threshold=0.90)

        # Both types should still be represented
        with bank._lock:
            types = set(m.mem_type for m in bank._metadata)
        assert "fact" in types
        assert "user_input" in types

    def test_consolidation_preserves_identity_type(self):
        """Identity memories should never be consolidated (they never decay either)."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)

        base_key = torch.randn(32)
        base_key = base_key / base_key.norm()

        # Identity memories with similar keys
        for i in range(3):
            key = base_key + torch.randn(32) * 0.01
            bank.write(key, torch.randn(32), text=f"my name is Alice {i}", mem_type="identity")

        initial_size = bank.size
        bank.consolidate(similarity_threshold=0.90)

        # Identity entries should not be consolidated
        assert bank.size == initial_size, "Identity memories must not be consolidated"

    def test_eviction_triggers_consolidation(self):
        """When at capacity, writing should attempt consolidation before FIFO eviction."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=10, dedup_enabled=False)

        # Fill with a mix: 5 similar + 5 distinct
        base_key = torch.randn(32)
        base_key = base_key / base_key.norm()
        for i in range(5):
            key = base_key + torch.randn(32) * 0.01
            bank.write(key, torch.randn(32), text=f"similar fact {i}", mem_type="fact")
        for i in range(5):
            bank.write(torch.randn(32), torch.randn(32), text=f"unique fact {i}", mem_type="fact")

        assert bank.size == 10

        # This write would normally trigger FIFO eviction of "similar fact 0"
        # With consolidation-before-eviction, the similar cluster should merge first
        bank.write(torch.randn(32), torch.randn(32), text="new important fact", mem_type="fact")

        with bank._lock:
            texts = [m.text for m in bank._metadata]

        # The new fact should be present
        assert "new important fact" in texts

        # If consolidation ran, we should still have some unique facts that
        # FIFO would have evicted without consolidation
        unique_count = sum(1 for t in texts if "unique fact" in t)
        assert unique_count >= 4, (
            f"Expected consolidation to preserve unique facts, "
            f"but only {unique_count}/5 survived"
        )

    def test_consolidate_empty_bank(self):
        """Consolidating an empty bank should be a no-op."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        merged = bank.consolidate(similarity_threshold=0.90)
        assert merged == 0
        assert bank.size == 0

    def test_consolidate_single_entry(self):
        """Consolidating a bank with one entry should be a no-op."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        bank.write(torch.randn(32), torch.randn(32), text="only fact")
        merged = bank.consolidate(similarity_threshold=0.90)
        assert merged == 0
        assert bank.size == 1

    def test_consolidated_key_is_centroid(self):
        """The key of a consolidated entry should be the centroid of merged keys."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)

        base_key = torch.randn(32)
        base_key = base_key / base_key.norm()
        keys = []
        for i in range(5):
            key = base_key + torch.randn(32) * 0.01
            keys.append(key.clone())
            bank.write(key, torch.randn(32), text=f"similar {i}", mem_type="fact")

        bank.consolidate(similarity_threshold=0.90)

        # Find the consolidated entry's key
        with bank._lock:
            consolidated_idx = None
            for idx, m in enumerate(bank._metadata):
                if m.extra.get("consolidated"):
                    consolidated_idx = idx
                    break

        assert consolidated_idx is not None, "Should have a consolidated entry"

        with bank._lock:
            result_key = bank._keys[consolidated_idx]

        # Centroid should be close to mean of original keys
        expected_centroid = torch.stack(keys).mean(dim=0)
        expected_centroid = expected_centroid / expected_centroid.norm()
        result_norm = result_key / result_key.norm()

        cos_sim = torch.dot(expected_centroid, result_norm)
        assert cos_sim > 0.95, f"Consolidated key should be near centroid, got cos_sim={cos_sim:.4f}"
