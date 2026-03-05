"""Upgrade 3 TDD tests: Graph-structured memory.

These tests MUST FAIL before implementation — they define the target behavior.

Graph memory adds an edge overlay to MemoryBank. When relational facts
are stored ("Alice knows Bob"), an edge is created between the entities.
Multi-hop queries traverse edges instead of relying on brittle regex
entity bridging.

Research basis:
  AMM_SCALING_IDEAS.md #2: Graph-structured memory — medium effort, high impact
  Current limitation: regex-based hop-2 bridging is brittle and doesn't generalize
"""

import pytest
import torch

from nexus2.memory.memory_bank import MemoryBank, MemoryEntry


class TestGraphEdges:
    """Tests for edge storage and retrieval in MemoryBank."""

    def test_add_edge_method_exists(self):
        """MemoryBank should have an add_edge() method."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        assert hasattr(bank, "add_edge"), "MemoryBank.add_edge() not implemented"
        assert callable(bank.add_edge)

    def test_get_edges_method_exists(self):
        """MemoryBank should have a get_edges() method."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        assert hasattr(bank, "get_edges"), "MemoryBank.get_edges() not implemented"
        assert callable(bank.get_edges)

    def test_store_creates_edge(self):
        """Storing a relational fact should create an edge between entities."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)

        idx_alice = bank.write(torch.randn(32), torch.randn(32),
                               text="Alice knows Bob", mem_type="fact", subject="Alice")
        bank.add_edge(source="Alice", target="Bob", relation="knows")

        edges = bank.get_edges(entity="Alice")
        assert len(edges) > 0
        assert any(e["target"] == "Bob" and e["relation"] == "knows" for e in edges)

    def test_edge_bidirectional_query(self):
        """Edges should be queryable from both directions."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        bank.add_edge(source="Alice", target="Bob", relation="knows")

        # Forward: Alice -> Bob
        fwd = bank.get_edges(entity="Alice")
        assert any(e["target"] == "Bob" for e in fwd)

        # Reverse: Bob -> Alice
        rev = bank.get_edges(entity="Bob")
        assert any(e["target"] == "Alice" for e in rev)

    def test_multiple_edges_from_same_entity(self):
        """An entity can have multiple outgoing edges."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        bank.add_edge(source="Alice", target="Bob", relation="knows")
        bank.add_edge(source="Alice", target="Charlie", relation="knows")
        bank.add_edge(source="Alice", target="red", relation="likes")

        edges = bank.get_edges(entity="Alice")
        assert len(edges) == 3
        targets = {e["target"] for e in edges}
        assert targets == {"Bob", "Charlie", "red"}

    def test_no_edges_for_unknown_entity(self):
        """Querying edges for an unknown entity should return empty list."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        edges = bank.get_edges(entity="Unknown")
        assert edges == []


class TestGraphTraversal:
    """Tests for multi-hop graph traversal."""

    def test_traverse_method_exists(self):
        """MemoryBank should have a traverse() method."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        assert hasattr(bank, "traverse"), "MemoryBank.traverse() not implemented"
        assert callable(bank.traverse)

    def test_traverse_2hop(self):
        """2-hop traversal: Alice->Bob->Charlie via 'knows' edges."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        bank.add_edge(source="Alice", target="Bob", relation="knows")
        bank.add_edge(source="Bob", target="Charlie", relation="knows")

        # Traverse 2 hops from Alice via 'knows'
        result = bank.traverse(start="Alice", relation="knows", hops=2)
        assert "Charlie" in result, f"Expected Charlie in 2-hop result, got {result}"

    def test_traverse_3hop(self):
        """3-hop traversal: A->B->C->D."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        bank.add_edge(source="A", target="B", relation="knows")
        bank.add_edge(source="B", target="C", relation="knows")
        bank.add_edge(source="C", target="D", relation="knows")

        result = bank.traverse(start="A", relation="knows", hops=3)
        assert "D" in result

    def test_traverse_5hop(self):
        """5-hop chain traversal."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        entities = ["E0", "E1", "E2", "E3", "E4", "E5"]
        for i in range(len(entities) - 1):
            bank.add_edge(source=entities[i], target=entities[i + 1], relation="knows")

        result = bank.traverse(start="E0", relation="knows", hops=5)
        assert "E5" in result

    def test_traverse_1hop(self):
        """1-hop is just a direct edge lookup."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        bank.add_edge(source="Alice", target="Bob", relation="knows")

        result = bank.traverse(start="Alice", relation="knows", hops=1)
        assert "Bob" in result

    def test_traverse_no_path(self):
        """Traversal with no valid path should return empty."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        bank.add_edge(source="Alice", target="Bob", relation="knows")
        # No edge from Bob onwards

        result = bank.traverse(start="Alice", relation="knows", hops=2)
        assert result == []

    def test_traverse_handles_cycles(self):
        """Cycles should not cause infinite loops."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        bank.add_edge(source="A", target="B", relation="knows")
        bank.add_edge(source="B", target="C", relation="knows")
        bank.add_edge(source="C", target="A", relation="knows")  # cycle

        # Should terminate and not hang
        result = bank.traverse(start="A", relation="knows", hops=3)
        # After 3 hops: A->B->C->A (cycle back), result is A
        assert isinstance(result, list)

    def test_traverse_mixed_relations(self):
        """Only follow edges matching the specified relation."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        bank.add_edge(source="Alice", target="Bob", relation="knows")
        bank.add_edge(source="Alice", target="red", relation="likes")
        bank.add_edge(source="Bob", target="Charlie", relation="knows")

        # Only follow 'knows' — should not reach 'red'
        result = bank.traverse(start="Alice", relation="knows", hops=2)
        assert "Charlie" in result
        assert "red" not in result


class TestGraphWithFlatMemory:
    """Tests that graph overlay coexists with flat cosine retrieval."""

    def test_flat_retrieval_still_works(self):
        """Non-relational facts should still be retrievable via cosine similarity."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)

        key = torch.randn(32)
        bank.write(key, torch.randn(32), text="Alice likes red", mem_type="fact")
        bank.add_edge(source="Alice", target="Bob", relation="knows")

        # Flat cosine retrieval should still work
        values, weights, indices = bank.read(key, top_k=1)
        assert values.shape[0] == 1

    def test_edges_survive_clear(self):
        """Clearing memory entries should also clear edges."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        bank.add_edge(source="Alice", target="Bob", relation="knows")
        bank.write(torch.randn(32), torch.randn(32), text="some fact")

        bank.clear()
        assert bank.size == 0
        assert bank.get_edges(entity="Alice") == []

    def test_graph_scales_to_200_entities(self):
        """Graph should handle 200 entities with edges efficiently."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=500)

        # Create a chain of 200 entities
        for i in range(200):
            bank.write(torch.randn(32), torch.randn(32),
                       text=f"Entity_{i} knows Entity_{i+1}", mem_type="fact")
            bank.add_edge(source=f"Entity_{i}", target=f"Entity_{i+1}", relation="knows")

        # 3-hop traversal from Entity_0
        result = bank.traverse(start="Entity_0", relation="knows", hops=3)
        assert "Entity_3" in result

        # Verify edge count
        edges = bank.get_edges(entity="Entity_100")
        assert len(edges) >= 1  # at least the outgoing edge

    def test_edges_snapshot_for_persistence(self):
        """get_snapshot should include edge data for persistence."""
        bank = MemoryBank(d_key=32, d_val=32, max_slots=100)
        bank.add_edge(source="Alice", target="Bob", relation="knows")
        bank.write(torch.randn(32), torch.randn(32), text="test")

        keys, vals, meta = bank.get_snapshot()
        # The bank should be able to reconstruct edges after snapshot/restore
        # This tests that edge state is either in metadata or separately serializable
        assert hasattr(bank, '_edges') or any(
            m.extra.get("edges") for m in meta
        ), "Edge data should be accessible for persistence"
