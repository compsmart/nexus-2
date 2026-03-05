"""Multi-hop reasoning chain tests (2-5 hop), entropy penalty, curriculum engine."""

import pytest
import torch

from nexus2.reasoning.nhop_reader import ExplicitNReadChain, SharedNReadChain
from nexus2.reasoning.chain_executor import ChainExecutor, ReasoningResult
from nexus2.learning.curriculum_engine import CurriculumEngine


class TestExplicitNReadChain:
    """Tests for explicit per-hop projection reader."""

    def test_forward_shapes(self):
        reader = ExplicitNReadChain(d_key=32, d_val=32, n_hops=3, n_entities=100)
        query = torch.randn(2, 32)
        keys = torch.randn(2, 10, 32)
        values = torch.randn(2, 10, 32)

        state, logits, attn = reader(query, keys, values)
        assert state.shape == (2, 32)
        assert len(logits) == 3
        assert logits[0].shape == (2, 100)
        assert len(attn) == 3
        assert attn[0].shape == (2, 10)

    def test_attention_sums_to_one(self):
        reader = ExplicitNReadChain(d_key=32, d_val=32, n_hops=2, n_entities=50)
        query = torch.randn(1, 32)
        keys = torch.randn(1, 5, 32)
        values = torch.randn(1, 5, 32)

        _, _, attn = reader(query, keys, values)
        for a in attn:
            assert torch.allclose(a.sum(dim=-1), torch.ones(1), atol=1e-5)

    def test_per_hop_projections_are_different(self):
        reader = ExplicitNReadChain(d_key=32, d_val=32, n_hops=3, n_entities=50)
        # Each hop should have its own projection
        assert len(reader.query_projs) == 3
        # Weights should be different
        w0 = reader.query_projs[0].weight.data
        w1 = reader.query_projs[1].weight.data
        assert not torch.allclose(w0, w1)

    def test_2hop_through_5hop(self):
        for n_hops in [2, 3, 4, 5]:
            reader = ExplicitNReadChain(
                d_key=32, d_val=32, n_hops=n_hops, n_entities=100,
            )
            query = torch.randn(1, 32)
            keys = torch.randn(1, 20, 32)
            values = torch.randn(1, 20, 32)

            state, logits, attn = reader(query, keys, values)
            assert len(logits) == n_hops
            assert len(attn) == n_hops


class TestSharedNReadChain:
    """Tests for shared-weight reader."""

    def test_forward_shapes(self):
        reader = SharedNReadChain(d_key=32, d_val=32, max_hops=5, n_entities=100)
        query = torch.randn(1, 32)
        keys = torch.randn(1, 10, 32)
        values = torch.randn(1, 10, 32)

        state, logits, attn = reader(query, keys, values, n_hops=3)
        assert state.shape == (1, 32)
        assert len(logits) == 3
        assert len(attn) == 3

    def test_variable_hops(self):
        reader = SharedNReadChain(d_key=32, d_val=32, max_hops=5, n_entities=50)
        query = torch.randn(1, 32)
        keys = torch.randn(1, 8, 32)
        values = torch.randn(1, 8, 32)

        for n in [2, 3, 4, 5]:
            _, logits, _ = reader(query, keys, values, n_hops=n)
            assert len(logits) == n


class TestChainExecutor:
    """Tests for the full chain executor."""

    def test_forward_known(self, chain_executor):
        query = torch.randn(1, 32)
        keys = torch.randn(1, 10, 32)
        values = torch.randn(1, 10, 32)

        result = chain_executor(query, keys, values)
        assert isinstance(result, ReasoningResult)
        assert result.retrieval_vectors is not None
        assert result.route in ("known", "novel")
        assert len(result.intermediate_logits) > 0

    def test_empty_memory(self, chain_executor):
        query = torch.randn(1, 32)
        keys = torch.randn(1, 0, 32)
        values = torch.randn(1, 0, 32)

        result = chain_executor(query, keys, values)
        assert result.route == "novel"
        assert result.confidence == 0.0

    def test_compute_loss(self, chain_executor):
        query = torch.randn(1, 32)
        keys = torch.randn(1, 10, 32)
        values = torch.randn(1, 10, 32)

        result = chain_executor(query, keys, values, n_hops=3)
        targets = [0, 1, 2]  # dummy targets

        loss = chain_executor.compute_loss(result, targets, supervision_weight=1.0)
        assert loss.item() > 0
        assert loss.requires_grad

    def test_supervision_weight_guard(self, chain_executor):
        """Must reject supervision_weight < 0.5."""
        query = torch.randn(1, 32)
        keys = torch.randn(1, 10, 32)
        values = torch.randn(1, 10, 32)

        result = chain_executor(query, keys, values, n_hops=2)
        targets = [0, 1]

        with pytest.raises(AssertionError, match="supervision weight"):
            chain_executor.compute_loss(result, targets, supervision_weight=0.3)

    def test_attention_entropy_computation(self, chain_executor):
        """Entropy should be non-negative for valid attention distributions."""
        # Uniform attention
        attn = torch.ones(1, 10) / 10.0
        entropy = ChainExecutor._attention_entropy(attn)
        assert entropy.item() > 0

        # One-hot attention (entropy should be ~0)
        attn_sharp = torch.zeros(1, 10)
        attn_sharp[0, 0] = 1.0
        entropy_sharp = ChainExecutor._attention_entropy(attn_sharp)
        assert entropy_sharp.item() < entropy.item()

    def test_entropy_penalty_reduces_loss(self, chain_executor):
        """With positive entropy_lambda, loss should differ from base."""
        query = torch.randn(1, 32)
        keys = torch.randn(1, 10, 32)
        values = torch.randn(1, 10, 32)

        result = chain_executor(query, keys, values, n_hops=2)
        targets = [0, 1]

        # Base loss (no entropy penalty)
        chain_executor.config.entropy_lambda = 0.0
        loss_base = chain_executor.compute_loss(result, targets).item()

        # With entropy penalty
        chain_executor.config.entropy_lambda = 0.01
        loss_ent = chain_executor.compute_loss(result, targets).item()

        # They should differ (entropy term subtracts from loss)
        assert loss_base != loss_ent


class TestCurriculumEngine:
    """Tests for curriculum engine with encoder-type safety guards (D-188)."""

    def test_lstm_caps_at_500(self):
        """LSTM encoder must cap k-schedule at 500."""
        engine = CurriculumEngine(
            k_schedule=[5, 10, 100, 500, 750, 1000],
            encoder_type="lstm",
        )
        assert max(engine.k_schedule) <= 500
        assert 750 not in engine.k_schedule
        assert 1000 not in engine.k_schedule

    def test_mamba_allows_past_500(self):
        """Mamba encoder can use k > 500."""
        engine = CurriculumEngine(
            k_schedule=[5, 10, 100, 500, 750, 1000],
            encoder_type="mamba",
        )
        assert 750 in engine.k_schedule
        assert 1000 in engine.k_schedule
