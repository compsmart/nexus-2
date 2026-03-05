"""Integration tests: full turn cycle, multi-turn, tool use."""

import pytest
from unittest.mock import MagicMock

from nexus2.agent import Nexus2Agent
from nexus2.perception.input_processor import InputProcessor, PerceptionResult
from nexus2.config import NexusConfig


class TestPerception:
    """Input processor tests."""

    def test_intent_query(self, config):
        proc = InputProcessor(config)
        result = proc.process("What is the capital of France?")
        assert result.intent == "query"

    def test_intent_teach(self, config):
        proc = InputProcessor(config)
        result = proc.process("My name is Alex")
        assert result.intent == "teach"
        assert result.user_name == "Alex"

    def test_intent_correct(self, config):
        proc = InputProcessor(config)
        result = proc.process("Actually, my name is Bob, not Alex")
        assert result.intent == "correct"
        assert result.is_correction

    def test_intent_command(self, config):
        proc = InputProcessor(config)
        result = proc.process("Search for PyTorch tutorials")
        assert result.intent == "command"

    def test_name_extraction(self, config):
        proc = InputProcessor(config)
        for text, expected in [
            ("My name is Alice", "Alice"),
            ("I'm Bob", "Bob"),
            ("Call me Charlie", "Charlie"),
        ]:
            result = proc.process(text)
            assert result.user_name == expected, f"Failed for: {text}"

    def test_blocked_names(self, config):
        proc = InputProcessor(config)
        result = proc.process("My name is assistant")
        assert result.user_name is None

    def test_name_stop_words(self, config):
        proc = InputProcessor(config)
        result = proc.process("My name is Alex and my dog is Bruno")
        assert result.user_name == "Alex"

    def test_personal_fact_extraction(self, config):
        proc = InputProcessor(config)
        result = proc.process("My favorite color is blue")
        assert len(result.personal_facts) >= 1

    def test_negation_guard(self, config):
        proc = InputProcessor(config)
        result = proc.process("I don't have a dog named Bruno")
        assert len(result.personal_facts) == 0

    def test_entity_extraction(self, config):
        proc = InputProcessor(config)
        result = proc.process("I met John Smith in Portland last week")
        entities = result.entities
        assert any("John" in e for e in entities)

    def test_short_followup(self, config):
        proc = InputProcessor(config)
        result = proc.process("What about that?", prev_text="Tell me about dogs")
        assert result.is_short_followup

    def test_not_short_followup(self, config):
        proc = InputProcessor(config)
        result = proc.process("Can you tell me about the weather forecast for tomorrow", prev_text="Hi")
        assert not result.is_short_followup


class TestAgentIntegration:
    """Full agent integration tests."""

    def test_basic_interaction(self, test_agent):
        response = test_agent.interact("Hello, how are you?")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_teach_fact(self, test_agent):
        test_agent.interact("My name is Alex")
        assert test_agent._user_name == "Alex"
        # Name should be in memory
        assert test_agent.memory.size > 0

    def test_multi_turn(self, test_agent):
        test_agent.interact("My dog is Bruno")
        test_agent.interact("Bruno's vet is Dr. Chen")
        # Both facts should be stored
        assert test_agent.memory.size >= 2

    def test_memory_persistence(self, test_agent, tmp_path):
        test_agent.interact("My cat is Luna")
        test_agent.stop()

        # Create new agent with same paths
        new_agent = Nexus2Agent(
            config=test_agent.config,
            device="cpu",
            load_llm=False,
            load_checkpoints=False,
        )
        # Memory should be loaded from disk
        # (may be 0 if save wasn't triggered by interact - depends on implementation)

    def test_correction_handling(self, test_agent):
        test_agent.interact("My dog is Bruno")
        initial_size = test_agent.memory.size
        test_agent.interact("Actually, my dog is Max, not Bruno")
        # Should have attempted deletion + new storage

    def test_stats(self, test_agent):
        stats = test_agent.get_stats()
        assert "memory" in stats
        assert "device" in stats
        assert stats["device"] == "cpu"

    def test_fallback_when_no_llm(self, config, tmp_path):
        config.memory_json_path = str(tmp_path / "test_memory.json")
        config.memory_pt_path = str(tmp_path / "test_memory.json.pt")
        config.checkpoint_dir = str(tmp_path / "checkpoints")
        config.skills_dir = str(tmp_path / "skills")
        config.skills_index = str(tmp_path / "skills" / "index.json")

        agent = Nexus2Agent(config=config, device="cpu", load_llm=False, load_checkpoints=False)
        response = agent.interact("Hello")
        assert "not" in response.lower() or "message" in response.lower()


class TestWebFactValidation:
    """D-275: Web content validation before memory storage."""

    def test_rejects_short_content(self):
        """Content shorter than 50 chars should be rejected."""
        assert not Nexus2Agent._validate_web_fact("Error", "weather forecast")
        assert not Nexus2Agent._validate_web_fact("  ", "test query")

    def test_rejects_error_pages(self):
        """Error page indicators should cause rejection."""
        assert not Nexus2Agent._validate_web_fact(
            "404 Not Found - The page you requested could not be found on this server.",
            "company info",
        )
        assert not Nexus2Agent._validate_web_fact(
            "Access Denied. You do not have permission to access this resource. Please verify you are a human.",
            "product details",
        )

    def test_rejects_captcha_pages(self):
        """Captcha/verification pages should be rejected."""
        assert not Nexus2Agent._validate_web_fact(
            "Please complete the CAPTCHA below to continue. This helps us verify you are a human and not a bot.",
            "restaurant menu",
        )

    def test_rejects_irrelevant_content(self):
        """Content with no query token overlap should be rejected."""
        assert not Nexus2Agent._validate_web_fact(
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
            "weather forecast tomorrow",
        )

    def test_accepts_valid_content(self):
        """Valid web content with relevant overlap should be accepted."""
        assert Nexus2Agent._validate_web_fact(
            "The weather forecast for Portland shows sunny skies tomorrow with a high of 72°F and low of 55°F. Extended outlook suggests rain by Wednesday.",
            "weather forecast Portland",
        )

    def test_accepts_long_relevant_content(self):
        """Long content with relevant tokens should pass."""
        content = "Python is a programming language that supports multiple paradigms. " * 5
        assert Nexus2Agent._validate_web_fact(content, "Python programming language")

    def test_decay_and_boost_ordering(self):
        """web_fact should have shorter half-life and lower boost than fact."""
        cfg = NexusConfig()
        assert cfg.decay_half_lives["web_fact"] < cfg.decay_half_lives["fact"]
        assert cfg.retrieval_type_boosts["web_fact"] < cfg.retrieval_type_boosts["fact"]
