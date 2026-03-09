"""NEXUS-2 baseline adapter for benchmarks.

Clean agent pass-through -- all queries go through the real Nexus2Agent pipeline.
No shortcuts or pattern-matching; the agent handles everything.
"""

import sys
from pathlib import Path

# Ensure nexus-2 root is importable
_NEXUS2_DIR = Path(__file__).resolve().parent.parent.parent
if str(_NEXUS2_DIR) not in sys.path:
    sys.path.insert(0, str(_NEXUS2_DIR))

from nexus2.agent import Nexus2Agent
from nexus2.config import NexusConfig


class Nexus2Baseline:
    """Wraps Nexus2Agent for benchmark interface.

    All queries pass through the real agent pipeline (perception, memory,
    reasoning, generation). No shortcuts or regex pattern matching.
    """

    agent_name = "nexus-2"

    def __init__(self, config=None, device="cpu", load_llm=True):
        self.config = config or NexusConfig()
        self.device = device
        self._load_llm = load_llm
        self._agent = None
        self._init_agent()

    def _init_agent(self):
        self._agent = Nexus2Agent(
            config=self.config,
            device=self.device,
            load_llm=self._load_llm,
            load_checkpoints=True,
        )

    def reset(self):
        """Reset memory state."""
        self._agent.memory.bank.clear()

    def teach(self, text: str):
        """Teach a fact to the agent."""
        self._agent.memory.store(text, mem_type="fact")

    def query(self, text: str) -> str:
        """Query the agent -- full pipeline, no shortcuts."""
        return self._agent.interact(text)
