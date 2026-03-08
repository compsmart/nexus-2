"""Clean adapter wrapping the actual Nexus-2 agent for benchmarking.

No shortcuts -- all queries go through the real agent pipeline.
This replaces the old Nexus2Baseline which had ~400 lines of regex
pattern matching that bypassed the agent entirely.
"""

import sys
from pathlib import Path

# Ensure nexus-2 root is importable
_NEXUS2_DIR = Path(__file__).resolve().parent.parent
if str(_NEXUS2_DIR) not in sys.path:
    sys.path.insert(0, str(_NEXUS2_DIR))


class Nexus2Adapter:
    """Wraps the actual Nexus2Agent for benchmark evaluation."""

    agent_name = "nexus-2"

    def __init__(self, device="cpu", **kwargs):
        self.device = device
        self._agent = None
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"

    def _ensure_loaded(self):
        if self._agent is not None:
            return
        from nexus2.config import NexusConfig
        from nexus2.agent import Nexus2Agent
        cfg = NexusConfig()
        self._agent = Nexus2Agent(config=cfg, device=self.device)

    def reset(self):
        self._ensure_loaded()
        # Clear the adaptive modular memory
        self._agent.memory.bank.clear()
        if hasattr(self._agent.memory, '_embedding_cache'):
            self._agent.memory._embedding_cache.clear()
        if hasattr(self._agent.memory, 'text_index'):
            self._agent.memory.text_index.clear()

    def teach(self, text: str):
        self._ensure_loaded()
        self._agent.memory.store(text, mem_type="fact")

    def query(self, text: str) -> str:
        self._ensure_loaded()
        return self._agent.interact(text)
