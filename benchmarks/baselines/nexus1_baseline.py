"""NEXUS-1 baseline: wraps existing AGIAgent from nexus-1/."""

import logging
import sys
import os


class Nexus1Baseline:
    """Wraps the existing nexus-1 AGIAgent for comparison."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._agent = None

    def _ensure_loaded(self):
        if self._agent is not None:
            return
        try:
            # Add nexus-1 to path
            agi_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "nexus-1",
            ))
            if agi_path not in sys.path:
                sys.path.insert(0, agi_path)

            from agent import NexusAgent
            from config import AgentConfig

            config = AgentConfig()
            config.device = self.device
            self._agent = NexusAgent(config)
        except Exception as e:
            logging.warning("Failed to load NEXUS-1 agent: %s", e)

    def reset(self):
        """Reset agent memory."""
        self._ensure_loaded()
        if self._agent is not None:
            try:
                self._agent.memory.clear()
            except Exception:
                pass

    def teach(self, text: str):
        """Teach via interact."""
        self._ensure_loaded()
        if self._agent is not None:
            try:
                self._agent.interact(text)
            except Exception:
                pass

    def query(self, text: str) -> str:
        """Query via interact."""
        self._ensure_loaded()
        if self._agent is None:
            return ""
        try:
            return self._agent.interact(text)
        except Exception:
            return ""
