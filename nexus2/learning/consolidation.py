"""Background reflection and memory consolidation.

Runs periodically during idle time to:
  - Sample decay-at-risk memories
  - Generate LLM insights
  - Store insights back to memory
"""

import logging
import threading
import time
from typing import Optional

from ..config import NexusConfig
from ..memory.amm import AdaptiveModularMemory


class ConsolidationLoop:
    """Background thread for memory reflection and consolidation."""

    def __init__(
        self,
        config: NexusConfig,
        memory: AdaptiveModularMemory,
        llm=None,
    ):
        self.config = config
        self.memory = memory
        self.llm = llm
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_interaction = time.time()
        self._last_flush = time.time()

    def start(self):
        """Start the background consolidation loop."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the consolidation loop and flush memory."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        # Final flush
        try:
            self.memory.save()
        except Exception as e:
            logging.error("Final memory flush failed: %s", e)

    def touch(self):
        """Record user interaction (resets idle timer)."""
        self._last_interaction = time.time()

    def _loop(self):
        """Main loop: reflect when idle, flush periodically."""
        while not self._stop_event.wait(timeout=5.0):
            now = time.time()
            idle_secs = now - self._last_interaction

            # Reflect if idle long enough
            if idle_secs >= self.config.think_interval_secs:
                try:
                    self._reflect()
                except Exception as e:
                    logging.error("Reflection error: %s", e)

            # Periodic flush
            if now - self._last_flush >= self.config.flush_interval_secs:
                if self.memory.bank.dirty:
                    try:
                        self.memory.save()
                        self._last_flush = now
                    except Exception as e:
                        logging.error("Periodic flush error: %s", e)

    def _reflect(self):
        """Sample memories and generate insights."""
        if self.llm is None or self.memory.size < 5:
            return

        # Sample a few memories weighted toward decay-risk
        results = self.memory.retrieve("summary of recent knowledge", top_k=5)
        if not results:
            return

        memory_texts = [text for text, _, _ in results]
        context = "\n".join(f"- {t}" for t in memory_texts)

        messages = [
            {"role": "system", "content": (
                "You are a reflection assistant. Given these memory entries, "
                "generate a brief insight or connection between them. "
                "Output only the insight in one sentence."
            )},
            {"role": "user", "content": f"Memories:\n{context}"},
        ]

        try:
            insight = self.llm.chat(messages, max_new_tokens=100, temperature=0.5)
            if insight and len(insight) > 10:
                self.memory.store(
                    insight,
                    mem_type="fact",
                    subject="reflection_insight",
                )
        except Exception as e:
            logging.debug("Reflection LLM call failed: %s", e)
