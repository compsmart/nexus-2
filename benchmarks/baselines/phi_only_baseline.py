"""LLM-only baseline: no memory system, just raw LLM."""

import logging


class LLMOnlyBaseline:
    """Raw LLM with no memory augmentation."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._llm = None
        self._context: list = []

    def _ensure_loaded(self):
        if self._llm is None:
            try:
                from nexus2.generation.llm_engine import LLMEngine
                self._llm = LLMEngine(
                    model_name=self.model_name,
                    device=self.device,
                    use_4bit=(self.device == "cuda"),
                )
            except Exception as e:
                logging.warning("LLM load failed: %s", e)

    def reset(self):
        """Reset conversation context."""
        self._context = []

    def teach(self, text: str):
        """Add fact to conversation context (within context window)."""
        self._context.append(text)
        # Keep only last N facts to fit context window
        if len(self._context) > 50:
            self._context = self._context[-50:]

    def query(self, text: str) -> str:
        """Query with all context in the prompt."""
        self._ensure_loaded()
        if self._llm is None:
            return ""

        context = "\n".join(f"- {f}" for f in self._context)
        messages = [
            {"role": "system", "content": f"Previously stated facts:\n{context}"},
            {"role": "user", "content": text},
        ]
        return self._llm.chat(messages, max_new_tokens=100)


# Backward compatibility alias
PhiOnlyBaseline = LLMOnlyBaseline
