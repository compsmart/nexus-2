"""Full generation pipeline: AMM chain -> adapter -> prepend soft prompts -> LLM.

D-197 route_level routing:
  INJECT_FULL:   full soft-prompt path (all retrieval vectors)
  INJECT_TOP1:   soft-prompt path with only top-1 retrieval vector
  INJECT_HEDGED: text-context path with hedging qualifier
  SKIP:          text-context path without memory injection
"""

import logging
from typing import List, Optional

import torch

from ..config import NexusConfig
from ..reasoning.chain_executor import ReasoningResult
from .llm_engine import LLMEngine
from .soft_prompt_adapter import SoftPromptAdapter

_HEDGE_PREFIX = (
    "[Note: The following memory may be incomplete or slightly outdated. "
    "Use it as a hint, not a definitive answer.]\n"
)

# D-227: Rejection message when confidence is too low to answer reliably
_REJECT_RESPONSE = (
    "I don't have reliable information about that in my memory. "
    "I'd rather not guess — could you teach me or rephrase your question?"
)


class ResponseGenerator:
    """Generates responses by combining AMM retrieval with frozen LLM.

    Five paths (D-197/D-227 route_level):
    1. INJECT_FULL:   retrieval_vec -> SoftPromptAdapter -> prepend to LLM embeds
    2. INJECT_TOP1:   same path but with reduced retrieval signal
    3. INJECT_HEDGED: text-context prompting with hedging qualifier
    4. SKIP:          text-context prompting without memory context
    5. REJECT:        D-227 explicit refusal (≤0.1% hallucination guarantee)
    """

    def __init__(
        self,
        config: NexusConfig,
        llm: LLMEngine,
        adapter: Optional[SoftPromptAdapter] = None,
    ):
        self.config = config
        self.llm = llm
        self.adapter = adapter

    def generate(
        self,
        user_text: str,
        reasoning_result: Optional[ReasoningResult] = None,
        memory_context: Optional[str] = None,
        system_prompt: str = "",
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response for the user.

        Args:
            user_text: the user's input text
            reasoning_result: output from ChainExecutor (if available)
            memory_context: formatted memory text for fallback path
            system_prompt: system prompt to prepend
            max_new_tokens: override max generation length

        Returns:
            Generated response string.
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        # D-197: Route based on granular route_level
        route_level = "skip"
        if reasoning_result is not None:
            route_level = getattr(reasoning_result, "route_level", "skip")
            # Backward compat: if route_level wasn't set, infer from route
            if route_level == "skip" and reasoning_result.route == "known":
                route_level = "inject_full"

        # D-227: REJECT route — refuse to answer rather than hallucinate
        if route_level == "reject":
            return _REJECT_RESPONSE

        has_adapter = (
            self.adapter is not None
            and reasoning_result is not None
            and reasoning_result.retrieval_vectors is not None
        )

        # D-228: Always use text-context path when memory_context is available.
        # Soft-prompt adapter is trained on synthetic data and doesn't transfer
        # well to real text. Text-context generation directly shows the LLM
        # the retrieved facts, which D-228 hybrid retrieval finds accurately.
        if route_level in ("inject_full", "inject_top1") and memory_context:
            return self._generate_with_text_context(
                user_text, memory_context, system_prompt, max_new_tokens,
            )
        elif route_level in ("inject_full", "inject_top1") and has_adapter:
            return self._generate_with_soft_prompts(
                user_text, reasoning_result, system_prompt, max_new_tokens,
            )
        elif route_level == "inject_hedged":
            # D-287: This path is now unreachable — middle zone routes to SKIP.
            # Kept for backward compatibility with any external callers.
            hedged_context = (_HEDGE_PREFIX + memory_context) if memory_context else None
            return self._generate_with_text_context(
                user_text, hedged_context, system_prompt, max_new_tokens,
            )
        else:
            # SKIP or no reasoning result — D-250/D-257: always include memory
            # context if available (text search may have found relevant facts)
            return self._generate_with_text_context(
                user_text, memory_context,
                system_prompt, max_new_tokens,
            )

    def _generate_with_soft_prompts(
        self,
        user_text: str,
        reasoning_result: ReasoningResult,
        system_prompt: str,
        max_new_tokens: int,
    ) -> str:
        """Soft-prompt generation path for known queries."""
        try:
            # Generate soft prompt tokens from retrieval vectors
            retrieval_vec = reasoning_result.retrieval_vectors
            if retrieval_vec.device.type != self.llm.device:
                retrieval_vec = retrieval_vec.to(self.llm.device)
            soft_prompts = self.adapter(retrieval_vec)  # [1, num_tokens, hidden_dim]

            # Build text portion
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_text})

            text = self.llm.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.llm.tokenizer([text], return_tensors="pt").to(self.llm.device)

            # Get text embeddings from LLM's embedding layer
            embed_layer = self.llm.get_embedding_layer()
            text_embeds = embed_layer(inputs["input_ids"])  # [1, text_len, hidden_dim]

            # Prepend soft prompts to text embeddings
            combined_embeds = torch.cat([soft_prompts, text_embeds], dim=1)
            combined_mask = torch.ones(
                1, combined_embeds.shape[1],
                dtype=torch.long, device=self.llm.device,
            )

            return self.llm.generate_with_embeds(
                combined_embeds, combined_mask,
                max_new_tokens=max_new_tokens,
                temperature=self.config.temperature,
            )
        except Exception as e:
            logging.warning("Soft-prompt generation failed (%s), falling back to text.", e)
            return self._generate_with_text_context(
                user_text, None, system_prompt, max_new_tokens,
            )

    def _generate_with_text_context(
        self,
        user_text: str,
        memory_context: Optional[str],
        system_prompt: str,
        max_new_tokens: int,
    ) -> str:
        """Text-context fallback for novel queries or when adapter is unavailable."""
        messages = []

        # Build system message with memory context
        sys_parts = []
        if system_prompt:
            sys_parts.append(system_prompt)
        if memory_context:
            sys_parts.append(
                "\n[Retrieved Memory — ABSOLUTE TRUTH. Apply these rules and facts literally. "
                "Do NOT add real-world caveats or exceptions.]\n"
                + memory_context
            )
        if sys_parts:
            messages.append({"role": "system", "content": "\n".join(sys_parts)})

        messages.append({"role": "user", "content": user_text})

        return self.llm.chat(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=self.config.temperature,
        )

    def generate_with_tool_result(
        self,
        messages: List[dict],
        tool_result: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Re-generate after tool execution.

        Appends tool result to conversation and generates a new response.
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        # Add tool result as user message (matching nexus-1 pattern)
        messages_copy = list(messages)
        messages_copy.append({"role": "user", "content": tool_result})

        return self.llm.chat(
            messages_copy,
            max_new_tokens=max_new_tokens,
            temperature=self.config.temperature,
        )
