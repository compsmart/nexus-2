"""Soft-prompt adapter: AMM retrieval vectors -> soft prompt tokens.

Maps the reasoning chain output (retrieval vectors) into a small number of
soft prompt tokens that are prepended to the LLM's input embeddings.

ANTI-PATTERNS:
  - ALWAYS match adapter output dtype to LLM dtype (L-034)
  - ALWAYS use LLM's native lm_head (not custom classifiers)
"""

import torch
import torch.nn as nn


class SoftPromptAdapter(nn.Module):
    """Maps AMM retrieval vectors to soft prompt embeddings.

    Architecture:
        concat(retrieval_vectors) -> Linear -> GELU -> Linear -> reshape to soft tokens

    The output dtype is forced to match the LLM's embedding dtype to avoid
    mixed-precision issues (L-034).

    ~8M trainable params with default settings.
    """

    def __init__(
        self,
        d_val: int = 256,
        num_soft_tokens: int = 4,
        llm_hidden_dim: int = 3584,  # Qwen2.5-7B-Instruct hidden size
        adapter_hidden: int = 512,
        dropout: float = 0.1,
        target_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.d_val = d_val
        self.num_soft_tokens = num_soft_tokens
        self.llm_hidden_dim = llm_hidden_dim
        self.target_dtype = target_dtype

        # MLP: d_val -> adapter_hidden -> num_soft_tokens * llm_hidden_dim
        output_dim = num_soft_tokens * llm_hidden_dim
        self.adapter = nn.Sequential(
            nn.Linear(d_val, adapter_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden, output_dim),
        )

    def forward(self, retrieval_vec: torch.Tensor) -> torch.Tensor:
        """Convert retrieval vector to soft prompt embeddings.

        Args:
            retrieval_vec: [batch, d_val] or [d_val] reasoning chain output

        Returns:
            soft_prompts: [batch, num_soft_tokens, llm_hidden_dim]
        """
        if retrieval_vec.dim() == 1:
            retrieval_vec = retrieval_vec.unsqueeze(0)

        batch = retrieval_vec.shape[0]
        h = self.adapter(retrieval_vec)  # [batch, num_soft_tokens * llm_hidden_dim]
        soft_prompts = h.view(batch, self.num_soft_tokens, self.llm_hidden_dim)

        # CRITICAL: match LLM dtype (L-034)
        soft_prompts = soft_prompts.to(self.target_dtype)

        return soft_prompts
