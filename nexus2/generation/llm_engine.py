"""Frozen LLM wrapper with inputs_embeds generation path.

Supports Qwen2.5-7B-Instruct (default) and other HuggingFace causal LMs.
Adapted from nexus-1/llm.py with added support for soft-prompt injection
via inputs_embeds. The LLM backbone remains frozen (no gradient computation).

ANTI-PATTERNS:
  - ALWAYS use LLM's native lm_head (not custom classifiers)
"""

import logging
import threading
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_GREEDY_TEMP_THRESHOLD = 0.15
_MODEL_MAX_LENGTH_SENTINEL = 10_000_000


class LLMEngine:
    """Frozen LLM wrapper with soft-prompt embedding injection.

    Process-wide cache ensures only one copy of the model is loaded.
    """

    _CACHE_LOCK = threading.Lock()
    _MODEL_CACHE: Dict[Tuple[str, str, bool], Tuple] = {}

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        use_4bit: bool = True,
        repetition_penalty: float = 1.1,
        context_fallback_tokens: int = 4096,
        shared_cache: bool = True,
    ):
        self.model_name = model_name
        self.repetition_penalty = max(1.0, float(repetition_penalty))
        self._context_fallback = int(context_fallback_tokens)

        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        cache_key = (model_name, device, bool(use_4bit))
        if shared_cache:
            with self._CACHE_LOCK:
                cached = self._MODEL_CACHE.get(cache_key)
            if cached is not None:
                self.tokenizer, self.model, self._max_ctx, self.device = cached
                logging.info("Reusing cached LLM %s on %s", model_name, self.device)
                return

        tokenizer, model, loaded_device = self._load_model(model_name, device, use_4bit)
        self.tokenizer = tokenizer
        self.model = model
        self.device = loaded_device
        self._max_ctx = self._resolve_context_limit(tokenizer, model, self._context_fallback)

        logging.info(
            "LLM loaded: model=%s device=%s context=%s 4-bit=%s",
            model_name, self.device, self._max_ctx, use_4bit,
        )

        if shared_cache:
            with self._CACHE_LOCK:
                self._MODEL_CACHE[cache_key] = (
                    self.tokenizer, self.model, self._max_ctx, self.device,
                )

    def _load_model(self, model_name, device, use_4bit):
        logging.info("Loading LLM %s on %s (4-bit=%s)...", model_name, device, use_4bit)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if use_4bit and device == "cuda":
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
                model.eval()
                return tokenizer, model, "cuda"
            except Exception as e:
                logging.warning("4-bit load failed (%s). Falling back.", e)

        try:
            dtype = torch.float16 if device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
            )
            if device == "cpu":
                model.to(device)
            model.eval()
            return tokenizer, model, device
        except Exception as e:
            if device != "cuda":
                raise
            logging.warning("CUDA load failed (%s). Falling back to CPU.", e)

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map=None,
        )
        model.to("cpu")
        model.eval()
        return tokenizer, model, "cpu"

    def _resolve_context_limit(self, tokenizer, model, fallback):
        candidates = []
        tok_max = getattr(tokenizer, "model_max_length", None)
        if isinstance(tok_max, int) and 0 < tok_max < _MODEL_MAX_LENGTH_SENTINEL:
            candidates.append(tok_max)
        cfg = getattr(model, "config", None)
        if cfg:
            for attr in ("max_position_embeddings", "n_positions", "seq_length"):
                v = getattr(cfg, attr, None)
                if isinstance(v, int) and v > 0:
                    candidates.append(v)
        return max(candidates) if candidates else max(512, fallback)

    def get_embedding_layer(self):
        """Return the model's input embedding layer for soft-prompt injection."""
        return self.model.get_input_embeddings()

    def get_embedding_dtype(self) -> torch.dtype:
        """Return dtype of the model's embedding layer."""
        return self.model.get_input_embeddings().weight.dtype

    def get_embedding_dim(self) -> int:
        """Return hidden dimension of the model."""
        return self.model.config.hidden_size

    def chat(
        self,
        messages: list,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
    ) -> str:
        """Standard chat generation (no soft prompts)."""
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            kwargs = self._gen_kwargs(inputs["input_ids"], max_new_tokens, temperature)
            prompt_len = kwargs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = self.model.generate(**kwargs)
            return self.tokenizer.decode(
                outputs[0][prompt_len:], skip_special_tokens=True,
            ).strip()
        except Exception as e:
            logging.error("LLM chat error: %s", e)
            return ""

    def generate_with_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
    ) -> str:
        """Generate using pre-computed input embeddings (for soft-prompt injection).

        Args:
            inputs_embeds: [1, seq_len, hidden_dim] input embeddings
            attention_mask: [1, seq_len] attention mask
            max_new_tokens: max tokens to generate
            temperature: sampling temperature

        Returns:
            Generated text string.
        """
        try:
            do_sample = temperature > _GREEDY_TEMP_THRESHOLD
            kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": self.repetition_penalty,
            }
            if do_sample:
                kwargs["temperature"] = temperature

            with torch.no_grad():
                outputs = self.model.generate(**kwargs)

            # The model generates token IDs; decode them all
            # (inputs_embeds path doesn't return the prompt tokens)
            return self.tokenizer.decode(
                outputs[0], skip_special_tokens=True,
            ).strip()
        except Exception as e:
            logging.error("LLM embeds generate error: %s", e)
            return ""

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
    ) -> str:
        """Raw prompt generation."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            kwargs = self._gen_kwargs(inputs["input_ids"], max_new_tokens, temperature)
            prompt_len = kwargs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = self.model.generate(**kwargs)
            return self.tokenizer.decode(
                outputs[0][prompt_len:], skip_special_tokens=True,
            ).strip()
        except Exception as e:
            logging.error("LLM generate error: %s", e)
            return ""

    def _gen_kwargs(self, input_ids, max_new_tokens, temperature):
        available = self._max_ctx - max_new_tokens
        if available <= 0:
            available = max(256, self._max_ctx // 2)
        if input_ids.shape[1] > available:
            logging.warning("Prompt truncated: %s -> %s tokens.", input_ids.shape[1], available)
            input_ids = input_ids[:, -available:]

        do_sample = temperature > _GREEDY_TEMP_THRESHOLD
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": self.repetition_penalty,
        }
        if do_sample:
            kwargs["temperature"] = temperature
        return kwargs
