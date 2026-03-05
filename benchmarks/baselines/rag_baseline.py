"""RAG baseline: ChromaDB vector store + LLM retrieval-augmented generation."""

import logging
from typing import Optional


class RagBaseline:
    """ChromaDB vector retrieval + Qwen2.5-7B generation."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._collection = None
        self._llm = None
        self._facts: list = []
        self._doc_id = 0

    def _ensure_loaded(self):
        if self._collection is None:
            try:
                import chromadb
                self._client = chromadb.Client()
                self._collection = self._client.create_collection(
                    name="rag_bench",
                    metadata={"hnsw:space": "cosine"},
                )
            except ImportError:
                logging.warning("chromadb not installed, using simple list fallback")
                self._collection = "fallback"

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
        """Reset the vector store."""
        self._facts = []
        self._doc_id = 0
        if self._collection is not None and self._collection != "fallback":
            try:
                import chromadb
                self._client = chromadb.Client()
                self._collection = self._client.create_collection(
                    name=f"rag_bench_{self._doc_id}",
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception:
                self._collection = "fallback"

    def teach(self, text: str):
        """Add a fact to the vector store."""
        self._ensure_loaded()
        self._facts.append(text)
        self._doc_id += 1

        if self._collection != "fallback":
            try:
                self._collection.add(
                    documents=[text],
                    ids=[f"doc_{self._doc_id}"],
                )
            except Exception:
                pass

    def query(self, text: str) -> str:
        """Retrieve context and generate response."""
        self._ensure_loaded()

        # Retrieve relevant facts
        context_facts = []
        if self._collection != "fallback":
            try:
                results = self._collection.query(query_texts=[text], n_results=5)
                if results and results["documents"]:
                    context_facts = results["documents"][0]
            except Exception:
                context_facts = self._facts[-10:]  # fallback
        else:
            # Simple substring match fallback
            query_lower = text.lower()
            for fact in self._facts:
                if any(w in fact.lower() for w in query_lower.split()):
                    context_facts.append(fact)
            context_facts = context_facts[:10]

        # Generate with context
        if self._llm is not None:
            context = "\n".join(f"- {f}" for f in context_facts)
            messages = [
                {"role": "system", "content": f"Use these facts to answer:\n{context}"},
                {"role": "user", "content": text},
            ]
            return self._llm.chat(messages, max_new_tokens=100)

        # No LLM fallback
        return context_facts[0] if context_facts else ""
