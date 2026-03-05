"""Knowledge tools: ingest_document, search_memory, forget, remember, correct."""

import os
import re as _re

from ..tool_registry import ToolResult


class IngestDocumentTool:
    name = "ingest_document"
    description = (
        "Read a document and store its contents as searchable memory facts. "
        "Usage: [TOOL_CALL: ingest_document | path/to/document.txt]"
    )

    def __init__(self, memory):
        self.memory = memory

    def run(self, path: str) -> ToolResult:
        if self.memory is None:
            return ToolResult(self.name, "", success=False, error="Memory not available.")
        try:
            path = path.strip()
            if not os.path.isfile(path):
                return ToolResult(self.name, "", success=False,
                                  error=f"File not found: {path}")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            # Split into paragraphs/chunks
            chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
            if not chunks:
                return ToolResult(self.name, "Document is empty.")

            stored = 0
            for chunk in chunks[:100]:  # limit to 100 chunks
                if len(chunk) < 10:
                    continue
                ok = self.memory.store(
                    chunk[:500],
                    mem_type="document",
                    subject=os.path.basename(path),
                )
                if ok:
                    stored += 1

            return ToolResult(
                self.name,
                f"Ingested {stored} chunks from {os.path.basename(path)}",
            )
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


class SearchMemoryTool:
    name = "search_memory"
    description = (
        "Search the agent's memory for relevant information. "
        "Usage: [TOOL_CALL: search_memory | query text]"
    )

    def __init__(self, memory):
        self.memory = memory

    def run(self, query: str) -> ToolResult:
        if self.memory is None:
            return ToolResult(self.name, "", success=False, error="Memory not available.")
        try:
            results = self.memory.retrieve(query.strip(), top_k=5)
            if not results:
                return ToolResult(self.name, "No relevant memories found.")
            lines = []
            for text, score, entry in results:
                lines.append(f"  [{entry.mem_type}] (score={score:.3f}) {text}")
            return ToolResult(self.name, "\n".join(lines))
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


class ForgetTool:
    name = "forget"
    description = (
        "Delete memories matching a text pattern. "
        "Usage: [TOOL_CALL: forget | pattern to match]"
    )

    def __init__(self, memory):
        self.memory = memory

    def run(self, pattern: str) -> ToolResult:
        if self.memory is None:
            return ToolResult(self.name, "", success=False, error="Memory not available.")
        try:
            pattern = pattern.strip()
            count = self.memory.delete_matching(pattern)
            # Store a permanent note so this topic doesn't resurface
            self.memory.store(
                f"User wants to forget about: {pattern[:100]}",
                mem_type="correction",
                subject=pattern[:60],
            )
            return ToolResult(self.name, f"Forgotten: removed {count} matching memories.")
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


class RememberTool:
    name = "remember"
    description = (
        "Store one or more facts in persistent memory. "
        "Separate multiple facts with semicolons. "
        "Usage: [TOOL_CALL: remember | fact1; fact2; fact3]"
    )

    def __init__(self, memory):
        self.memory = memory

    def run(self, arg: str) -> ToolResult:
        if self.memory is None:
            return ToolResult(self.name, "", success=False, error="Memory not available.")
        facts = [f.strip() for f in _re.split(r"[;\n]+", arg.strip())
                 if f.strip() and len(f.strip()) >= 5]
        if not facts:
            return ToolResult(self.name, "", success=False, error="No facts provided.")
        stored = 0
        for fact in facts:
            ok = self.memory.store(fact, mem_type="fact", subject="personal_fact")
            if ok:
                stored += 1
        if stored == 0:
            return ToolResult(self.name, "Already known — no new facts stored.")
        label = "facts" if stored != 1 else "fact"
        return ToolResult(self.name, f"Remembered {stored} {label}.")


class CorrectTool:
    name = "correct"
    description = (
        "Replace a wrong memory with a correction that never decays. "
        "Usage: [TOOL_CALL: correct | wrong info | correct info]"
    )

    def __init__(self, memory):
        self.memory = memory

    def run(self, arg: str) -> ToolResult:
        if self.memory is None:
            return ToolResult(self.name, "", success=False, error="Memory not available.")
        parts = [p.strip() for p in arg.split("|", 1)]
        if len(parts) != 2 or not parts[1]:
            return ToolResult(self.name, "", success=False,
                              error="Format: [TOOL_CALL: correct | wrong | correct]")
        wrong, correction = parts
        deleted = self.memory.delete_matching(wrong) if wrong else 0
        self.memory.store(
            correction,
            mem_type="correction",
            subject="correction",
        )
        return ToolResult(
            self.name,
            f"Corrected: removed {deleted} old entries, stored correction.",
        )
