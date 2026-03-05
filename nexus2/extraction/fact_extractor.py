"""Fact extraction via Gemini Flash — cheap, fast, structured output.

Takes raw web/article content and extracts discrete factual statements.
Each fact becomes its own AMM entry for precise semantic retrieval.
Falls back to raw paragraph chunking if Gemini is unavailable.
"""

import json
import logging
import re
from typing import List, Optional

import requests

_EXTRACT_PROMPT = (
    "Extract every discrete factual statement from the following content.\n"
    "Each fact should be a single, self-contained sentence that could be understood without context.\n"
    "Include: names, dates, numbers, addresses, phone numbers, emails, URLs, relationships, descriptions.\n"
    "Exclude: navigation text, cookie notices, boilerplate, opinions, marketing fluff.\n"
    "If the content has no extractable facts, return an empty list.\n"
    "\nContent:\n{content}"
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of discrete factual statements extracted from the content",
        }
    },
    "required": ["facts"],
}


class FactExtractor:
    """Extracts structured facts from raw text using Gemini Flash."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-2.0-flash",
        timeout: int = 60,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        )

    def extract(self, content: str, source: str = "") -> List[str]:
        """Extract facts from content. Returns list of fact strings.

        Falls back to paragraph chunking if Gemini call fails.
        """
        if not self.api_key:
            logging.warning("[fact_extractor] No API key — falling back to raw chunks")
            return self._fallback_chunk(content)

        try:
            facts = self._call_gemini(content, source)
            if facts:
                logging.info(
                    "[fact_extractor] Extracted %d facts from %d chars (%s)",
                    len(facts), len(content), source[:60],
                )
                return facts
            logging.warning("[fact_extractor] Gemini returned empty — falling back")
            return self._fallback_chunk(content)
        except Exception as e:
            logging.warning("[fact_extractor] Gemini call failed (%s) — falling back", e)
            return self._fallback_chunk(content)

    def _call_gemini(self, content: str, source: str) -> Optional[List[str]]:
        """Call Gemini API for structured fact extraction."""
        prompt = _EXTRACT_PROMPT.format(content=content)

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": _RESPONSE_SCHEMA,
                "temperature": 0.1,
            },
        }

        resp = requests.post(
            self._endpoint,
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return None

        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        if not text:
            return None

        parsed = json.loads(text)
        facts = parsed.get("facts", [])

        return [f.strip() for f in facts if isinstance(f, str) and len(f.strip()) >= 10]

    @staticmethod
    def _fallback_chunk(content: str) -> List[str]:
        """Fallback: split into paragraphs when Gemini is unavailable."""
        raw = [c.strip() for c in content.split("\n\n") if c.strip()]
        chunks = []
        for chunk in raw:
            if len(chunk) > 2000:
                sub = [s.strip() for s in chunk.split("\n") if s.strip()]
                chunks.extend(sub)
            else:
                chunks.append(chunk)
        return [c for c in chunks if len(c) >= 20]
