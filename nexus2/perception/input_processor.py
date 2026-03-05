"""Perception layer: entity extraction, intent classification, fact extraction.

Adapted from nexus-1/agent.py extraction patterns (lines 459-626).
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

from ..config import NexusConfig


@dataclass
class PerceptionResult:
    """Structured output from input processing."""
    raw_text: str
    intent: str = "query"               # query | teach | correct | command
    entities: List[str] = field(default_factory=list)
    user_name: Optional[str] = None
    personal_facts: List[str] = field(default_factory=list)
    is_correction: bool = False
    is_short_followup: bool = False
    correction_topic: Optional[str] = None  # what user is correcting/rejecting


# Intent keywords
_TEACH_CUES = {"my name is", "i am", "i'm", "my favorite", "my dog", "i live",
               "i work", "i have", "remember that", "i like", "i love"}
_CORRECT_CUES = {"actually", "correction", "i meant", "that's wrong",
                 "not anymore", "no longer", "wrong", "incorrect",
                 "stop asking", "don't ask me about", "don't ask about",
                 "dont ask me about", "dont ask about",
                 "i don't like", "i dont like",
                 "i don't want", "i dont want",
                 "stop talking about", "that's not right", "thats not right",
                 "that's not true", "thats not true", "please stop",
                 "enough about", "i never said"}
_COMMAND_CUES = {"search for", "look up", "calculate", "what time",
                 "convert", "execute", "run", "read file", "write file"}

# Personal fact patterns
_PERSONAL_FACT_PATTERNS = [
    re.compile(r"\bmy\s+(\w[\w\s]{4,58})\s+is\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"\bi\s+live\s+(?:in|at)\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"\bi\s+am\s+(\d{1,3})\s+years?\s+old\b", re.IGNORECASE),
    re.compile(r"\bi\s+have\s+a?\s*(\w+)\s+(?:called|named)\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"\bi\s+(?:like|love|enjoy|prefer)\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"\bi\s+work\s+(?:at|for|in)\s+(.{2,60})", re.IGNORECASE),
]

# Correction topic extraction: captures WHAT the user is correcting/rejecting
_CORRECTION_TOPIC_PATTERNS = [
    re.compile(r"stop (?:asking|talking) (?:me )?about\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"don'?t (?:ask|talk) (?:me )?about\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"enough about\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"i (?:don'?t|never) (?:like|want|enjoy|eat|care about)\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"i'?m not (?:into|interested in|a fan of)\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"(?:that'?s|it'?s|this is) (?:wrong|incorrect|not (?:right|true))(?:[.,!]?\s*(?:it'?s|i)\s+(.{2,60}))?", re.IGNORECASE),
    re.compile(r"(?:actually|no),?\s+(?:i|my)\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"i never said\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"not anymore[.,!]?\s*(?:i\s+)?(.{2,60})?", re.IGNORECASE),
]

# Entity pattern: capitalized words or quoted strings
_ENTITY_RE = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
_QUOTED_RE = re.compile(r'"([^"]{2,50})"')

# Stop words for name extraction
_NAME_STOP_RE = re.compile(r'\b(?:and|but|because|that|who|which|where|when)\b', re.IGNORECASE)


class InputProcessor:
    """Processes user input into structured perception results."""

    def __init__(self, config: NexusConfig):
        self.config = config
        self._name_patterns = [re.compile(p, re.IGNORECASE) for p in config.name_patterns]
        self._blocked_names = config.blocked_names
        self._negation_cues = config.negation_cues

    def process(self, text: str, prev_text: Optional[str] = None) -> PerceptionResult:
        """Process user input into structured result.

        Args:
            text: current user input
            prev_text: previous turn text (for followup detection)

        Returns:
            PerceptionResult with extracted information.
        """
        result = PerceptionResult(raw_text=text)

        # Intent classification
        result.intent = self._classify_intent(text)
        result.is_correction = result.intent == "correct"

        # Correction topic extraction
        if result.is_correction:
            result.correction_topic = self._extract_correction_topic(text)

        # Entity extraction
        result.entities = self._extract_entities(text)

        # User name detection
        result.user_name = self._extract_user_name(text)

        # Personal fact extraction (with negation guard)
        if not self._has_negation(text):
            result.personal_facts = self._extract_personal_facts(text)

        # Short followup detection
        if prev_text is not None:
            result.is_short_followup = self._is_short_followup(text)

        return result

    def _classify_intent(self, text: str) -> str:
        """Classify user intent."""
        text_lower = text.lower()

        # Check correction first (highest priority)
        for cue in _CORRECT_CUES:
            if cue in text_lower:
                return "correct"

        # Check teach
        for cue in _TEACH_CUES:
            if cue in text_lower:
                return "teach"

        # Check command
        for cue in _COMMAND_CUES:
            if cue in text_lower:
                return "command"

        return "query"

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        entities = set()

        # Capitalized names
        for match in _ENTITY_RE.finditer(text):
            ent = match.group(1).strip()
            if len(ent) >= 2:
                entities.add(ent)

        # Quoted strings
        for match in _QUOTED_RE.finditer(text):
            entities.add(match.group(1).strip())

        return list(entities)

    def _extract_user_name(self, text: str) -> Optional[str]:
        """Extract user's name from self-identification patterns."""
        # Skip name extraction when text contains negation to prevent
        # "don't call me bob" -> extracting "bob" as user name
        if self._has_negation(text):
            return None
        for pattern in self._name_patterns:
            match = pattern.search(text)
            if match:
                name = match.group(1).strip()
                # Truncate at stop words
                stop_match = _NAME_STOP_RE.search(name)
                if stop_match:
                    name = name[:stop_match.start()].strip()
                # Validate
                name_lower = name.lower()
                if name_lower in self._blocked_names:
                    continue
                if len(name) < 2 or len(name) > 40:
                    continue
                return name
        return None

    def _extract_personal_facts(self, text: str) -> List[str]:
        """Extract personal facts from user input."""
        facts = []
        for pattern in _PERSONAL_FACT_PATTERNS:
            for match in pattern.finditer(text):
                fact = match.group(0).strip()
                if 6 <= len(fact) <= 200:
                    facts.append(fact)
        return facts

    def _has_negation(self, text: str) -> bool:
        """Check if text contains negation cues."""
        text_lower = text.lower()
        return any(cue in text_lower for cue in self._negation_cues)

    def _extract_correction_topic(self, text: str) -> Optional[str]:
        """Extract the topic being corrected/rejected from user input."""
        for pattern in _CORRECTION_TOPIC_PATTERNS:
            match = pattern.search(text)
            if match:
                topic = match.group(1)
                if topic:
                    # Clean trailing punctuation
                    topic = topic.strip().rstrip(".,!?;:")
                    if len(topic) >= 2:
                        return topic
        # Fallback: use the full text minus the correction cue
        return None

    def _is_short_followup(self, text: str) -> bool:
        """Detect short followup messages (pronouns, < 7 words)."""
        words = text.split()
        if len(words) >= 7:
            return False
        followup_cues = {"it", "that", "he", "she", "they", "this",
                         "what is it", "what about", "tell me more",
                         "how about", "and"}
        text_lower = text.lower()
        return any(cue in text_lower for cue in followup_cues)
