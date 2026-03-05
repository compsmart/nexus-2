"""File-backed YAML skill catalog.

Simplified from nexus-1/skills_store.py. Skills are markdown files with
YAML frontmatter, organized into drafts/ and published/ directories.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional


def _slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text[:60]


class SkillStore:
    """File-backed skill catalog with draft/published lifecycle."""

    def __init__(self, skills_dir: str = "data/skills", index_path: str = "data/skills/index.json"):
        self.skills_dir = Path(skills_dir)
        self.drafts_dir = self.skills_dir / "drafts"
        self.published_dir = self.skills_dir / "published"
        self.index_path = Path(index_path)

        # Ensure directories exist
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        self.published_dir.mkdir(parents=True, exist_ok=True)

        # Load or create index
        self._index: Dict = {}
        self._load_index()

    def _load_index(self):
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._index = {}

    def _save_index(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, ensure_ascii=False)

    def create_draft(self, title: str, purpose: str = "", steps: str = "") -> str:
        """Create a draft skill markdown file.

        Returns the skill_id.
        """
        skill_id = _slugify(title) + "-v1"
        path = self.drafts_dir / f"{skill_id}.md"

        content = f"""---
skill_id: {skill_id}
title: {title}
status: draft
version: "1.0.0"
created_at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
---

## Purpose
{purpose}

## Steps
{steps}
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        self._index[skill_id] = {
            "title": title,
            "status": "draft",
            "path": str(path),
        }
        self._save_index()
        return skill_id

    def publish(self, skill_id: str) -> bool:
        """Move a draft skill to published."""
        info = self._index.get(skill_id)
        if not info or info["status"] != "draft":
            return False

        draft_path = Path(info["path"])
        if not draft_path.exists():
            return False

        pub_path = self.published_dir / draft_path.name
        content = draft_path.read_text(encoding="utf-8")
        content = content.replace("status: draft", "status: published")

        with open(pub_path, "w", encoding="utf-8") as f:
            f.write(content)

        draft_path.unlink()
        self._index[skill_id]["status"] = "published"
        self._index[skill_id]["path"] = str(pub_path)
        self._save_index()
        return True

    def get_skill(self, skill_id: str) -> Optional[str]:
        """Read skill content by ID."""
        info = self._index.get(skill_id)
        if not info:
            return None
        path = Path(info["path"])
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def list_skills(self) -> List[Dict]:
        """List all skills with metadata."""
        result = []
        for skill_id, info in sorted(self._index.items()):
            result.append({
                "skill_id": skill_id,
                "title": info.get("title", ""),
                "status": info.get("status", "unknown"),
            })
        return result
