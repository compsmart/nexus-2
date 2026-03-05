"""Skill tools: create_skill, learn_skill, list_skills, show_skill, publish_skill."""

import logging

from ..tool_registry import ToolResult


class CreateSkillTool:
    name = "create_skill"
    description = (
        "Create a draft skill from structured input. "
        "Usage: [TOOL_CALL: create_skill | title: ..., purpose: ..., steps: ...]"
    )

    def __init__(self, memory, skill_store):
        self.memory = memory
        self.skill_store = skill_store

    def run(self, arg: str) -> ToolResult:
        if self.skill_store is None:
            return ToolResult(self.name, "", success=False, error="Skill store not available.")
        try:
            # Parse simple key: value pairs
            fields = {}
            for line in arg.split(","):
                if ":" in line:
                    key, val = line.split(":", 1)
                    fields[key.strip().lower()] = val.strip()

            title = fields.get("title", "Untitled Skill")
            purpose = fields.get("purpose", "")
            steps = fields.get("steps", "")

            skill_id = self.skill_store.create_draft(
                title=title, purpose=purpose, steps=steps,
            )

            # Store reference in memory
            if self.memory is not None:
                self.memory.store(
                    f"Skill: {title} ({skill_id})",
                    mem_type="skill_ref",
                    subject=title,
                )

            return ToolResult(self.name, f"Draft skill created: {skill_id}")
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


class LearnSkillTool:
    name = "learn_skill"
    description = (
        "Search the web for a topic and save a draft skill. "
        "Usage: [TOOL_CALL: learn_skill | topic to learn about]"
    )

    def __init__(self, memory, skill_store):
        self.memory = memory
        self.skill_store = skill_store

    def run(self, topic: str) -> ToolResult:
        if self.skill_store is None:
            return ToolResult(self.name, "", success=False, error="Skill store not available.")
        try:
            # Web search for the topic
            from .web import WebSearchTool
            search = WebSearchTool()
            search_result = search.run(topic.strip())

            if not search_result.success:
                return ToolResult(self.name, "", success=False,
                                  error=f"Web search failed: {search_result.error}")

            # Create draft skill from search results
            skill_id = self.skill_store.create_draft(
                title=f"Learned: {topic.strip()[:50]}",
                purpose=f"Knowledge about: {topic}",
                steps=search_result.output[:2000],
            )

            if self.memory is not None:
                self.memory.store(
                    f"Learned skill about {topic}: {skill_id}",
                    mem_type="skill_ref",
                    subject=topic,
                )

            return ToolResult(self.name, f"Skill learned and saved: {skill_id}")
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


class ListSkillsTool:
    name = "list_skills"
    description = (
        "List all available skills (drafts and published). "
        "Usage: [TOOL_CALL: list_skills | any]"
    )

    def __init__(self, skill_store):
        self.skill_store = skill_store

    def run(self, _arg: str) -> ToolResult:
        if self.skill_store is None:
            return ToolResult(self.name, "", success=False, error="Skill store not available.")
        try:
            skills = self.skill_store.list_skills()
            if not skills:
                return ToolResult(self.name, "No skills found.")
            lines = []
            for s in skills:
                lines.append(f"  [{s['status']}] {s['title']} ({s['skill_id']})")
            return ToolResult(self.name, "\n".join(lines))
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


class ShowSkillTool:
    name = "show_skill"
    description = (
        "Show the content of a specific skill. "
        "Usage: [TOOL_CALL: show_skill | skill-id]"
    )

    def __init__(self, skill_store):
        self.skill_store = skill_store

    def run(self, skill_id: str) -> ToolResult:
        if self.skill_store is None:
            return ToolResult(self.name, "", success=False, error="Skill store not available.")
        try:
            content = self.skill_store.get_skill(skill_id.strip())
            if content is None:
                return ToolResult(self.name, f"Skill '{skill_id}' not found.")
            return ToolResult(self.name, content)
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


class PublishSkillTool:
    name = "publish_skill"
    description = (
        "Promote a draft skill to published status. "
        "Usage: [TOOL_CALL: publish_skill | skill-id]"
    )

    def __init__(self, memory, skill_store):
        self.memory = memory
        self.skill_store = skill_store

    def run(self, skill_id: str) -> ToolResult:
        if self.skill_store is None:
            return ToolResult(self.name, "", success=False, error="Skill store not available.")
        try:
            ok = self.skill_store.publish(skill_id.strip())
            if not ok:
                return ToolResult(self.name, "", success=False,
                                  error=f"Failed to publish '{skill_id}'.")
            return ToolResult(self.name, f"Skill '{skill_id}' published successfully.")
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))
