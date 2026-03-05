"""Tool registry and dispatch system.

Adapted from nexus-1/tools.py. Tools are simple classes with:
  - name: str
  - description: str
  - run(arg: str) -> ToolResult

Dispatch: parse [TOOL_CALL: name | arg] from LLM output and execute.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, Optional

from ..config import NexusConfig


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    output: str
    success: bool = True
    error: Optional[str] = None

    def to_context(self) -> str:
        if not self.success:
            return f"[{self.tool_name} ERROR] {self.error}"
        return self.output


class ToolRegistry:
    """Registry and dispatcher for agent tools."""

    def __init__(self, config: NexusConfig, memory=None, skill_store=None):
        self.config = config
        self._tools: Dict[str, object] = {}
        self._pattern = re.compile(config.tool_call_pattern, re.DOTALL)
        self._build_registry(memory, skill_store)

    def _build_registry(self, memory, skill_store):
        """Import and register all tools."""
        from .tools.web import WebSearchTool, WebFetchTool, WikipediaTool
        from .tools.compute import CalculatorTool, PythonExecTool
        from .tools.datetime_tools import DatetimeNowTool, TimerDeltaTool, UnitConvertTool
        from .tools.filesystem import FileReadTool, FileWriteTool, FileListTool, FileSearchTool
        from .tools.knowledge import IngestDocumentTool, SearchMemoryTool, ForgetTool, RememberTool, CorrectTool
        from .tools.skills import (
            CreateSkillTool, LearnSkillTool, ListSkillsTool,
            ShowSkillTool, PublishSkillTool,
        )

        tools = [
            WebSearchTool(),
            WebFetchTool(),
            WikipediaTool(),
            CalculatorTool(),
            PythonExecTool(),
            DatetimeNowTool(),
            TimerDeltaTool(),
            UnitConvertTool(),
            FileReadTool(),
            FileWriteTool(),
            FileListTool(),
            FileSearchTool(),
            IngestDocumentTool(memory),
            SearchMemoryTool(memory),
            ForgetTool(memory),
            RememberTool(memory),
            CorrectTool(memory),
            CreateSkillTool(memory, skill_store),
            LearnSkillTool(memory, skill_store),
            ListSkillsTool(skill_store),
            ShowSkillTool(skill_store),
            PublishSkillTool(memory, skill_store),
        ]

        for tool in tools:
            self._tools[tool.name] = tool

    def parse_tool_call(self, text: str) -> Optional[tuple]:
        """Parse a tool call from LLM output.

        Returns (tool_name, argument) or None if no tool call found.
        """
        match = self._pattern.search(text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return None

    def execute(self, tool_name: str, arg: str) -> ToolResult:
        """Execute a tool by name.

        Returns ToolResult.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                output="",
                success=False,
                error=f"Unknown tool: {tool_name}",
            )
        try:
            return tool.run(arg)
        except Exception as e:
            logging.error("Tool %s error: %s", tool_name, e)
            return ToolResult(
                tool_name=tool_name,
                output="",
                success=False,
                error=str(e),
            )

    def get_tool_descriptions(self) -> str:
        """Return formatted tool descriptions for the system prompt."""
        lines = ["Available tools:"]
        for name, tool in sorted(self._tools.items()):
            lines.append(f"  - {tool.description}")
        return "\n".join(lines)

    def list_tools(self) -> list:
        """Return list of tool names."""
        return sorted(self._tools.keys())
