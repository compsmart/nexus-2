"""Filesystem tools: file_read, file_write, file_list, file_search."""

import fnmatch
import os

from ..tool_registry import ToolResult


class FileReadTool:
    name = "file_read"
    description = (
        "Read a text file from disk. "
        "Usage: [TOOL_CALL: file_read | path/to/file.txt]"
    )

    def run(self, path: str) -> ToolResult:
        try:
            path = path.strip()
            if not os.path.isfile(path):
                return ToolResult(self.name, "", success=False,
                                  error=f"File not found: {path}")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if len(content) > 8000:
                content = content[:8000] + "\n...[truncated]"
            return ToolResult(self.name, content)
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


class FileWriteTool:
    name = "file_write"
    description = (
        "Write text to a file. Use 'append:' prefix to append. "
        "Usage: [TOOL_CALL: file_write | path/to/file.txt:::content here]"
    )

    def run(self, arg: str) -> ToolResult:
        try:
            if ":::" not in arg:
                return ToolResult(self.name, "", success=False,
                                  error="Format: path:::content")
            path, content = arg.split(":::", 1)
            path = path.strip()
            mode = "a" if path.startswith("append:") else "w"
            if mode == "a":
                path = path[7:].strip()
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, mode, encoding="utf-8") as f:
                f.write(content)
            action = "appended to" if mode == "a" else "written to"
            return ToolResult(self.name, f"Content {action} {path}")
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


class FileListTool:
    name = "file_list"
    description = (
        "List files and directories in a path. "
        "Usage: [TOOL_CALL: file_list | path/to/directory]"
    )

    def run(self, path: str) -> ToolResult:
        try:
            path = path.strip() or "."
            if not os.path.isdir(path):
                return ToolResult(self.name, "", success=False,
                                  error=f"Not a directory: {path}")
            entries = sorted(os.listdir(path))
            lines = []
            for e in entries[:100]:
                full = os.path.join(path, e)
                kind = "DIR" if os.path.isdir(full) else "FILE"
                lines.append(f"  [{kind}] {e}")
            result = f"{path} ({len(entries)} entries):\n" + "\n".join(lines)
            if len(entries) > 100:
                result += f"\n  ...and {len(entries) - 100} more"
            return ToolResult(self.name, result)
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


class FileSearchTool:
    name = "file_search"
    description = (
        "Search for files by name pattern or content substring. "
        "Usage: [TOOL_CALL: file_search | *.py in /path] or "
        "[TOOL_CALL: file_search | content:search_term in /path]"
    )

    def run(self, arg: str) -> ToolResult:
        try:
            # Parse "pattern in path"
            if " in " in arg:
                pattern, search_path = arg.rsplit(" in ", 1)
                pattern = pattern.strip()
                search_path = search_path.strip()
            else:
                pattern = arg.strip()
                search_path = "."

            if not os.path.isdir(search_path):
                return ToolResult(self.name, "", success=False,
                                  error=f"Not a directory: {search_path}")

            content_search = pattern.startswith("content:")
            if content_search:
                substr = pattern[8:].strip()
                return self._search_content(substr, search_path)
            else:
                return self._search_name(pattern, search_path)
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))

    def _search_name(self, pattern: str, search_path: str) -> ToolResult:
        matches = []
        for root, dirs, files in os.walk(search_path):
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    matches.append(os.path.join(root, f))
                    if len(matches) >= 50:
                        break
            if len(matches) >= 50:
                break
        if not matches:
            return ToolResult(self.name, f"No files matching '{pattern}' found.")
        return ToolResult(self.name, "\n".join(matches))

    def _search_content(self, substr: str, search_path: str) -> ToolResult:
        matches = []
        for root, dirs, files in os.walk(search_path):
            for f in files:
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                        for i, line in enumerate(fh, 1):
                            if substr in line:
                                matches.append(f"{path}:{i}: {line.strip()[:120]}")
                                break
                except (OSError, UnicodeDecodeError):
                    pass
                if len(matches) >= 30:
                    break
            if len(matches) >= 30:
                break
        if not matches:
            return ToolResult(self.name, f"No files containing '{substr}' found.")
        return ToolResult(self.name, "\n".join(matches))
