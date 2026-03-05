"""Compute tools: calculator, python_exec."""

import logging
import math
import subprocess
import sys
import tempfile
import os

from ..tool_registry import ToolResult

# Safe math namespace for calculator
_SAFE_MATH = {
    k: getattr(math, k) for k in dir(math) if not k.startswith("_")
}
_SAFE_MATH.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})


class CalculatorTool:
    name = "calculator"
    description = (
        "Evaluate a math expression safely. "
        "Usage: [TOOL_CALL: calculator | 2**32 + sqrt(144)]"
    )

    def run(self, expression: str) -> ToolResult:
        try:
            expr = expression.strip()
            # Basic safety: only allow math-like characters
            if any(c in expr for c in ("import", "exec", "eval", "__", "open")):
                return ToolResult(self.name, "", success=False,
                                  error="Expression contains blocked keywords.")
            result = eval(expr, {"__builtins__": {}}, _SAFE_MATH)
            return ToolResult(self.name, str(result))
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


class PythonExecTool:
    name = "python_exec"
    description = (
        "Execute a Python code snippet in a sandboxed subprocess. "
        "Usage: [TOOL_CALL: python_exec | print('hello')]"
    )

    def run(self, code: str) -> ToolResult:
        try:
            code = code.strip()
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8",
            ) as f:
                f.write(code)
                tmp_path = f.name

            try:
                result = subprocess.run(
                    [sys.executable, tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                output = result.stdout
                if result.stderr:
                    output += "\n[STDERR]\n" + result.stderr
                if len(output) > 4000:
                    output = output[:4000] + "\n...[truncated]"
                return ToolResult(self.name, output or "(no output)")
            finally:
                os.unlink(tmp_path)
        except subprocess.TimeoutExpired:
            return ToolResult(self.name, "", success=False, error="Execution timed out (30s).")
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))
