"""Datetime tools: datetime_now, timer_delta, unit_convert."""

import re
from datetime import datetime, timezone, timedelta

from ..tool_registry import ToolResult


class DatetimeNowTool:
    name = "datetime_now"
    description = (
        "Get current date, time, and timezone. "
        "Usage: [TOOL_CALL: datetime_now | any]"
    )

    def run(self, _arg: str) -> ToolResult:
        now_utc = datetime.now(timezone.utc)
        local = datetime.now()
        return ToolResult(
            self.name,
            f"UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"Local: {local.strftime('%Y-%m-%d %H:%M:%S')}",
        )


class TimerDeltaTool:
    name = "timer_delta"
    description = (
        "Calculate time between two dates (YYYY-MM-DD format). "
        "Usage: [TOOL_CALL: timer_delta | 2024-01-01 to 2024-12-31]"
    )

    def run(self, arg: str) -> ToolResult:
        try:
            # Parse "date1 to date2" or "date1, date2"
            parts = re.split(r'\s+to\s+|,\s*', arg.strip())
            if len(parts) != 2:
                return ToolResult(self.name, "", success=False,
                                  error="Provide two dates separated by 'to' or ','")
            d1 = datetime.strptime(parts[0].strip(), "%Y-%m-%d")
            d2 = datetime.strptime(parts[1].strip(), "%Y-%m-%d")
            delta = d2 - d1
            days = abs(delta.days)
            hours = days * 24
            return ToolResult(
                self.name,
                f"{parts[0].strip()} -> {parts[1].strip()}: "
                f"{delta.days} days ({hours} hours)",
            )
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


# Unit conversion factors (to base unit)
_LENGTH = {"m": 1, "km": 1000, "cm": 0.01, "mm": 0.001, "mi": 1609.344, "ft": 0.3048, "in": 0.0254, "yd": 0.9144}
_WEIGHT = {"kg": 1, "g": 0.001, "mg": 0.000001, "lb": 0.453592, "oz": 0.0283495, "ton": 907.185}
_TEMP_NAMES = {"c", "f", "k", "celsius", "fahrenheit", "kelvin"}

_ALL_UNITS = {}
_ALL_UNITS.update({k: ("length", v) for k, v in _LENGTH.items()})
_ALL_UNITS.update({k: ("weight", v) for k, v in _WEIGHT.items()})


class UnitConvertTool:
    name = "unit_convert"
    description = (
        "Convert between common units (length, weight, temperature). "
        "Usage: [TOOL_CALL: unit_convert | 5 km to mi]"
    )

    def run(self, arg: str) -> ToolResult:
        try:
            # Parse "VALUE UNIT to UNIT"
            match = re.match(
                r'([\d.]+)\s*(\w+)\s+(?:to|in)\s+(\w+)',
                arg.strip(),
                re.IGNORECASE,
            )
            if not match:
                return ToolResult(self.name, "", success=False,
                                  error="Format: VALUE UNIT to UNIT (e.g., 5 km to mi)")

            value = float(match.group(1))
            from_unit = match.group(2).lower()
            to_unit = match.group(3).lower()

            # Temperature special case
            if from_unit in _TEMP_NAMES or to_unit in _TEMP_NAMES:
                result = self._convert_temp(value, from_unit, to_unit)
                return ToolResult(self.name, f"{value} {from_unit} = {result:.4g} {to_unit}")

            # Standard unit conversion
            from_info = _ALL_UNITS.get(from_unit)
            to_info = _ALL_UNITS.get(to_unit)
            if not from_info or not to_info:
                return ToolResult(self.name, "", success=False,
                                  error=f"Unknown units: {from_unit}, {to_unit}")
            if from_info[0] != to_info[0]:
                return ToolResult(self.name, "", success=False,
                                  error=f"Cannot convert {from_info[0]} to {to_info[0]}")

            base_value = value * from_info[1]
            result = base_value / to_info[1]
            return ToolResult(self.name, f"{value} {from_unit} = {result:.6g} {to_unit}")
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))

    def _convert_temp(self, value: float, from_u: str, to_u: str) -> float:
        f, t = from_u[0], to_u[0]
        # Normalize to Celsius first
        if f == "f":
            c = (value - 32) * 5 / 9
        elif f == "k":
            c = value - 273.15
        else:
            c = value
        # Convert from Celsius
        if t == "f":
            return c * 9 / 5 + 32
        elif t == "k":
            return c + 273.15
        return c
