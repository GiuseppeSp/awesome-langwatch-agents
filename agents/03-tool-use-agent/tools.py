"""
Tool definitions for the tool-use agent.

Each tool is a plain Python function plus an OpenAI function-calling schema.
Two schema sets — VAGUE_TOOLS and PRECISE_TOOLS — describe the SAME four
functions in deliberately different ways. The agent.py switches between them
via an env var; everything else stays identical.

This is the entire experiment in two variables.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────────────────────────
# The four tools themselves — self-contained, no external API needed.
# ──────────────────────────────────────────────────────────────────────


def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression."""
    # Restrict allowed characters so eval() can't reach beyond arithmetic.
    if not re.match(r"^[0-9+\-*/().\s]+$", expression):
        return f"Error: invalid characters in expression: {expression!r}"
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307 — namespace is empty
        return str(result)
    except Exception as exc:  # pragma: no cover
        return f"Error: {exc}"


def date_parser(natural_date: str) -> str:
    """
    Convert a natural-language date description to ISO format.

    Handles: today, tomorrow, yesterday, day names (with optional 'next ' or
    'this ' qualifier). Anything more elaborate returns an error string so the
    agent can fall back to chat-only behavior.
    """
    today = datetime.now()
    nd = natural_date.lower().strip()
    if "today" in nd:
        return today.strftime("%Y-%m-%d")
    if "tomorrow" in nd:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    if "yesterday" in nd:
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, day in enumerate(days):
        if day in nd:
            current_dow = today.weekday()
            days_ahead = (i - current_dow + 7) % 7
            if "next" in nd and days_ahead == 0:
                days_ahead = 7
            if days_ahead == 0:
                days_ahead = 7  # "Tuesday" said on Tuesday = next Tuesday
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    return f"Error: couldn't parse date: {natural_date!r}"


_UNIT_FACTORS = {
    "length": {"m": 1.0, "km": 1000.0, "mi": 1609.34, "ft": 0.3048, "in": 0.0254, "cm": 0.01},
    "mass": {"g": 1.0, "kg": 1000.0, "lb": 453.592, "oz": 28.3495},
    "time": {"s": 1.0, "min": 60.0, "h": 3600.0, "hour": 3600.0, "day": 86400.0},
}


def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between units of the same category."""
    from_unit_lc = from_unit.lower().rstrip("s")  # tolerate plurals: "miles" → "mile"
    to_unit_lc = to_unit.lower().rstrip("s")
    # Aliases
    aliases = {
        "mile": "mi", "kilometer": "km", "meter": "m", "centimeter": "cm",
        "inch": "in", "foot": "ft", "feet": "ft",
        "pound": "lb", "ounce": "oz", "gram": "g", "kilogram": "kg",
        "second": "s", "minute": "min", "hour": "h",
    }
    from_unit_lc = aliases.get(from_unit_lc, from_unit_lc)
    to_unit_lc = aliases.get(to_unit_lc, to_unit_lc)

    # Temperature is non-linear — handle separately.
    if from_unit_lc in ("c", "celsius") and to_unit_lc in ("f", "fahrenheit"):
        return f"{value * 9 / 5 + 32:.2f} F"
    if from_unit_lc in ("f", "fahrenheit") and to_unit_lc in ("c", "celsius"):
        return f"{(value - 32) * 5 / 9:.2f} C"

    # Linear conversions within a category.
    for units in _UNIT_FACTORS.values():
        if from_unit_lc in units and to_unit_lc in units:
            converted = value * units[from_unit_lc] / units[to_unit_lc]
            return f"{converted:g} {to_unit_lc}"
    return f"Error: can't convert {from_unit!r} to {to_unit!r}"


# Hardcoded for reproducibility — real apps would hit a live rates API.
_CURRENCY_RATES_TO_USD = {
    "usd": 1.0, "eur": 1.08, "gbp": 1.27, "jpy": 0.0067,
    "cny": 0.14, "inr": 0.012, "cad": 0.74, "aud": 0.66, "chf": 1.14,
}


def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert money between currencies using a hardcoded rate table."""
    from_c = from_currency.lower()
    to_c = to_currency.lower()
    if from_c not in _CURRENCY_RATES_TO_USD or to_c not in _CURRENCY_RATES_TO_USD:
        supported = sorted(_CURRENCY_RATES_TO_USD.keys())
        return f"Error: unsupported currency. Supported: {supported}"
    usd_value = amount * _CURRENCY_RATES_TO_USD[from_c]
    converted = usd_value / _CURRENCY_RATES_TO_USD[to_c]
    return f"{converted:.2f} {to_c.upper()}"


# Dispatch table for the agent — name → callable.
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "date_parser": date_parser,
    "unit_converter": unit_converter,
    "currency_converter": currency_converter,
}


# ──────────────────────────────────────────────────────────────────────
# Schema set #1: VAGUE — deliberately ambiguous descriptions.
# ──────────────────────────────────────────────────────────────────────

VAGUE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Does math.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "date_parser",
            "description": "Handles dates.",
            "parameters": {
                "type": "object",
                "properties": {"natural_date": {"type": "string"}},
                "required": ["natural_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": "Converts things.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "currency_converter",
            "description": "Money stuff.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
]


# ──────────────────────────────────────────────────────────────────────
# Schema set #2: PRECISE — action-oriented descriptions with examples
# and explicit "use this when…" / "do NOT use this for…" framing.
# ──────────────────────────────────────────────────────────────────────

PRECISE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate an arithmetic expression like '2 * (3 + 4)' or '15 / 3 + 7'. "
                "Use this for ANY purely numerical calculation involving +, -, *, /, "
                "or parentheses. Do NOT use this for unit conversions or currency "
                "conversions — those have their own dedicated tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "An arithmetic expression with digits, operators (+,-,*,/), "
                            "and parentheses. Example: '12 * 4 + 3'."
                        ),
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "date_parser",
            "description": (
                "Convert a natural-language date description (like 'next Tuesday', "
                "'tomorrow', or 'today') into an ISO date (YYYY-MM-DD). Use this "
                "whenever the user asks 'what date is X?' where X is a relative or "
                "named day. Do NOT use this for arithmetic on numbers — for that, use "
                "the calculator."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "natural_date": {
                        "type": "string",
                        "description": "Natural-language date like 'tomorrow', 'next Friday', 'today'.",
                    },
                },
                "required": ["natural_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": (
                "Convert a physical measurement from one unit to another within the "
                "same category (length, mass, time, or temperature). Use this for "
                "questions like 'how many miles is 10 km?' or 'convert 32 F to C'. "
                "Do NOT use this for currency — currency has its own dedicated tool. "
                "Supported units: m/km/mi/ft/in/cm for length, g/kg/lb/oz for mass, "
                "s/min/h/day for time, C/F for temperature."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "The numeric value to convert."},
                    "from_unit": {"type": "string", "description": "Source unit symbol (e.g. 'km', 'lb', 'C')."},
                    "to_unit": {"type": "string", "description": "Target unit symbol (e.g. 'mi', 'kg', 'F')."},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "currency_converter",
            "description": (
                "Convert an amount of money from one currency to another (e.g. EUR to USD). "
                "Use this for any 'how much is X in Y?' question about money. Do NOT use "
                "this for physical unit conversions (km, miles, kg, etc.) — those have "
                "their own tool. Supported currencies: USD, EUR, GBP, JPY, CNY, INR, "
                "CAD, AUD, CHF."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount of money to convert."},
                    "from_currency": {"type": "string", "description": "Source currency code (USD, EUR, GBP, etc.)."},
                    "to_currency": {"type": "string", "description": "Target currency code."},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
]
