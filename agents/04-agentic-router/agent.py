"""
LLM-as-router agent — same code, two system prompts, A/B'd on a 25-query dataset.

The experiment: does adding 2-3 disambiguation examples per category, explicit
edge-case guidance, and a fallback rule actually move routing accuracy?

Switch via env var:
    PROMPT_STYLE=bare    (default) — minimal "classify into one of these" prompt
    PROMPT_STYLE=tuned             — bare + 3 examples per category + edge-case rules

Every call is one LangWatch trace with one llm-typed child span carrying the
prompt + the chosen route.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration ----

PROMPT_STYLE = os.getenv("PROMPT_STYLE", "bare")  # "bare" | "tuned"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
SERVICE_NAME = "agentic-router"

os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- Categories — loaded from JSON so they're configurable per project ----

with open(os.path.join(os.path.dirname(__file__) or ".", "categories.json"), encoding="utf-8") as _f:
    CATEGORIES = json.load(_f)["categories"]

CATEGORY_NAMES = [c["name"] for c in CATEGORIES]


# ---- The two prompt styles ----

def _bare_prompt() -> str:
    """
    Minimal classifier — no examples, no guidance, no edge-case rules.
    The model gets just the category names + their short descriptions.
    """
    listing = "\n".join(
        f"- {c['name']}: {c['short_description']}" for c in CATEGORIES
    )
    return (
        "You are a routing classifier. Read the user's query and respond with "
        "EXACTLY ONE category name from the list below.\n\n"
        f"Categories:\n{listing}\n\n"
        "Respond with only the category name, nothing else."
    )


def _tuned_prompt() -> str:
    """
    Same task with three additions over the bare version:
    (1) 3 example queries per category,
    (2) explicit edge-case guidance for ambiguous queries,
    (3) a fallback rule when nothing fits cleanly.
    """
    sections = []
    for c in CATEGORIES:
        examples = "\n".join(f"  - \"{ex}\"" for ex in c["examples"])
        sections.append(
            f"### {c['name']}\n{c['short_description']}\nExamples:\n{examples}"
        )
    rules = (
        "## Disambiguation rules\n"
        "1. If a query has both a factual lookup AND a generation/writing component, "
        "the **generation** is the deliverable — pick `creative_task`.\n"
        "2. If a query has both a math computation AND a code component, pick "
        "`code_question` only when the answer is an algorithm or program; pick "
        "`math_calculation` when the answer is a number.\n"
        "3. If a query asks for an opinion or recommendation that requires *generating* "
        "a new option (a name, a tagline, a dog name), prefer `creative_task` over "
        "`general_chat`.\n"
        "4. When in genuine doubt and no category clearly applies, default to "
        "`general_chat` rather than guessing."
    )
    return (
        "You are a routing classifier for a general AI assistant. Read the user's "
        "query and respond with EXACTLY ONE category name.\n\n"
        + "\n\n".join(sections)
        + "\n\n"
        + rules
        + "\n\nRespond with ONLY the category name, lowercase, no punctuation, no "
        "explanation."
    )


def _get_system_prompt(style: str) -> str:
    if style == "tuned":
        return _tuned_prompt()
    if style == "bare":
        return _bare_prompt()
    raise ValueError(f"Unknown PROMPT_STYLE: {style!r}")


# ---- The router ----

@dataclass
class RouteResult:
    query: str
    raw_output: str
    chosen: str   # parsed category name OR the literal "invalid:..." string
    style: str

    @property
    def is_valid_category(self) -> bool:
        return self.chosen in CATEGORY_NAMES


def _parse_output(raw: str) -> str:
    """
    Extract the chosen category. Tolerant of trailing punctuation, casing,
    and accidental wrapping like 'code_question.' or '"factual_question"'.
    """
    cleaned = raw.strip().strip(".,'\"").lower()
    # Sometimes models say "category: code_question" — grab the last token.
    last = re.split(r"\s+", cleaned)[-1]
    if last in CATEGORY_NAMES:
        return last
    # Sometimes the whole reply is just the category.
    if cleaned in CATEGORY_NAMES:
        return cleaned
    return f"invalid:{raw[:60]}"


@langwatch.trace(name="agentic_router")
def route(query: str, *, style: str = PROMPT_STYLE) -> RouteResult:
    """
    Classify one user query into one of the 5 categories.

    Returns a RouteResult with the raw model output and the parsed category.
    """
    root = langwatch.get_current_trace().root_span
    root.update(input=query)

    system = _get_system_prompt(style)
    with langwatch.span(name="classify", type="llm") as s:
        s.update(input=query)
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=20,
        )
        raw = (completion.choices[0].message.content or "").strip()
        s.update(
            output=raw,
            metrics={
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
            },
        )

    chosen = _parse_output(raw)
    root.update(output=chosen)
    return RouteResult(query=query, raw_output=raw, chosen=chosen, style=style)


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What's 17 times 84?"
    style = os.getenv("PROMPT_STYLE", "bare")
    print(f"\n=== Query: {q}\n=== Style: {style}\n")
    result = route(q, style=style)
    print(f"Raw output: {result.raw_output!r}")
    print(f"Chosen category: {result.chosen}")
    print(f"Valid category: {result.is_valid_category}")
