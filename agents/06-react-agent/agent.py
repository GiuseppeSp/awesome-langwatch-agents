"""
ReAct agent — interleaved Thought → Action → Observation loop with OpenAI
function calling against a small local knowledge base.

Switch via env var:
    REASONING_MODE=bare   (default) — model calls tools freely, no scratchpad
    REASONING_MODE=react            — model must emit a 'Thought:' line before
                                      each tool call (or final answer)

Both modes share the same loop, same tools, same termination condition. The
ONLY difference is the system prompt — that's what makes the comparison clean.

Each loop iteration is a typed LangWatch span (`agent`) under the workflow
root, with the LLM call (`llm`) and each tool call (`tool`) as child spans.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration ----

REASONING_MODE = os.getenv("REASONING_MODE", "bare")  # "bare" | "react"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "8"))
SERVICE_NAME = "react-agent"

os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- Knowledge base + tool implementations ----

KB_PATH = Path(__file__).parent / "kb.json"
KB = json.loads(KB_PATH.read_text())["facts"]


def search_kb(query: str) -> str:
    """Lenient keyword-overlap search over kb.json. Returns top-3 facts."""
    q_tokens = set(re.findall(r"\w+", query.lower()))
    scored = []
    for fact in KB:
        f_tokens = set(re.findall(r"\w+", fact["text"].lower()))
        overlap = len(q_tokens & f_tokens)
        if overlap > 0:
            scored.append((overlap, fact["text"]))
    scored.sort(key=lambda x: -x[0])
    top = scored[:3]
    if not top:
        return "No results found."
    return "\n".join(f"- {t}" for _, t in top)


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression. Only digits and +-*/().  allowed."""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return f"Error: only +, -, *, /, parentheses, and numbers allowed; got: {expression!r}"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


def current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


TOOLS_IMPL = {
    "search": lambda **kw: search_kb(kw["query"]),
    "calculator": lambda **kw: calculator(kw["expression"]),
    "current_date": lambda **kw: current_date(),
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Look up facts in the knowledge base. Returns up to 3 relevant facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression. Only +, -, *, /, parentheses, and numbers are supported.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The arithmetic expression to evaluate"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_date",
            "description": "Returns today's date in YYYY-MM-DD format. Use this when the question depends on knowing the current year.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---- System prompts ----

_BARE_SYSTEM = (
    "You are a helpful assistant that answers questions accurately. "
    "You have access to three tools: search (knowledge base lookup), "
    "calculator (arithmetic), and current_date (today's date). "
    "Use the tools as needed. When you have enough information, give a final answer."
)

_REACT_SYSTEM = (
    "You are a helpful assistant that solves problems using a Reason+Act loop. "
    "You have access to three tools: search (knowledge base lookup), "
    "calculator (arithmetic), and current_date (today's date).\n\n"
    "For EVERY response, follow this format strictly:\n"
    "1. First, write a single line starting with 'Thought:' explaining what you "
    "need to figure out next and why this tool/answer choice makes sense.\n"
    "2. Then either call ONE tool, OR if you have enough information, give the "
    "final answer prefixed with 'Final Answer:'.\n\n"
    "Always externalize your reasoning before acting. Do not skip the Thought step."
)


def _system_prompt(mode: str) -> str:
    return _REACT_SYSTEM if mode == "react" else _BARE_SYSTEM


# ---- The ReAct loop ----

@dataclass
class ReactResult:
    query: str
    mode: str
    answer: str
    tool_calls: list[str] = field(default_factory=list)
    thoughts: list[str] = field(default_factory=list)
    iterations: int = 0
    hit_iteration_limit: bool = False


@langwatch.trace(name="react_agent")
def run(query: str, *, mode: str = REASONING_MODE) -> ReactResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {query}")

    messages: list[dict] = [
        {"role": "system", "content": _system_prompt(mode)},
        {"role": "user", "content": query},
    ]
    tool_calls_made: list[str] = []
    thoughts: list[str] = []
    final_answer = ""
    hit_limit = False
    iteration = 0

    for iteration in range(1, MAX_ITERATIONS + 1):
        with langwatch.span(name=f"iteration_{iteration}", type="agent") as iter_span:
            with langwatch.span(name="reason", type="llm") as llm_span:
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    temperature=0,
                )
                msg = completion.choices[0].message
                llm_span.update(
                    input=str(messages[-1])[:300],
                    output=(msg.content or "")[:300],
                    metrics={
                        "prompt_tokens": completion.usage.prompt_tokens,
                        "completion_tokens": completion.usage.completion_tokens,
                    },
                )

            messages.append(msg.model_dump(exclude_unset=True))

            if msg.content and "Thought:" in msg.content:
                for line in msg.content.split("\n"):
                    if line.strip().startswith("Thought:"):
                        thoughts.append(line.strip()[len("Thought:"):].strip())

            if not msg.tool_calls:
                final_answer = (msg.content or "").strip()
                iter_span.update(output=f"final answer reached: {final_answer[:200]}")
                break

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                with langwatch.span(name=tc.function.name, type="tool") as tool_span:
                    tool_span.update(input=json.dumps(args))
                    impl = TOOLS_IMPL.get(tc.function.name)
                    result = impl(**args) if impl else f"Unknown tool: {tc.function.name}"
                    tool_span.update(output=str(result)[:500])

                tool_calls_made.append(tc.function.name)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

            iter_span.update(
                output=f"called {len(msg.tool_calls)} tool(s): "
                       f"{[tc.function.name for tc in msg.tool_calls]}"
            )
    else:
        hit_limit = True
        final_answer = "(no final answer — hit iteration limit)"

    root.update(output=final_answer)
    return ReactResult(
        query=query,
        mode=mode,
        answer=final_answer,
        tool_calls=tool_calls_made,
        thoughts=thoughts,
        iterations=iteration,
        hit_iteration_limit=hit_limit,
    )


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "How many years apart were the Eiffel Tower's completion and the Statue of Liberty's dedication?"
    )
    mode = os.getenv("REASONING_MODE", "bare")
    print(f"\n=== Query: {q}\n=== Mode: {mode}\n")
    result = run(q, mode=mode)
    print(f"Iterations: {result.iterations}{' (hit limit)' if result.hit_iteration_limit else ''}")
    print(f"Tool calls ({len(result.tool_calls)}): {result.tool_calls}")
    if result.thoughts:
        print(f"\nThoughts ({len(result.thoughts)}):")
        for t in result.thoughts:
            print(f"  - {t}")
    print(f"\nFinal answer:\n{result.answer}")
