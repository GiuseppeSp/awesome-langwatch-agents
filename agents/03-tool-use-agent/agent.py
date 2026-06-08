"""
Tool-use agent — same code, two tool-description sets, A/B'd on a 20-row dataset.

The experiment is simple: does rewriting tool descriptions from vague
("Does math") to precise ("Evaluate an arithmetic expression. Do NOT use for
unit conversions.") actually move tool-selection accuracy?

Switch via env var:
    TOOL_DESCRIPTIONS=vague    (default)
    TOOL_DESCRIPTIONS=precise

Every call is one LangWatch trace. Each tool the agent invokes becomes a
typed `tool` span with the args + result captured inline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

import langwatch
import tools

load_dotenv()

# ---- Configuration ----

TOOL_DESCRIPTIONS = os.getenv("TOOL_DESCRIPTIONS", "vague")  # "vague" | "precise"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
SERVICE_NAME = "tool-use-agent"

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a few tools. "
    "When a user's question genuinely requires one of those tools, call it. "
    "When the question is general knowledge or doesn't need a tool, just answer directly."
)

os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


def _tool_schemas(mode: str) -> list[dict]:
    """Pick the schema set based on env var. Both name the same 4 functions."""
    if mode == "precise":
        return tools.PRECISE_TOOLS
    if mode == "vague":
        return tools.VAGUE_TOOLS
    raise ValueError(f"Unknown TOOL_DESCRIPTIONS mode: {mode!r}")


# ---- The agent ----

@dataclass
class ToolCall:
    name: str
    args: dict
    result: str


@dataclass
class AgentResult:
    query: str
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    mode: str = ""

    @property
    def first_tool(self) -> str:
        """The first tool the agent chose to call, or 'none' if it called none."""
        return self.tool_calls[0].name if self.tool_calls else "none"


@langwatch.trace(name="tool_use_agent")
def answer(query: str, *, mode: str = TOOL_DESCRIPTIONS) -> AgentResult:
    """
    Answer one query. Returns the final text answer plus a record of which
    tools were called and with what arguments.
    """
    root = langwatch.get_current_trace().root_span
    root.update(input=query)

    schemas = _tool_schemas(mode)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    tool_calls_made: list[ToolCall] = []
    # Bounded loop — caps runaway agents at a sensible number of turns.
    for _ in range(5):
        with langwatch.span(name="model_turn", type="llm") as s:
            s.update(input=json.dumps(messages[-1])[:200])
            completion = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=schemas,
                tool_choice="auto",
                temperature=0,
            )
            msg = completion.choices[0].message
            s.update(
                output=(msg.content or "").strip() or f"[{len(msg.tool_calls or [])} tool call(s)]",
                metrics={
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                },
            )

        # If the model didn't ask for a tool, we have the final answer.
        if not msg.tool_calls:
            final = (msg.content or "").strip()
            root.update(output=final)
            return AgentResult(query=query, answer=final, tool_calls=tool_calls_made, mode=mode)

        # Otherwise run each requested tool, attach the result, loop.
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            with langwatch.span(name=f"tool:{tool_name}", type="tool") as s:
                s.update(input=json.dumps(args)[:200])
                fn = tools.TOOL_FUNCTIONS.get(tool_name)
                if fn is None:
                    result = f"Error: unknown tool {tool_name!r}"
                else:
                    try:
                        result = fn(**args)
                    except Exception as exc:
                        result = f"Error executing {tool_name}: {exc}"
                s.update(output=str(result)[:500])
            tool_calls_made.append(ToolCall(name=tool_name, args=args, result=result))
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # Loop limit hit without a final answer — extremely unlikely.
    root.update(output="[agent exceeded turn limit]")
    return AgentResult(
        query=query, answer="[exceeded turn limit]", tool_calls=tool_calls_made, mode=mode,
    )


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is 15 percent of 240?"
    mode = os.getenv("TOOL_DESCRIPTIONS", "vague")
    print(f"\n=== Query: {q}\n=== Mode: {mode}\n")
    result = answer(q, mode=mode)
    print(f"Answer: {result.answer}\n")
    print(f"Tool calls ({len(result.tool_calls)}):")
    for tc in result.tool_calls:
        print(f"  - {tc.name}({tc.args}) → {tc.result}")
