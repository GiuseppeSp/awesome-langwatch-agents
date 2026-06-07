"""
Multi-turn travel-planning chatbot — same pattern, different memory strategies.

Compares two ways of handling growing conversation history:

    window  — keep only the most recent N turns verbatim, drop everything older
    summary — keep the most recent N turns verbatim, replace older turns with an
              LLM-generated summary prepended as a system message

The strategy is a single env var (MEMORY_STRATEGY=window|summary), so the same
agent code drives both comparisons in run_eval.py.

Every call is one LangWatch trace. When a thread_id is passed, all turns of the
same conversation group together in LangWatch's Thread tab — the whole point of
having a chatbot agent in the catalog at all.
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration knobs (the tuning experiment changes these) ----

MEMORY_STRATEGY: Literal["window", "summary"] = os.getenv("MEMORY_STRATEGY", "window")  # type: ignore[assignment]
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "4"))     # turns kept verbatim
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")
SERVICE_NAME = "multi-turn-chatbot"

SYSTEM_PROMPT = (
    "You are a knowledgeable, friendly travel-planning assistant. Use everything you remember "
    "about the user's trip, preferences, and constraints to give recommendations they can actually use. "
    "Be specific. Don't ignore details they've shared earlier in the conversation."
)

# Service name via env (the Python SDK's setup() doesn't take it as a kwarg).
os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- Memory strategies ----

def _summarize_history(history: list[dict]) -> str:
    """
    Compress a list of older turns into a 2-3 sentence summary that captures the
    user's stated facts and constraints. Uses a small dedicated model to keep
    cost down — summarization happens on every query in the summary strategy.
    """
    transcript = "\n".join(f"{t['role']}: {t['content']}" for t in history)
    prompt = (
        "Summarize the following conversation in 2-3 sentences. Focus on facts the user "
        "shared about themselves, their trip, their preferences, and any constraints "
        "(dietary, budget, physical, family). Drop small talk.\n\n"
        f"Conversation:\n{transcript}"
    )
    with langwatch.span(name="summarize_history", type="llm") as s:
        s.update(input=f"{len(history)} turns to summarize")
        completion = client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        summary = completion.choices[0].message.content or ""
        s.update(
            output=summary,
            metrics={
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
            },
        )
    return summary


def _build_memory(
    history: list[dict],
    strategy: str,
    window: int,
) -> list[dict]:
    """
    Return the message list to send to the LLM, given a strategy.

    window  — last `window` turns only; older turns are gone
    summary — summary of older turns (as a system message) + last `window` turns
    """
    if len(history) <= window:
        return list(history)
    if strategy == "window":
        return list(history[-window:])
    if strategy == "summary":
        older = history[:-window]
        recent = list(history[-window:])
        summary = _summarize_history(older)
        return [
            {
                "role": "system",
                "content": f"Summary of earlier conversation: {summary}",
            },
            *recent,
        ]
    raise ValueError(f"Unknown MEMORY_STRATEGY: {strategy!r}")


# ---- The chatbot ----

@langwatch.trace(name="travel_chatbot")
def chat(
    history: list[dict],
    new_message: str,
    *,
    strategy: str = MEMORY_STRATEGY,
    window: int = MEMORY_WINDOW,
    thread_id: str | None = None,
) -> str:
    """
    Answer one user turn given the prior conversation history.

    Args:
        history: prior turns as [{role, content}, ...]
        new_message: the current user message
        strategy: "window" or "summary"
        window: how many recent turns to keep verbatim
        thread_id: optional conversation identifier — when set, all turns of the
            same conversation group together in LangWatch's Thread tab.
    """
    root_span = langwatch.get_current_trace().root_span
    root_span.update(input=new_message)
    if thread_id is not None:
        # LangWatch threads — groups multi-turn traces in the dashboard.
        try:
            langwatch.get_current_trace().update(thread_id=thread_id)
        except Exception:
            # Fallback: set as an attribute on the root span if the trace-level
            # API isn't available in the installed SDK version.
            root_span.update(input=new_message)

    # Build the actual message list sent to the LLM.
    with langwatch.span(name="build_memory", type="span") as s:
        s.update(input=f"strategy={strategy} history_turns={len(history)} window={window}")
        memory = _build_memory(history, strategy, window)
        s.update(output=f"{len(memory)} messages prepared for LLM")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *memory,
        {"role": "user", "content": new_message},
    ]

    # Synthesize the response.
    with langwatch.span(name="synthesize", type="llm") as s:
        s.update(input=new_message)
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0,
        )
        answer = completion.choices[0].message.content or ""
        s.update(
            output=answer,
            metrics={
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            },
        )

    root_span.update(output=answer)
    return answer


# ---- CLI: replay one conversation, ask the test question, print the answer ----

if __name__ == "__main__":
    import json
    import sys

    path = "conversations.json"
    convo_id = sys.argv[1] if len(sys.argv) > 1 else "tokyo-vegan"
    strategy = sys.argv[2] if len(sys.argv) > 2 else MEMORY_STRATEGY

    with open(path, encoding="utf-8") as f:
        conversations = json.load(f)
    convo = next((c for c in conversations if c["id"] == convo_id), None)
    if convo is None:
        print(f"No conversation with id={convo_id!r}. Available:")
        for c in conversations:
            print(f"  - {c['id']}")
        sys.exit(1)

    print(f"\n=== Conversation: {convo['id']} (strategy={strategy}) ===")
    print(f"Setup: {len(convo['setup'])} turns")
    print(f"Test question: {convo['test_question']}")
    print(f"Expected constraint: {convo['expected_constraint']}\n")

    answer = chat(
        convo["setup"],
        convo["test_question"],
        strategy=strategy,
        thread_id=f"{convo_id}-{strategy}",
    )
    print(f"Answer:\n{answer}\n")
