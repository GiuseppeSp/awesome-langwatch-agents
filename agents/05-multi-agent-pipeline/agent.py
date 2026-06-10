"""
Multi-agent pipeline — orchestrator delegates to specialized worker agents.

Pattern: plan-and-execute, with an optional verification step.

    planner  →  writer  →  [fact_checker]  →  (writer revision if needed)

The planner extracts the key facts a good answer should cover. The writer
turns those facts into prose. In fact_checked mode, the fact_checker
verifies the writer's prose against the planner's facts and flags any
unsupported claims; if it does, the writer gets one revision pass.

Switch via env var:
    PIPELINE_MODE=basic        (default) — planner + writer only
    PIPELINE_MODE=fact_checked            — planner + writer + checker + revise

Every worker is a typed `agent` span under the workflow root, so the trace
tree visually mirrors the multi-agent structure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration ----

PIPELINE_MODE = os.getenv("PIPELINE_MODE", "basic")  # "basic" | "fact_checked"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
SERVICE_NAME = "multi-agent-pipeline"

os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- Worker prompts ----

_PLANNER_PROMPT = (
    "You are the PLANNER in a multi-agent pipeline. Your job is to read the "
    "user's question and produce a short, structured list of 3-7 key facts "
    "that a correct, complete answer would mention. Format: one fact per line, "
    "prefixed with '- '. Be specific about dates, names, places, and numbers. "
    "Do NOT write prose — only the fact list."
)

_WRITER_PROMPT = (
    "You are the WRITER in a multi-agent pipeline. You will be given a user's "
    "question and a fact list compiled by the PLANNER. Your job is to write a "
    "concise, well-structured answer (2-4 sentences) that incorporates every "
    "fact from the list. Don't invent additional facts. Don't omit facts from "
    "the list. Write naturally, not as a bullet list."
)

_FACT_CHECKER_PROMPT = (
    "You are the FACT-CHECKER in a multi-agent pipeline. You will receive the "
    "PLANNER's fact list and the WRITER's prose. Your job is to identify any "
    "claim in the prose that is NOT supported by the fact list, OR any required "
    "fact from the list that the writer omitted. Reply with:\n"
    "- 'OK' on the first line if the prose is fully grounded and complete\n"
    "- Otherwise, list specific issues, one per line, prefixed with '- '"
)


# ---- Helpers ----

def _llm(role_name: str, system: str, user: str, *, span_type: str = "agent") -> str:
    """One LLM call wrapped in a typed LangWatch span."""
    with langwatch.span(name=role_name, type=span_type) as s:
        s.update(input=user[:400])
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        out = (completion.choices[0].message.content or "").strip()
        s.update(
            output=out,
            metrics={
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
            },
        )
        return out


# ---- Worker functions ----

def plan(query: str) -> str:
    return _llm("planner", _PLANNER_PROMPT, f"Question: {query}")


def write(query: str, facts: str) -> str:
    user = f"Question: {query}\n\nFacts to cover:\n{facts}"
    return _llm("writer", _WRITER_PROMPT, user)


def fact_check(facts: str, prose: str) -> str:
    user = f"Fact list:\n{facts}\n\nWriter's prose:\n{prose}"
    return _llm("fact_checker", _FACT_CHECKER_PROMPT, user, span_type="evaluation")


def revise(query: str, facts: str, prose: str, issues: str) -> str:
    user = (
        f"Question: {query}\n\n"
        f"Facts to cover:\n{facts}\n\n"
        f"Your previous answer:\n{prose}\n\n"
        f"Issues the fact-checker raised:\n{issues}\n\n"
        "Rewrite the answer to fix these specific issues. Keep it concise."
    )
    return _llm("writer_revision", _WRITER_PROMPT, user)


# ---- The pipeline ----

@dataclass
class PipelineResult:
    query: str
    facts: str
    answer: str
    mode: str
    fact_check_issues: str = ""
    revised: bool = False
    worker_calls: list[str] = field(default_factory=list)


@langwatch.trace(name="multi_agent_pipeline")
def run(query: str, *, mode: str = PIPELINE_MODE) -> PipelineResult:
    """
    Run one query through the pipeline.

    Workers: planner → writer → [fact_checker → optional revision]. Each is a
    typed `agent` span (the fact-checker is `evaluation`) nested under the
    workflow root.
    """
    root = langwatch.get_current_trace().root_span
    root.update(input=query)

    worker_calls: list[str] = []
    facts = plan(query)
    worker_calls.append("planner")

    answer = write(query, facts)
    worker_calls.append("writer")

    fact_check_issues = ""
    revised = False

    if mode == "fact_checked":
        check_output = fact_check(facts, answer)
        worker_calls.append("fact_checker")
        first_line = check_output.split("\n", 1)[0].strip().upper()
        if first_line != "OK":
            fact_check_issues = check_output
            answer = revise(query, facts, answer, check_output)
            worker_calls.append("writer_revision")
            revised = True

    root.update(output=answer)
    return PipelineResult(
        query=query,
        facts=facts,
        answer=answer,
        mode=mode,
        fact_check_issues=fact_check_issues,
        revised=revised,
        worker_calls=worker_calls,
    )


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "When did the Apollo 11 mission land on the moon, and who were the astronauts?"
    )
    mode = os.getenv("PIPELINE_MODE", "basic")
    print(f"\n=== Query: {q}\n=== Mode: {mode}\n")
    result = run(q, mode=mode)
    print(f"Worker calls ({len(result.worker_calls)}): {' → '.join(result.worker_calls)}\n")
    print(f"PLANNER facts:\n{result.facts}\n")
    if result.fact_check_issues:
        print(f"FACT-CHECKER issues:\n{result.fact_check_issues}\n")
        print(f"Revised: {result.revised}\n")
    print(f"Final answer:\n{result.answer}")
