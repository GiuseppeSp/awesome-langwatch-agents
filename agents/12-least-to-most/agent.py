"""
Least-to-most agent — decompose a problem into an ordered sequence of simpler
subproblems, then solve them one at a time, each step seeing the answers to the
previous ones.

Switch via env var:
    CAI_MODE  is not used here; use LTM_MODE.

    LTM_MODE=cot             (default) — one chain-of-thought pass. 1 call.
    LTM_MODE=least_to_most             — call 1 decomposes the problem into
                                         ordered subquestions; calls 2..N solve
                                         each subquestion in order, feeding the
                                         prior answers forward. 1 + N calls.

This is least-to-most prompting (Zhou et al., 2022) reduced to its core idea:
*explicit* decomposition + sequential solving, versus the *implicit*
decomposition a single chain-of-thought already does inside one call.

The domain is dependency-chain arithmetic word problems of increasing depth
(depth = number of sequential operations). Each step consumes the previous
result, so a single dropped sub-result poisons the final answer — which is
exactly where the paper claims explicit decomposition earns its keep, and
exactly the axis (problem depth) along which to look for the win.

Trace tree (typed LangWatch spans):

    least_to_most (workflow root)
    ├─ cot:           reason (llm)            one chain -> Answer: N
    └─ least_to_most:
       ├─ decompose   (llm)                   ordered subquestions Q1..QN
       ├─ solve_step  (llm)  x N              each subquestion, prior answers in context
       └─ (final answer = the last step's answer)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration ----

LTM_MODE = os.getenv("LTM_MODE", "cot")  # "cot" | "least_to_most"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
SERVICE_NAME = "least-to-most"

os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- Prompts ----

_COT_PROMPT = (
    "You are a careful problem solver. Solve the word problem step by step, "
    "then state the final answer on its own line in the exact form 'Answer: <number>'."
)

_DECOMPOSE_PROMPT = (
    "Break the problem into an ordered list of simpler subquestions that build on "
    "one another. Each subquestion should require a single arithmetic step and may "
    "use the answers to earlier subquestions. The final subquestion's answer must be "
    "the answer to the whole problem.\n"
    "Do NOT solve them. Output only the list, one per line, as 'Q1: ...', 'Q2: ...', etc."
)

_SOLVE_STEP_PROMPT = (
    "You are solving one subquestion of a larger problem. Use the original problem "
    "and the answers to the previous subquestions. Answer ONLY the current subquestion "
    "with a single number on its own line in the form 'Answer: <number>'."
)


# ---- LLM helper ----

def _chat(system: str, user: str) -> tuple[str, int, int]:
    c = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=TEMPERATURE,
    )
    return (c.choices[0].message.content or "").strip(), c.usage.prompt_tokens, c.usage.completion_tokens


# ---- Answer parsing ----

def extract_number(text: str) -> float | None:
    """Pull the answer out of model output. Prefer an explicit 'Answer: N' line;
    otherwise fall back to the last number in the text. Tolerates $, commas, %."""
    if not text:
        return None
    m = re.findall(r"answer\s*[:=]\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    candidates = m if m else re.findall(r"-?\$?\s*([\d,]+(?:\.\d+)?)", text)
    if not candidates:
        return None
    raw = candidates[-1].replace(",", "").replace("$", "").strip()
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_subquestions(text: str) -> list[str]:
    """Parse 'Q1: ...' / '1. ...' / '- ...' lines into a list of subquestions."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^(?:Q?\s*\d+\s*[:.)\-]|\-|\*)\s*(.+)$", line, flags=re.IGNORECASE)
        if m and m.group(1).strip():
            out.append(m.group(1).strip())
    return out


# ---- Result type ----

@dataclass
class LTMResult:
    problem: str
    mode: str
    answer: float | None = None
    final_text: str = ""
    subquestions: list[str] = field(default_factory=list)
    llm_calls: int = 0


# ---- The agent ----

@langwatch.trace(name="least_to_most")
def run(problem: str, *, mode: str = LTM_MODE) -> LTMResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {problem}")

    result = LTMResult(problem=problem, mode=mode)

    if mode == "cot":
        with langwatch.span(name="reason", type="llm") as s:
            s.update(input=problem)
            text, pt, ct = _chat(_COT_PROMPT, problem)
            s.update(output=text[:500], metrics={"prompt_tokens": pt, "completion_tokens": ct})
        result.llm_calls += 1
        result.final_text = text
        result.answer = extract_number(text)
        root.update(output=f"[cot] answer={result.answer}")
        return result

    if mode != "least_to_most":
        raise ValueError(f"unknown mode: {mode!r}")

    # --- decompose ---
    with langwatch.span(name="decompose", type="llm") as s:
        s.update(input=problem)
        decomp, pt, ct = _chat(_DECOMPOSE_PROMPT, problem)
        s.update(output=decomp[:500], metrics={"prompt_tokens": pt, "completion_tokens": ct})
    result.llm_calls += 1
    result.subquestions = _parse_subquestions(decomp)

    # Fallback: if decomposition produced nothing parseable, treat the whole
    # problem as a single subquestion so the run still yields an answer.
    subqs = result.subquestions or [problem]

    # --- solve each subquestion in order, prior answers carried forward ---
    history: list[str] = []
    last_answer: float | None = None
    for i, sub in enumerate(subqs, 1):
        ctx = (
            f"ORIGINAL PROBLEM:\n{problem}\n\n"
            + ("PREVIOUS SUBQUESTIONS AND ANSWERS:\n" + "\n".join(history) + "\n\n" if history else "")
            + f"CURRENT SUBQUESTION (Q{i}):\n{sub}"
        )
        with langwatch.span(name="solve_step", type="llm") as s:
            s.update(input=ctx[:500])
            text, pt, ct = _chat(_SOLVE_STEP_PROMPT, ctx)
            s.update(output=text[:300], metrics={"prompt_tokens": pt, "completion_tokens": ct})
        result.llm_calls += 1
        ans = extract_number(text)
        last_answer = ans if ans is not None else last_answer
        history.append(f"Q{i}: {sub}\nA{i}: {ans}")

    result.final_text = "\n".join(history)
    result.answer = last_answer
    root.update(output=f"[ltm {len(subqs)} steps] answer={result.answer}")
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "A baker has 24 eggs. She uses 6 to make a cake, then splits the rest equally into 3 cartons. How many eggs are in each carton?"
    )
    mode = os.getenv("LTM_MODE", "cot")
    print(f"\n=== Problem: {q}\n=== Mode: {mode}\n")
    res = run(q, mode=mode)
    if res.mode == "least_to_most":
        print("Decomposition:")
        for i, sub in enumerate(res.subquestions, 1):
            print(f"  Q{i}: {sub}")
        print(f"\nSequential solve:\n{res.final_text}\n")
    else:
        print(f"{res.final_text}\n")
    print(f"=== answer: {res.answer}  |  llm calls: {res.llm_calls}")
