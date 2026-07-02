"""
Code-interpreter agent — answer a question either by reasoning it out, or by
writing and RUNNING code.

Switch via env var:
    CI_MODE=reason  (default) — solve step by step in prose, end with 'Answer: X'.
                                No code. The model does the arithmetic in its head.
    CI_MODE=code              — write a short Python program that prints the answer,
                                execute it in a subprocess, use its stdout.

Same questions, one LLM call each; the difference is whether the computation runs
on a CPU or in the model's head. The questions are deliberately computational
(large products, letter counts, date math, compound interest, powers) — exactly
where LLM mental arithmetic is unreliable. The experiment measures where handing
the work to an interpreter pays, and where it's just overhead.

Trace tree (typed LangWatch spans):

    code_interpreter (workflow root)
    ├─ reason: reason (llm)                          -> parse 'Answer: X'
    └─ code:   write_code (llm) -> execute (tool)    -> stdout
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration ----

CI_MODE = os.getenv("CI_MODE", "reason")  # "reason" | "code"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
EXEC_TIMEOUT = int(os.getenv("EXEC_TIMEOUT", "5"))

os.environ.setdefault("OTEL_SERVICE_NAME", "code-interpreter")
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- Prompts ----

_REASON_SYSTEM = (
    "Solve the problem step by step, doing any arithmetic yourself. "
    "End with a line 'Answer: <value>' giving ONLY the final value (a number)."
)

_CODE_SYSTEM = (
    "Write a short Python 3 program that computes the answer and prints ONLY the final "
    "value (the number) to stdout. You may use the standard library (math, datetime, "
    "statistics). Output only the code — no explanation, no markdown fences."
)


# ---- Helpers ----

def _extract_answer(text: str) -> str:
    m = re.findall(r"answer\s*[:=]\s*\$?\s*([-\d,]+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    cand = m[-1] if m else (re.findall(r"-?\d[\d,]*(?:\.\d+)?", text) or [""])[-1]
    return cand.replace(",", "").strip()


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.+?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else text).strip()


def run_code(code: str) -> tuple[str, str | None]:
    """Execute Python in a subprocess; return (stdout_answer, error)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=EXEC_TIMEOUT,
        )
        if proc.returncode != 0:
            return "", (proc.stderr.strip().splitlines() or ["nonzero exit"])[-1]
        out = proc.stdout.strip().splitlines()
        return (out[-1].strip() if out else ""), None
    except subprocess.TimeoutExpired:
        return "", f"timed out after {EXEC_TIMEOUT}s"
    except Exception as e:  # noqa: BLE001
        return "", str(e)


def _chat(system: str, user: str) -> tuple[str, int, int]:
    c = client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (c.choices[0].message.content or "").strip(), c.usage.prompt_tokens, c.usage.completion_tokens


# ---- Result type ----

@dataclass
class CIResult:
    question: str
    mode: str
    answer: str = ""
    code: str = ""
    error: str | None = None
    raw: str = ""
    llm_calls: int = 0

    @property
    def executed_ok(self) -> bool:
        return self.mode == "reason" or self.error is None


# ---- The agent ----

@langwatch.trace(name="code_interpreter")
def run(question: str, *, mode: str = CI_MODE) -> CIResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {question}")
    result = CIResult(question=question, mode=mode)

    if mode == "reason":
        with langwatch.span(name="reason", type="llm") as s:
            s.update(input=question)
            text, pt, ct = _chat(_REASON_SYSTEM, question)
            s.update(output=text[:400], metrics={"prompt_tokens": pt, "completion_tokens": ct})
        result.llm_calls += 1
        result.raw = text
        result.answer = _extract_answer(text)
        root.update(output=f"[reason] {result.answer}")
        return result

    if mode != "code":
        raise ValueError(f"unknown mode: {mode!r}")

    with langwatch.span(name="write_code", type="llm") as s:
        s.update(input=question)
        text, pt, ct = _chat(_CODE_SYSTEM, question)
        s.update(output=text[:400], metrics={"prompt_tokens": pt, "completion_tokens": ct})
    result.llm_calls += 1
    result.raw = text
    result.code = _extract_code(text)

    with langwatch.span(name="execute", type="tool") as s:
        s.update(input=result.code[:400])
        stdout, err = run_code(result.code)
        result.error = err
        s.update(output=(f"ERROR: {err}" if err else stdout)[:300])
    result.answer = "" if result.error else _extract_answer(stdout) or stdout.strip()

    root.update(output=f"[code ok={result.executed_ok}] {result.answer or result.error}"[:200])
    return result


# ---- CLI ----

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How many times does the letter 'r' appear in 'strawberry raspberry mirror'?"
    mode = os.getenv("CI_MODE", "reason")
    print(f"\n=== Question: {q}\n=== Mode: {mode}\n")
    res = run(q, mode=mode)
    if res.mode == "code":
        print(f"Code:\n{res.code}\n")
        print(f"Executed: {res.executed_ok}" + (f"  (ERROR: {res.error})" if res.error else ""))
    else:
        print(f"Reasoning:\n{res.raw}\n")
    print(f"Answer: {res.answer}")
