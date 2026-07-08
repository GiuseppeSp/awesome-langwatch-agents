"""
Supervisor-worker agent — does splitting a bundled request across a supervisor and
per-subtask workers beat one agent doing the whole bundle in a single pass?

Switch via env var:
    MA_MODE=single            (default) — ONE llm call answers the whole bundle of N
                                          subtasks, returning a JSON map.
    MA_MODE=supervisor_worker           — a supervisor decomposes the bundle into N
                                          subtasks, dispatches ONE worker call per
                                          subtask, and aggregates the answers.

Both see the identical numbered request. The only difference is whether each subtask
gets its own isolated worker call. Every subtask is an atomic op a capable model
nails in isolation, so the experiment isolates ONE thing: as the bundle grows, does a
single pass start dropping / fumbling subtasks (attention split across N instructions)
while per-worker isolation holds — and at what bundle size does that crossover appear,
against supervisor_worker's N+1x call cost.

Trace tree (typed LangWatch spans):

    supervisor_worker (workflow root)
    ├─ single:            solve (llm, all N subtasks -> JSON)
    └─ supervisor_worker: decompose (llm) -> worker (agent) xN -> aggregate
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration ----

MA_MODE = os.getenv("MA_MODE", "single")  # "single" | "supervisor_worker"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

os.environ.setdefault("OTEL_SERVICE_NAME", "supervisor-worker")
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- LLM helpers ----

def _chat(system: str, user: str, json_mode: bool = False) -> str:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    c = client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **kwargs,
    )
    return (c.choices[0].message.content or "").strip()


def _numbered(instrs: list[str]) -> str:
    return "\n".join(f"{i}. {t}" for i, t in enumerate(instrs, 1))


# ---- Result type ----

@dataclass
class MAResult:
    mode: str
    answers: dict[str, str] = field(default_factory=dict)  # task-number (str) -> answer
    llm_calls: int = 0
    n_decomposed: int = 0  # supervisor_worker: how many subtasks the supervisor produced


# ---- single: one call does the whole bundle ----

_SINGLE_SYS = (
    "You complete every numbered task the user gives you. Answer ALL of them.\n"
    "Respond with ONE JSON object mapping each task number (as a string: \"1\", \"2\", ...) "
    "to its answer as a short string. Include every task number. Answers only — no explanation."
)


def _run_single(instrs: list[str], result: MAResult) -> None:
    with langwatch.span(name="solve", type="llm") as s:
        s.update(input=f"{len(instrs)} tasks")
        raw = _chat(_SINGLE_SYS, _numbered(instrs), json_mode=True)
        s.update(output=raw[:300])
    result.llm_calls += 1
    try:
        d = json.loads(raw)
        result.answers = {str(k): str(v) for k, v in d.items()}
    except json.JSONDecodeError:
        result.answers = {}


# ---- supervisor_worker: decompose -> per-subtask workers -> aggregate ----

_DECOMPOSE_SYS = (
    "You are a supervisor. Split the user's numbered request into its individual tasks.\n"
    "Respond with ONE JSON object: {\"tasks\": [\"<task 1>\", \"<task 2>\", ...]} — one entry "
    "per task, in order, each the full standalone instruction. Do not solve them."
)

_WORKER_SYS = (
    "You are a worker with exactly one task. Do it and reply with ONLY the answer — "
    "a short string, no label, no explanation."
)


def _run_supervisor_worker(instrs: list[str], result: MAResult) -> None:
    with langwatch.span(name="decompose", type="llm") as s:
        s.update(input=f"{len(instrs)} tasks")
        raw = _chat(_DECOMPOSE_SYS, _numbered(instrs), json_mode=True)
        try:
            subtasks = [str(t) for t in json.loads(raw).get("tasks", [])]
        except json.JSONDecodeError:
            subtasks = []
        s.update(output=f"decomposed into {len(subtasks)} subtasks")
    result.llm_calls += 1
    result.n_decomposed = len(subtasks)

    for i, task in enumerate(subtasks, 1):
        with langwatch.span(name="worker", type="agent") as s:
            s.update(input=task)
            ans = _chat(_WORKER_SYS, task)
            s.update(output=ans[:120])
        result.llm_calls += 1
        result.answers[str(i)] = ans


# ---- The agent ----

@langwatch.trace(name="supervisor_worker")
def run(instrs: list[str], *, mode: str = MA_MODE) -> MAResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {len(instrs)} tasks")

    if mode not in ("single", "supervisor_worker"):
        raise ValueError(f"unknown mode: {mode!r}")

    result = MAResult(mode=mode)
    if mode == "single":
        _run_single(instrs, result)
    else:
        _run_supervisor_worker(instrs, result)

    answered = sum(1 for v in result.answers.values() if v.strip())
    root.update(output=f"[{mode}] answered {answered}/{len(instrs)} ({result.llm_calls} llm calls)")
    return result


# ---- CLI ----

if __name__ == "__main__":
    demo = [
        'Count the number of words in this sentence: "the early bird catches the worm every morning"',
        "Multiply 42 by 17.",
        'Reverse the string "rocket".',
        "Sort these numbers in ascending order: 44, 12, 88, 30.",
        'What is the last letter of the word "silver"?',
    ]
    mode = os.getenv("MA_MODE", "single")
    print(f"\n=== Bundle of {len(demo)} tasks · mode: {mode}\n")
    res = run(demo, mode=mode)
    for i, t in enumerate(demo, 1):
        print(f"  {i}. {t[:60]}\n     -> {res.answers.get(str(i), '(no answer)')}")
    print(f"\nllm calls: {res.llm_calls}")
