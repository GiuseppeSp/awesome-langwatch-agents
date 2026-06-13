"""
Plan-and-Execute agent — plan the whole task upfront, then execute the plan.

Switch via env var:
    PLAN_MODE=static  (default) — planner writes a flat, unconditional plan once;
                                  the executor runs every step in order and never
                                  re-deliberates, even when an observation makes a
                                  later step wrong.
    PLAN_MODE=replan            — same planner, but after each observation a monitor
                                  decides whether reality has diverged from the plan
                                  and, if so, re-invokes the planner for the remaining
                                  steps (up to MAX_REPLANS times).

This is the classic Plan-and-Execute pattern: one planning pass, then mechanical
execution. Its textbook weakness is brittleness — when the world deviates from the
plan, a static executor cannot adapt.

The single design decision that makes the static/replan contrast honest:

    The EXECUTOR never sees the original task. It sees only the current plan step
    and the observations so far. So it physically cannot re-derive a branch the
    plan committed to wrongly — it just executes the step as written. The PLANNER
    (and, in replan mode, the monitor) is the only component that sees the goal.
    That is exactly the plan/execute separation: the executor doesn't re-think
    strategy, it carries out instructions.

Trace tree (one typed LangWatch span per node):

    plan_execute (workflow root)
    ├─ plan          (llm)        the upfront plan
    ├─ step_1        (agent)  → act (llm) + lookup/calc/report (tool)
    ├─ monitor       (evaluation) [replan mode only] continue or replan?
    ├─ step_2        (agent)  → act (llm) + tool
    ├─ replan        (llm)        [replan mode only, when monitor diverges]
    └─ step_3        (agent)  → act (llm) + report (tool)
"""

from __future__ import annotations

import ast
import json
import os
import re
import operator
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration ----

PLAN_MODE = os.getenv("PLAN_MODE", "static")  # "static" | "replan"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
MAX_REPLANS = int(os.getenv("MAX_REPLANS", "2"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
SERVICE_NAME = "plan-and-execute"

os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- The world: a tiny deterministic knowledge base ----
#
# Populations are in millions. The numbers are internally consistent with
# dataset.csv's ground-truth answers; they are not meant to be real-world exact.

CITIES = {
    "tokyo":   {"population_m": 13.5, "country": "japan",    "area_km2": 2194},
    "delhi":   {"population_m": 16.8, "country": "india",    "area_km2": 1484},
    "paris":   {"population_m": 2.1,  "country": "france",   "area_km2": 105},
    "berlin":  {"population_m": 3.7,  "country": "germany",  "area_km2": 891},
    "madrid":  {"population_m": 3.3,  "country": "spain",    "area_km2": 604},
    "rome":    {"population_m": 2.8,  "country": "italy",    "area_km2": 1285},
    "oslo":    {"population_m": 0.7,  "country": "norway",   "area_km2": 454},
    "lisbon":  {"population_m": 0.5,  "country": "portugal", "area_km2": 100},
    "cairo":   {"population_m": 9.5,  "country": "egypt",    "area_km2": 606},
    "toronto": {"population_m": 2.9,  "country": "canada",   "area_km2": 631},
}

COUNTRIES = {
    "japan":    {"population_m": 125.0,  "capital": "tokyo"},
    "india":    {"population_m": 1408.0, "capital": "delhi"},
    "france":   {"population_m": 67.0,   "capital": "paris"},
    "germany":  {"population_m": 83.0,   "capital": "berlin"},
    "spain":    {"population_m": 47.0,   "capital": "madrid"},
    "italy":    {"population_m": 59.0,   "capital": "rome"},
    "norway":   {"population_m": 5.4,    "capital": "oslo"},
    "portugal": {"population_m": 10.3,   "capital": "lisbon"},
    "egypt":    {"population_m": 109.0,  "capital": "cairo"},
    "canada":   {"population_m": 38.0,   "capital": "ottawa"},
}

_ATTR_ALIASES = {
    "population": "population_m",
    "pop": "population_m",
    "population_m": "population_m",
    "area": "area_km2",
    "area_km2": "area_km2",
    "country": "country",
    "capital": "capital",
}

TOOL_SCHEMA = (
    "Available tools (output exactly one as JSON):\n"
    '  {"tool": "lookup", "args": {"entity": "<city or country>", "attribute": '
    '"population|area|country|capital"}}\n'
    '  {"tool": "calc", "args": {"expression": "<arithmetic, e.g. 2.1+3.7>"}}\n'
    '  {"tool": "report", "args": {"value": <the final numeric answer>}}\n'
    "Notes: populations are in millions. 'lookup' on a city accepts population/"
    "area/country; on a country accepts population/capital. Reference an earlier "
    "result by reading it from the observations you are given."
)


# ---- Tools (deterministic) ----

_CALC_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _safe_calc(expr: str) -> float:
    """Evaluate a pure arithmetic expression without using Python's eval()."""
    def _node(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _CALC_OPS:
            return _CALC_OPS[type(n.op)](_node(n.left), _node(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _CALC_OPS:
            return _CALC_OPS[type(n.op)](_node(n.operand))
        raise ValueError(f"unsupported expression: {expr!r}")
    return float(_node(ast.parse(expr, mode="eval").body))


def run_tool(tool: str, args: dict) -> str:
    """Execute a tool call against the world. Returns a string observation."""
    if tool == "lookup":
        entity = str(args.get("entity", "")).strip().lower()
        attr = _ATTR_ALIASES.get(str(args.get("attribute", "")).strip().lower())
        if attr is None:
            return "NOT_FOUND (unknown attribute)"
        record = CITIES.get(entity) or COUNTRIES.get(entity)
        if record is None or attr not in record:
            return "NOT_FOUND"
        return str(record[attr])

    if tool == "calc":
        try:
            return str(round(_safe_calc(str(args.get("expression", ""))), 4))
        except Exception as e:  # noqa: BLE001 — surface the error as an observation
            return f"CALC_ERROR: {e}"

    if tool == "report":
        return f"FINAL={args.get('value')}"

    return f"UNKNOWN_TOOL: {tool}"


# ---- Prompts ----

_PLANNER_PROMPT = (
    "You are a planner. Given a question answerable with the tools below, write a "
    "FLAT, ORDERED plan of concrete steps that another agent will execute literally, "
    "one step at a time.\n\n"
    "Hard rules:\n"
    "- No conditionals and no placeholders. Do NOT write 'if/else', 'depending on', "
    "'or', 'either', or angle-bracket blanks like <value>. Every step names CONCRETE "
    "entities and attributes.\n"
    "- If the question contains a condition you cannot resolve yet (because you don't "
    "know an intermediate value), make ONE definite assumption and plan the concrete "
    "steps for that assumption. Commit — do not hedge.\n"
    "- Each step is a single concrete action phrased as an imperative "
    "(e.g. 'Look up the population of Berlin.', 'Report the population of Germany.').\n"
    "- The executor cannot see the original question — only your step text and the "
    "observations so far — so each step must be self-contained.\n"
    "- The FINAL step must report one concrete quantity "
    "(e.g. 'Report the population of Germany as the answer.').\n\n"
    f"{TOOL_SCHEMA}\n\n"
    'Output ONLY JSON: {"steps": ["step 1", "step 2", ...]}'
)

_REPLANNER_PROMPT = (
    "You are re-planning a task that is already partway executed. You now KNOW the "
    "intermediate values the original planner had to guess.\n\n"
    "First, evaluate any condition in the question against the observed values to "
    "decide which quantity is actually being asked for. Then write a NEW flat ordered "
    "plan for the REMAINING work: look up any value that has not been fetched yet, and "
    "make the final step report the CORRECT concrete quantity. Same hard rules as "
    "before (no conditionals, no placeholders; concrete, self-contained steps).\n\n"
    "Each step must be a FULL IMPERATIVE SENTENCE naming concrete entities — e.g. "
    "'Look up the population of Germany.', 'Report the population of Germany as the "
    "answer.' — NOT a bare tool name like 'lookup' or 'report'.\n\n"
    f"{TOOL_SCHEMA}\n\n"
    'Output ONLY JSON: {"steps": ["Look up ...", "Report ... as the answer."]}'
)

_ACT_PROMPT = (
    "You execute ONE step of a plan. You are given the step and the observations so "
    "far. Do NOT try to solve any larger task — perform exactly this step and nothing "
    "more. Choose the single tool call that performs it.\n\n"
    f"{TOOL_SCHEMA}\n\n"
    "Output ONLY the JSON for the one tool call."
)

_MONITOR_PROMPT = (
    "You are monitoring a plan as it executes. You see the original question, the "
    "observations so far, and the remaining planned steps.\n\n"
    "Evaluate any condition in the question against the observed values. Then ask: if "
    "the remaining steps are executed LITERALLY (no further thinking), will they fetch "
    "and report the exact quantity the question asks for given what we now know? If a "
    "resolved condition means the remaining steps would report the WRONG quantity — or "
    "the value needed has not been and will not be looked up — request a replan.\n\n"
    'Output ONLY JSON: {"action": "continue"} or {"action": "replan", "reason": "<why>"}'
)


# ---- JSON helpers ----

def _extract_json(raw: str) -> dict:
    """Pull the first JSON object out of an LLM response, tolerating code fences."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in: {raw[:120]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"unbalanced JSON in: {raw[:120]!r}")


def _chat(system: str, user: str) -> tuple[str, int, int]:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=TEMPERATURE,
    )
    msg = (completion.choices[0].message.content or "").strip()
    return msg, completion.usage.prompt_tokens, completion.usage.completion_tokens


# ---- Result types ----

@dataclass
class StepLog:
    text: str
    tool: str = ""
    args: dict = field(default_factory=dict)
    observation: str = ""


@dataclass
class PlanExecuteResult:
    question: str
    mode: str
    plan: list[str] = field(default_factory=list)
    steps: list[StepLog] = field(default_factory=list)
    final_answer: float | None = None
    num_replans: int = 0
    llm_calls: int = 0
    tool_calls: int = 0

    @property
    def num_steps(self) -> int:
        return len(self.steps)


# ---- Components ----

def _plan(question: str, system: str, user: str) -> tuple[list[str], int, int]:
    with langwatch.span(name="plan", type="llm") as s:
        s.update(input=user[:400])
        raw, pt, ct = _chat(system, user)
        try:
            steps = [str(x) for x in _extract_json(raw).get("steps", [])]
        except Exception:
            steps = []
        s.update(output="\n".join(steps)[:400], metrics={"prompt_tokens": pt, "completion_tokens": ct})
    return steps, pt, ct


def _act(step_text: str, observations: list[str]) -> tuple[str, dict, str, int, int]:
    """Run one step's act-LLM + tool. Returns (tool, args, observation, pt, ct)."""
    obs_block = "\n".join(observations) if observations else "(none yet)"
    user = f"Step to perform:\n{step_text}\n\nObservations so far:\n{obs_block}"
    with langwatch.span(name="act", type="llm") as s:
        s.update(input=user[:400])
        raw, pt, ct = _chat(_ACT_PROMPT, user)
        s.update(output=raw[:300], metrics={"prompt_tokens": pt, "completion_tokens": ct})
    try:
        call = _extract_json(raw)
        tool, args = str(call.get("tool", "")), dict(call.get("args", {}))
    except Exception as e:  # noqa: BLE001
        tool, args = "", {}
        return tool, args, f"PARSE_ERROR: {e}", pt, ct

    with langwatch.span(name=tool or "unknown_tool", type="tool") as s:
        s.update(input=json.dumps(args)[:300])
        observation = run_tool(tool, args)
        s.update(output=observation[:300])
    return tool, args, observation, pt, ct


def _monitor(question: str, observations: list[str], remaining: list[str]) -> tuple[bool, str, int, int]:
    """Replan-mode only. Returns (should_replan, reason, pt, ct)."""
    user = (
        f"Question:\n{question}\n\nObservations so far:\n" + "\n".join(observations)
        + "\n\nRemaining planned steps:\n" + "\n".join(remaining)
    )
    with langwatch.span(name="monitor", type="evaluation") as s:
        s.update(input=user[:400])
        raw, pt, ct = _chat(_MONITOR_PROMPT, user)
        try:
            decision = _extract_json(raw)
        except Exception:
            decision = {"action": "continue"}
        should = decision.get("action") == "replan"
        reason = str(decision.get("reason", ""))
        s.update(output=f"replan={should} {reason}"[:300], metrics={"prompt_tokens": pt, "completion_tokens": ct})
    return should, reason, pt, ct


# ---- The agent ----

@langwatch.trace(name="plan_execute")
def run(question: str, *, mode: str = PLAN_MODE) -> PlanExecuteResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {question}")

    result = PlanExecuteResult(question=question, mode=mode)

    # --- Plan once, upfront ---
    plan, pt, ct = _plan(question, _PLANNER_PROMPT, f"Question: {question}")
    result.plan = list(plan)
    result.llm_calls += 1

    queue = list(plan)  # remaining steps to execute
    observations: list[str] = []
    executed = 0

    while queue and executed < MAX_STEPS:
        # --- Monitor before executing the next step (replan mode only) ---
        if mode == "replan" and observations and result.num_replans < MAX_REPLANS:
            should_replan, _reason, _pt, _ct = _monitor(question, observations, queue)
            result.llm_calls += 1
            if should_replan:
                with langwatch.span(name="replan", type="llm") as s:
                    history = "\n".join(
                        f"- {st.text} -> {st.observation}" for st in result.steps
                    )
                    user = (
                        f"Question: {question}\n\nSteps already executed:\n{history}\n\n"
                        "Write the remaining plan."
                    )
                    s.update(input=user[:400])
                    raw, rpt, rct = _chat(_REPLANNER_PROMPT, user)
                    try:
                        new_steps = [str(x) for x in _extract_json(raw).get("steps", [])]
                    except Exception:
                        new_steps = queue
                    s.update(output="\n".join(new_steps)[:400],
                             metrics={"prompt_tokens": rpt, "completion_tokens": rct})
                result.llm_calls += 1
                result.num_replans += 1
                queue = new_steps  # replace remaining work

        if not queue:
            break

        step_text = queue.pop(0)
        executed += 1
        with langwatch.span(name=f"step_{executed}", type="agent") as s:
            tool, args, observation, pt, ct = _act(step_text, observations)
            s.update(output=f"{tool}({args}) -> {observation}"[:300])
        result.llm_calls += 1
        result.tool_calls += 1
        result.steps.append(StepLog(text=step_text, tool=tool, args=args, observation=observation))
        observations.append(f"step {executed}: {step_text} -> {observation}")

        if tool == "report":
            try:
                result.final_answer = float(args.get("value"))
            except (TypeError, ValueError):
                result.final_answer = None
            break

    # Fallback: if the plan never issued a 'report', try to read a number from the
    # last observation so the row still scores (rather than silently failing).
    if result.final_answer is None and result.steps:
        m = re.search(r"-?\d+(?:\.\d+)?", result.steps[-1].observation)
        if m:
            result.final_answer = float(m.group())

    root.update(output=f"answer={result.final_answer} | replans={result.num_replans} "
                       f"| steps={result.num_steps}")
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Look up the population of Berlin. If it is greater than 3 million, give the "
        "population of Berlin's country; otherwise give Berlin's own population. "
        "Answer in millions, one decimal."
    )
    mode = os.getenv("PLAN_MODE", "static")
    print(f"\n=== Question: {q}\n=== Mode: {mode}\n")
    res = run(q, mode=mode)
    print("Plan:")
    for i, st in enumerate(res.plan, 1):
        print(f"  {i}. {st}")
    print(f"\nReplans: {res.num_replans} | steps executed: {res.num_steps} | "
          f"llm_calls: {res.llm_calls}")
    print("\nExecution:")
    for i, st in enumerate(res.steps, 1):
        print(f"  step {i}: {st.text}")
        print(f"          -> {st.tool}({st.args}) = {st.observation}")
    print(f"\n=== Final answer: {res.final_answer}")
