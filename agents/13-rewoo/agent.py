"""
ReWOO agent — Reasoning WithOut Observation (Xu et al., 2023).

ReWOO decouples reasoning from tool execution. A *planner* writes the ENTIRE
plan upfront as a list of interdependent tool calls using #E variables as
placeholders for results it cannot see yet; a *worker* executes those calls and
fills in the variables; a *solver* reads the plan + all the evidence and writes
the final answer. The LLM is called exactly TWICE (plan + solve) no matter how
many tool calls the plan contains.

Switch via env var:
    REWOO_MODE=react   (default) — interleaved Thought -> Action -> Observation.
                                   One LLM call PER tool call; the model sees each
                                   observation before choosing the next action.
    REWOO_MODE=rewoo             — plan blind -> execute -> solve. Exactly 2 LLM
                                   calls regardless of tool-call count.

Both modes share the SAME tools and the SAME text action syntax
(`lookup[entity, attribute]`, `calc[expr]`); the ONLY difference is the
architecture — interleaved vs decoupled — which is what makes the comparison
clean. The headline contrast is LLM-call efficiency; the cost is that ReWOO
plans without ever seeing an observation, so it cannot adapt to one.

Trace tree (typed LangWatch spans):

    rewoo (workflow root)
    ├─ react:  iteration_k (agent) -> reason (llm) + tool spans, repeated
    └─ rewoo:  plan (llm) -> lookup/calc (tool) x N -> solve (llm)
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

REWOO_MODE = os.getenv("REWOO_MODE", "react")  # "react" | "rewoo"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
SERVICE_NAME = "rewoo"

os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- Fictional knowledge base (forces real tool use — nothing is in the model's memory) ----

KB: dict[str, dict[str, str]] = {
    # countries
    "mordania":  {"type": "country", "capital": "velin", "population": "30000000", "currency": "drake", "continent": "aurelia"},
    "tavros":    {"type": "country", "capital": "kesh",  "population": "12000000", "currency": "sol",   "continent": "aurelia"},
    "brightland":{"type": "country", "capital": "orne",  "population": "45000000", "currency": "mark",  "continent": "caldoria"},
    "quenia":    {"type": "country", "capital": "sable", "population": "8000000",  "currency": "lira",  "continent": "caldoria"},
    # cities
    "zelvora": {"type": "city", "country": "mordania",  "population": "1200000", "founded": "1340"},
    "pellin":  {"type": "city", "country": "tavros",    "population": "600000",  "founded": "1502"},
    "marn":    {"type": "city", "country": "brightland","population": "2100000", "founded": "1188"},
    "doverin": {"type": "city", "country": "quenia",    "population": "450000",  "founded": "1610"},
    "velin":   {"type": "city", "country": "mordania",  "population": "4000000", "founded": "900"},
    "kesh":    {"type": "city", "country": "tavros",    "population": "1500000", "founded": "1100"},
    "orne":    {"type": "city", "country": "brightland","population": "5000000", "founded": "1050"},
    "sable":   {"type": "city", "country": "quenia",    "population": "1000000", "founded": "1300"},
    # continents
    "aurelia":  {"type": "continent", "largest_country": "mordania"},
    "caldoria": {"type": "continent", "largest_country": "brightland"},
}


# ---- Tool implementations ----

def _clean(s: str) -> str:
    return s.strip().strip("'\"").strip().lower()


def lookup(args: str) -> str:
    """lookup[entity, attribute] -> the attribute's value, or UNKNOWN."""
    parts = [p for p in args.split(",")]
    if len(parts) < 2:
        return f"UNKNOWN (lookup needs 'entity, attribute'; got {args!r})"
    entity, attribute = _clean(parts[0]), _clean(parts[1])
    if entity not in KB:
        return f"UNKNOWN (no entity '{entity}')"
    if attribute not in KB[entity]:
        return f"UNKNOWN (entity '{entity}' has no attribute '{attribute}')"
    return KB[entity][attribute]


def calc(args: str) -> str:
    """calc[expression] -> arithmetic result. Only digits and + - * / ( ) allowed."""
    expr = args.strip()
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expr):
        return f"Error: only numbers and + - * / ( ) allowed; got {expr!r}"
    try:
        val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — sandboxed charset
        return str(int(val)) if float(val).is_integer() else str(val)
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


TOOLS = {"lookup": lookup, "calc": calc}

_TOOLS_DOC = (
    "TOOLS:\n"
    "- lookup[entity, attribute]  -> looks up an attribute of an entity in the knowledge base.\n"
    "    entities are cities, countries, continents. attributes include: country, capital,\n"
    "    population, currency, continent, founded, largest_country, type.\n"
    "- calc[expression]  -> evaluates an arithmetic expression, e.g. calc[1200000 + 2100000].\n"
)


# ---- Answer extraction ----

def _extract_answer(text: str) -> str:
    m = re.search(r"(?:final\s+answer|answer)\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE)
    ans = (m.group(1) if m else text).strip()
    return ans.splitlines()[0].strip().strip(".").strip() if ans else ""


def _chat(system: str, user: str) -> tuple[str, int, int]:
    c = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=TEMPERATURE,
    )
    return (c.choices[0].message.content or "").strip(), c.usage.prompt_tokens, c.usage.completion_tokens


# ---- Result type ----

@dataclass
class RewooResult:
    query: str
    mode: str
    answer: str = ""
    llm_calls: int = 0
    tool_calls: int = 0
    plan: str = ""
    trajectory: list[str] = field(default_factory=list)
    hit_iteration_limit: bool = False


# ---- ReAct mode: interleaved Thought -> Action -> Observation ----

_REACT_SYSTEM = (
    "You solve questions with a Reason+Act loop over a knowledge base.\n\n"
    + _TOOLS_DOC +
    "\nEach turn, output exactly one of:\n"
    "  Thought: <reasoning>\n  Action: lookup[...]   (or calc[...])\n"
    "or, when you know the answer:\n"
    "  Thought: <reasoning>\n  Finish[<answer>]\n\n"
    "Give ONLY the bare answer inside Finish[...] (a single name or number, no units)."
)

_ACTION_RE = re.compile(r"(lookup|calc)\s*\[(.*?)\]", flags=re.IGNORECASE | re.DOTALL)
_FINISH_RE = re.compile(r"finish\s*\[(.*?)\]", flags=re.IGNORECASE | re.DOTALL)


def _run_react(query: str, result: RewooResult) -> None:
    scratchpad = ""
    for i in range(1, MAX_ITERATIONS + 1):
        with langwatch.span(name=f"iteration_{i}", type="agent") as it:
            with langwatch.span(name="reason", type="llm") as s:
                user = f"Question: {query}\n\n{scratchpad}"
                s.update(input=user[-400:])
                text, pt, ct = _chat(_REACT_SYSTEM, user)
                s.update(output=text[:300], metrics={"prompt_tokens": pt, "completion_tokens": ct})
            result.llm_calls += 1
            result.trajectory.append(text)

            fin = _FINISH_RE.search(text)
            act = _ACTION_RE.search(text)
            # Prefer Finish only if it is not itself the action line
            if fin and (not act or fin.start() <= act.start()):
                result.answer = _extract_answer(fin.group(1))
                it.update(output=f"finish: {result.answer}")
                return
            if not act:
                result.answer = _extract_answer(text)
                it.update(output=f"no action; treating as answer: {result.answer}")
                return

            tool, args = act.group(1).lower(), act.group(2).strip()
            with langwatch.span(name=tool, type="tool") as ts:
                ts.update(input=args)
                obs = TOOLS[tool](args)
                ts.update(output=str(obs)[:300])
            result.tool_calls += 1
            scratchpad += f"Thought+Action: {tool}[{args}]\nObservation: {obs}\n"
            it.update(output=f"{tool}[{args}] -> {obs}")
    result.hit_iteration_limit = True
    result.answer = "(no answer — hit iteration limit)"


# ---- ReWOO mode: plan (blind) -> execute -> solve ----

_PLAN_SYSTEM = (
    "You are a planner. Produce a COMPLETE plan to answer the question using the tools, "
    "WITHOUT executing anything and WITHOUT seeing any results.\n\n"
    + _TOOLS_DOC +
    "\nWrite the plan as a sequence of steps. Each step is one line of the form:\n"
    "  #E1 = lookup[zelvora, country]\n"
    "  #E2 = lookup[#E1, capital]\n"
    "Use #E<k> to refer to the result of an earlier step inside later arguments. "
    "The LAST step's result must be the answer to the question. "
    "Output ONLY the #E lines, nothing else. You will NOT get to see any results, so the "
    "plan must be self-contained."
)

_PLAN_STEP_RE = re.compile(r"#E(\d+)\s*=\s*(lookup|calc)\s*\[(.*?)\]", flags=re.IGNORECASE | re.DOTALL)

_SOLVE_SYSTEM = (
    "You are given a question, a plan, and the evidence collected by executing the plan. "
    "Use the evidence to answer the question. Output ONLY the bare answer (a single name or "
    "number, no units) on a line prefixed with 'Answer:'."
)


def _run_rewoo(query: str, result: RewooResult) -> None:
    # --- plan (blind) ---
    with langwatch.span(name="plan", type="llm") as s:
        s.update(input=query)
        plan, pt, ct = _chat(_PLAN_SYSTEM, query)
        s.update(output=plan[:500], metrics={"prompt_tokens": pt, "completion_tokens": ct})
    result.llm_calls += 1
    result.plan = plan

    steps = _PLAN_STEP_RE.findall(plan)  # list of (index, tool, args)

    # --- worker: execute each step, substituting prior #E values ---
    evidence: dict[str, str] = {}
    evidence_lines: list[str] = []
    for idx, tool, args in steps:
        filled = args
        # substitute higher indices first so #E10 isn't clobbered by #E1
        for k in sorted(evidence, key=lambda x: -int(x[2:])):
            filled = filled.replace(k, evidence[k])
        with langwatch.span(name=tool.lower(), type="tool") as ts:
            ts.update(input=f"#E{idx} = {tool}[{filled}]")
            obs = TOOLS[tool.lower()](filled)
            ts.update(output=str(obs)[:300])
        result.tool_calls += 1
        evidence[f"#E{idx}"] = str(obs)
        evidence_lines.append(f"#E{idx} = {tool}[{filled}] -> {obs}")

    # --- solve ---
    solve_input = (
        f"QUESTION:\n{query}\n\nPLAN:\n{plan}\n\nEVIDENCE:\n" + "\n".join(evidence_lines)
    )
    with langwatch.span(name="solve", type="llm") as s:
        s.update(input=solve_input[:500])
        text, pt, ct = _chat(_SOLVE_SYSTEM, solve_input)
        s.update(output=text[:300], metrics={"prompt_tokens": pt, "completion_tokens": ct})
    result.llm_calls += 1
    result.answer = _extract_answer(text)
    result.trajectory = evidence_lines


# ---- The agent ----

@langwatch.trace(name="rewoo")
def run(query: str, *, mode: str = REWOO_MODE) -> RewooResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {query}")

    result = RewooResult(query=query, mode=mode)
    if mode == "react":
        _run_react(query, result)
    elif mode == "rewoo":
        _run_rewoo(query, result)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    root.update(output=f"[{mode}] answer={result.answer!r} | llm_calls={result.llm_calls} tools={result.tool_calls}")
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "What is the capital of the country that the city Zelvora is in?"
    )
    mode = os.getenv("REWOO_MODE", "react")
    print(f"\n=== Query: {q}\n=== Mode: {mode}\n")
    res = run(q, mode=mode)
    if res.mode == "rewoo":
        print(f"Plan:\n{res.plan}\n")
        print("Evidence:")
        for line in res.trajectory:
            print(f"  {line}")
    else:
        print("Trajectory:")
        for t in res.trajectory:
            print(f"  - {t.splitlines()[0][:90]}")
    print(f"\n=== answer: {res.answer!r}  |  llm calls: {res.llm_calls}  |  tool calls: {res.tool_calls}")
