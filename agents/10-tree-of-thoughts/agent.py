"""
Tree-of-Thoughts agent — solve Game of 24 by searching a tree of partial
solutions instead of committing to one linear chain.

Game of 24: given four numbers, combine them with + - * / (each number used
exactly once) to make 24. e.g. 4 9 10 13 -> (13 - 9) * (10 - 4) = 24.

Switch via env var:
    TOT_MODE=cot   (default) — one chain-of-thought pass, produce an expression.
    TOT_MODE=tot             — beam search over partial states. At each depth the
                                next states are ENUMERATED mechanically (combining
                                a pair of numbers is just arithmetic), and the LLM
                                acts as the VALUE FUNCTION: it scores each state
                                sure / likely / impossible, and the search keeps
                                the top-BEAM and prunes the rest. Repeat until one
                                number remains.

The variable under test is linear reasoning vs deliberate search. ToT is the
canonical "search beats a single chain" pattern, and Game of 24 is its headline
benchmark — a place where a single CoT pass tends to fail because one early
wrong combination dooms the whole chain with no way back.

Design choice (faithful to the ToT paper for Game of 24): move *generation* is
mechanical — the leverage of ToT here is entirely the **evaluator** that decides
which branches to keep. So this agent enumerates all legal moves in code and
spends its LLM calls on evaluation. That makes the experiment clean: ToT's
performance is bounded by the quality of its state evaluator, nothing else.

The catch ToT introduces: if that evaluator can't tell a reachable state from a
dead end, the search prunes the right answer or wastes its budget on doomed
branches. So this agent brute-forces, for every state the evaluator judges,
whether 24 is actually still reachable — turning "is ToT's pruning grounded?"
into a measured number (see evals.py: evaluator_accuracy).

Trace tree (typed LangWatch spans):

    tree_of_thoughts (workflow root)
    ├─ cot:  solve (llm)
    └─ tot:
       ├─ evaluate_d1 (evaluation)   score every 3-number state, keep top-BEAM
       ├─ evaluate_d2 (evaluation)   score every 2-number state from the beam
       └─ (1-number leaf states are checked for 24 mechanically — no LLM)
"""

from __future__ import annotations

import ast
import os
import re
import operator
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration ----

TOT_MODE = os.getenv("TOT_MODE", "cot")  # "cot" | "tot"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
BEAM = int(os.getenv("TOT_BEAM", "5"))                 # states kept per depth
MAX_EVAL_PER_DEPTH = int(os.getenv("TOT_MAX_EVAL", "64"))  # safety cap; high enough to
#   evaluate every candidate for 4-number Game of 24 (no biased truncation)
SERVICE_NAME = "tree-of-thoughts"

os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()

TOL = 1e-6


# ---- Game of 24 mechanics (deterministic) ----

def _combine(a: float, b: float):
    """All (value, op) obtainable from a, b with a single operation."""
    out = [(a + b, "+"), (a * b, "*"), (a - b, "-"), (b - a, "-")]
    if abs(b) > TOL:
        out.append((a / b, "/"))
    if abs(a) > TOL:
        out.append((b / a, "/"))
    return out


def _combine_expr(av, ae, bv, be):
    """All (value, expression) obtainable from two (value, expr) entries."""
    out = [(av + bv, f"({ae} + {be})"), (av * bv, f"({ae} * {be})"),
           (av - bv, f"({ae} - {be})"), (bv - av, f"({be} - {ae})")]
    if abs(bv) > TOL:
        out.append((av / bv, f"({ae} / {be})"))
    if abs(av) > TOL:
        out.append((bv / av, f"({be} / {ae})"))
    return out


@lru_cache(maxsize=None)
def solvable(nums: tuple) -> bool:
    """Brute-force oracle: can these numbers reach 24? (order-independent)

    Values are carried at full float precision — rounding intermediates would
    destroy non-terminating fractions like 8/3, which are exactly what the
    hardest Game of 24 puzzles (e.g. 3 3 8 8 = 8/(3-8/3)) depend on.
    """
    if len(nums) == 1:
        return abs(nums[0] - 24) < TOL
    n = len(nums)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rest = tuple(nums[k] for k in range(n) if k != i and k != j)
            for val, _op in _combine(nums[i], nums[j]):
                if solvable(tuple(sorted(rest + (val,)))):
                    return True
    return False


def _expand(state):
    """All distinct child states (one fewer number) reachable from `state`.

    A state is a list of (value, expression) pairs. Dedup by rounded value
    multiset so we don't evaluate the same set of numbers twice.
    """
    seen, out = set(), []
    n = len(state)
    for i in range(n):
        for j in range(i + 1, n):
            rest = [p for k, p in enumerate(state) if k != i and k != j]
            (av, ae), (bv, be) = state[i], state[j]
            for val, expr in _combine_expr(av, ae, bv, be):
                child = rest + [(val, expr)]
                key = tuple(sorted(round(v, 6) for v, _ in child))
                if key in seen:
                    continue
                seen.add(key)
                out.append(child)
    return out


# ---- expression evaluation (for grading any final answer string) ----

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv, ast.USub: operator.neg}


def _eval_expr(expr: str) -> float:
    def _n(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_n(node.left), _n(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_n(node.operand))
        raise ValueError(f"bad expr: {expr!r}")
    return float(_n(ast.parse(expr, mode="eval").body))


def numbers_in(expr: str) -> list:
    return [float(t) for t in re.findall(r"\d+(?:\.\d+)?", expr)]


# ---- LLM helper ----

def _chat(system: str, user: str) -> tuple[str, int, int]:
    c = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=TEMPERATURE,
    )
    return (c.choices[0].message.content or "").strip(), c.usage.prompt_tokens, c.usage.completion_tokens


# ---- Prompts ----

_COT_PROMPT = (
    "You play the Game of 24: use each of the given numbers exactly once, with "
    "the operations + - * / and parentheses, to make 24. Think step by step, "
    "then end with a single line:\nFinal answer: <expression>\n"
    "The expression must use each given number exactly once, e.g. "
    "'Final answer: (13 - 9) * (10 - 4)'."
)

_EVALUATE_PROMPT = (
    "Game of 24. Given a set of numbers, judge whether 24 can still be reached by "
    "combining ALL of them with + - * / (each used once, fractions allowed).\n"
    "First reason in ONE short sentence (try a combination or two). Then, on the "
    "FINAL line, output exactly one word:\n"
    "  sure       — clearly reachable\n"
    "  likely     — plausibly reachable\n"
    "  impossible — cannot reach 24\n"
)

_SCORE = {"sure": 2.0, "likely": 1.0, "impossible": 0.0}


# ---- Result types ----

@dataclass
class EvalJudgement:
    numbers: tuple
    verdict: str          # sure | likely | impossible
    kept: bool            # verdict != impossible
    truly_solvable: bool  # brute-force ground truth


@dataclass
class ToTResult:
    numbers: list
    mode: str
    final_expression: str = ""
    llm_calls: int = 0
    states_explored: int = 0
    judgements: list = field(default_factory=list)


# ---- CoT mode ----

def _solve_cot(numbers, result: ToTResult) -> None:
    user = "Numbers: " + ", ".join(str(int(n)) if n == int(n) else str(n) for n in numbers)
    with langwatch.span(name="solve", type="llm") as s:
        s.update(input=user)
        out, pt, ct = _chat(_COT_PROMPT, user)
        s.update(output=out[:500], metrics={"prompt_tokens": pt, "completion_tokens": ct})
    result.llm_calls = 1
    m = list(re.finditer(r"final answer\s*:?\s*(.+)", out, flags=re.IGNORECASE))
    result.final_expression = m[-1].group(1).strip().rstrip(".") if m else ""


# ---- ToT mode: beam search (mechanical expansion + LLM evaluator) ----

def _evaluate(candidates, depth: int, result: ToTResult):
    scored = []
    with langwatch.span(name=f"evaluate_d{depth}", type="evaluation") as s:
        verdicts = []
        for child in candidates:
            full = tuple(sorted(v for v, _ in child))   # full precision for oracle
            disp = tuple(round(v, 4) for v in full)
            user = "Numbers: " + ", ".join(str(v) for v in disp)
            out, _pt, _ct = _chat(_EVALUATE_PROMPT, user)
            result.llm_calls += 1
            # take the verdict that appears LAST (the final-line answer, after any reasoning)
            low = out.lower()
            idx = {w: low.rfind(w) for w in ("sure", "likely", "impossible")}
            verdict = max((w for w in idx if idx[w] >= 0), key=lambda w: idx[w], default="likely")
            kept = verdict != "impossible"
            result.judgements.append(EvalJudgement(disp, verdict, kept, solvable(full)))
            verdicts.append(f"{disp}->{verdict}")
            scored.append((_SCORE[verdict], child))
        s.update(input=f"{len(candidates)} states at depth {depth}",
                 output=" | ".join(verdicts)[:450])
    return scored


def _solve_tot(numbers, result: ToTResult) -> None:
    beam = [[(float(n), str(int(n)) if n == int(n) else str(n)) for n in numbers]]
    depth = 0
    while beam and len(beam[0]) > 1:
        depth += 1
        # Enumerate all distinct child states across the current beam.
        seen, candidates = set(), []
        for state in beam:
            for child in _expand(state):
                key = tuple(sorted(round(v, 6) for v, _ in child))
                if key not in seen:
                    seen.add(key)
                    candidates.append(child)
        result.states_explored += len(candidates)

        # Leaf depth: one number left — check for 24 mechanically, no LLM.
        if candidates and len(candidates[0]) == 1:
            for child in candidates:
                if abs(child[0][0] - 24) < TOL:
                    result.final_expression = child[0][1]
                    return
            return

        candidates = candidates[:MAX_EVAL_PER_DEPTH]  # budget cap
        scored = _evaluate(candidates, depth, result)
        scored = [(sc, st) for sc, st in scored if sc > 0] or scored
        scored.sort(key=lambda x: x[0], reverse=True)
        beam = [st for _sc, st in scored[:BEAM]]


# ---- The agent ----

@langwatch.trace(name="tree_of_thoughts")
def run(numbers, *, mode: str = TOT_MODE) -> ToTResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] Game of 24: {numbers}")

    result = ToTResult(numbers=list(numbers), mode=mode)
    if mode == "cot":
        _solve_cot(numbers, result)
    elif mode == "tot":
        _solve_tot(numbers, result)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    root.update(output=f"expr={result.final_expression!r} | calls={result.llm_calls}")
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    nums = [float(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [4, 9, 10, 13]
    mode = os.getenv("TOT_MODE", "cot")
    print(f"\n=== Game of 24: {nums}  (solvable={solvable(tuple(sorted(nums)))})")
    print(f"=== Mode: {mode}\n")
    res = run(nums, mode=mode)
    print(f"Final expression: {res.final_expression!r}")
    print(f"LLM calls: {res.llm_calls} | states explored: {res.states_explored}")
    if res.final_expression:
        try:
            print(f"Evaluates to: {_eval_expr(res.final_expression)}")
        except Exception as e:  # noqa: BLE001
            print(f"(could not evaluate: {e})")
    if res.judgements:
        correct = sum(1 for j in res.judgements if j.kept == j.truly_solvable)
        print(f"Evaluator judgements: {correct}/{len(res.judgements)} matched the oracle")
