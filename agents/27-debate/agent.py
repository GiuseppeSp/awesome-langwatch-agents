"""
Multi-agent debate — do N agents that SEE each other's answers and argue converge on
a better answer than a single agent, or than just voting N independent answers?

Switch via env var:
    MA_MODE=single           (default) — one agent answers once. The baseline.
    MA_MODE=debate                     — N agents answer independently (round 1), then
                                         for R rounds each sees the others' answers and
                                         reasoning and may revise; the final answer is a
                                         majority vote of the last round.
    MA_MODE=self_consistency           — N independent answers, majority vote, NO
                                         interaction (round 1 only). For standalone use;
                                         run_eval derives it from debate's round 1 so the
                                         samples are shared and the comparison is paired.

The whole experiment hinges on isolating the INTERACTION: debate's round 1 IS a set of
independent samples (self-consistency), so majority(round 1) vs majority(final round)
measures exactly what the arguing added — including its risk, CONFORMITY: a correct
agent talked out of the right answer by confident, wrong peers.

Debate needs the agents to actually differ, so they sample at DEBATE_TEMP > 0 (at temp 0
they would be identical and there is nothing to debate — that is #9's lesson). The single
baseline answers at temperature 0 (its most-likely single shot).

Trace tree (typed LangWatch spans):

    multi_agent_debate (workflow root)
    ├─ single: answer (llm)
    └─ debate: round_1 (span){ agent (llm) xN } -> round_2 (span){ agent xN } -> ... -> vote (span)
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration ----

MA_MODE = os.getenv("MA_MODE", "single")  # single | debate | self_consistency
MODEL = os.getenv("MODEL", "gpt-4o-mini")
N_AGENTS = int(os.getenv("N_AGENTS", "3"))
DEBATE_ROUNDS = int(os.getenv("DEBATE_ROUNDS", "2"))  # revision rounds AFTER the initial one
DEBATE_TEMP = float(os.getenv("DEBATE_TEMP", "0.7"))  # agents must diverge to debate

os.environ.setdefault("OTEL_SERVICE_NAME", "multi-agent-debate")
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- LLM helper ----

def _ask(question: str, temperature: float, peer_answers: list[dict] | None = None) -> dict:
    system = (
        "You solve the problem and give a single concrete final answer.\n"
        'Respond with ONE JSON object: {"reasoning": "<brief>", "answer": "<the final answer, '
        'as short as possible — just the number or word>"}.'
    )
    user = f"Problem: {question}"
    if peer_answers:
        peers = "\n".join(f'- Agent {i+1} answered "{p["answer"]}" because: {p["reasoning"]}'
                          for i, p in enumerate(peer_answers))
        user += (
            f"\n\nOther agents gave these answers and reasoning:\n{peers}\n\n"
            "Consider their reasoning carefully. If they are right, update your answer; "
            "if you are right, keep it. Give your updated reasoning and final answer."
        )
    c = client.chat.completions.create(
        model=MODEL, temperature=temperature,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    raw = (c.choices[0].message.content or "").strip()
    try:
        d = json.loads(raw)
        return {"reasoning": str(d.get("reasoning", "")), "answer": str(d.get("answer", "")).strip()}
    except json.JSONDecodeError:
        return {"reasoning": "", "answer": raw.strip()}


# ---- Vote normalization (shared with evals) ----

def norm_answer(a: str) -> str:
    """Canonicalize an answer for voting/scoring: the first number (int or decimal),
    numerically canonicalized (so 40, 40.0 and '$40' all match); else casefold text."""
    m = re.search(r"-?\d+(?:\.\d+)?", a.replace(",", ""))
    if m:
        v = float(m.group(0))
        return str(int(v)) if v == int(v) else str(v)
    return a.strip().casefold().strip('."\'')


def majority(answers: list[str]) -> str:
    norms = [norm_answer(a) for a in answers if a.strip()]
    if not norms:
        return ""
    return Counter(norms).most_common(1)[0][0]


# ---- Result type ----

@dataclass
class DebateResult:
    mode: str
    single_answer: str = ""
    rounds: list[list[dict]] = field(default_factory=list)  # rounds[r][i] = {reasoning, answer}
    llm_calls: int = 0

    def round_answers(self, r: int) -> list[str]:
        return [a["answer"] for a in self.rounds[r]]

    @property
    def sc_answer(self) -> str:              # majority of round 1 (independent samples)
        return majority(self.round_answers(0)) if self.rounds else ""

    @property
    def debate_answer(self) -> str:          # majority of the final round
        return majority(self.round_answers(-1)) if self.rounds else ""


# ---- The agent ----

@langwatch.trace(name="multi_agent_debate")
def run(question: str, *, mode: str = MA_MODE) -> DebateResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {question}")
    result = DebateResult(mode=mode)

    if mode == "single":
        with langwatch.span(name="answer", type="llm") as s:
            s.update(input=question)
            ans = _ask(question, temperature=0.0)
            s.update(output=ans["answer"])
        result.single_answer = ans["answer"]
        result.llm_calls = 1
        root.update(output=f"[single] {result.single_answer}")
        return result

    if mode not in ("debate", "self_consistency"):
        raise ValueError(f"unknown mode: {mode!r}")

    total_rounds = 1 if mode == "self_consistency" else DEBATE_ROUNDS + 1
    prev: list[dict] | None = None
    for r in range(total_rounds):
        label = "round_1_initial" if r == 0 else f"round_{r+1}_revise"
        with langwatch.span(name=label, type="span") as rs:
            rs.update(input=f"{N_AGENTS} agents")
            this_round = []
            for i in range(N_AGENTS):
                with langwatch.span(name=f"agent_{i+1}", type="agent") as a:
                    a.update(input=f"round {r+1}")
                    # each agent sees the OTHER agents' previous answers (not its own)
                    peers = None if prev is None else [p for j, p in enumerate(prev) if j != i]
                    ans = _ask(question, temperature=DEBATE_TEMP, peer_answers=peers)
                    a.update(output=ans["answer"])
                this_round.append(ans)
                result.llm_calls += 1
            rs.update(output="answers: " + ", ".join(a["answer"] for a in this_round))
        result.rounds.append(this_round)
        prev = this_round

    with langwatch.span(name="vote", type="span") as s:
        s.update(input="final round: " + ", ".join(result.round_answers(-1)),
                 output=f"majority = {result.debate_answer}")

    root.update(output=f"[{mode}] sc={result.sc_answer} debate={result.debate_answer}")
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost, in cents?"
    mode = os.getenv("MA_MODE", "single")
    print(f"\n=== {q}\n=== Mode: {mode}\n")
    res = run(q, mode=mode)
    if mode == "single":
        print(f"Answer: {res.single_answer}")
    else:
        for r, rnd in enumerate(res.rounds):
            print(f"  round {r+1}: {[a['answer'] for a in rnd]}")
        print(f"\nself-consistency (round 1 vote): {res.sc_answer}")
        print(f"debate (final round vote):       {res.debate_answer}")
    print(f"llm calls: {res.llm_calls}")
