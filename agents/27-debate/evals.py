"""
Evaluators for multi-agent debate.

Every question has one exact answer, so scoring is programmatic — and every answer
(single, each agent, each vote) goes through the SAME normalizer from agent.py, so a
difference between modes can never be a parser artifact (#20's lesson).

Per question we compare three answers, two of them PAIRED (SC and debate come from the
same debate run — SC is the round-1 vote, debate the final-round vote):

    correct              — does an answer match gold? Applied to single / sc / debate.

    conformity_harm      — SC (round-1 vote) was CORRECT but debate (final vote) is WRONG:
                           the arguing moved the consensus OFF a correct answer. Debate's
                           signature failure mode.
    correction_benefit   — SC was WRONG but debate is CORRECT: the arguing fixed it.

    agent_flips          — at the individual level, how many agents went right->wrong
                           (talked out of it) vs wrong->right across the rounds.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent import norm_answer


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float
    reason: str = ""


def correct(answer: str, gold: str) -> EvalResult:
    ok = norm_answer(answer) == norm_answer(gold)
    return EvalResult("correct", ok, 1.0 if ok else 0.0,
                      f"answer={answer!r} gold={gold!r}" + ("" if ok else "  <-- WRONG"))


def debate_delta(sc_answer: str, debate_answer: str, gold: str) -> EvalResult:
    """Paired SC -> debate outcome for one question."""
    sc_ok = norm_answer(sc_answer) == norm_answer(gold)
    db_ok = norm_answer(debate_answer) == norm_answer(gold)
    if sc_ok and not db_ok:
        return EvalResult("debate_delta", False, 0.0, "CONFORMITY HARM (SC right -> debate wrong)")
    if not sc_ok and db_ok:
        return EvalResult("debate_delta", True, 1.0, "CORRECTION (SC wrong -> debate right)")
    return EvalResult("debate_delta", True, 0.5, "no change")


def agent_flips(rounds: list[list[dict]], gold: str) -> dict:
    """Right->wrong and wrong->right at the individual-agent level, round 1 -> final."""
    if len(rounds) < 2:
        return {"right_to_wrong": 0, "wrong_to_right": 0}
    g = norm_answer(gold)
    first, last = rounds[0], rounds[-1]
    r2w = w2r = 0
    for a0, aN in zip(first, last):
        c0 = norm_answer(a0["answer"]) == g
        cN = norm_answer(aN["answer"]) == g
        if c0 and not cN:
            r2w += 1
        elif not c0 and cN:
            w2r += 1
    return {"right_to_wrong": r2w, "wrong_to_right": w2r}


def initial_agreement(rounds: list[list[dict]]) -> bool:
    """Did all round-1 agents already agree? (debate can only matter where they didn't.)"""
    if not rounds:
        return False
    norms = {norm_answer(a["answer"]) for a in rounds[0]}
    return len(norms) == 1
