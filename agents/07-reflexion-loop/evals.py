"""
Evaluators for the reflexion-loop agent.

Three scorers — all programmatic, because the dataset's constraints are
mechanically verifiable. No LLM judges needed; the comparison is sharper
without the judge-noise that earlier agents had to manage:

    constraint_satisfaction  — Did the FINAL output satisfy the constraint?
                                Pass/fail. Applies to both modes.
    retry_lift               — Did the retry loop help or hurt? Per row:
                                +1 if first attempt failed but final passed,
                                 0 if no net change, -1 if first passed but
                                final failed (the perverse / H3 case).
                                Reflexion-only.
    self_critique_accuracy   — When the agent said 'PASS' on its own answer,
                                was it actually right? When it said 'FAIL'
                                and listed issues, were there real issues?
                                Fraction of correct self-verdicts. Reflexion-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float
    reason: str = ""


# ---- The verifier (shared by constraint_satisfaction and self_critique_accuracy) ----

def verify_constraint(
    output: str, constraint_type: str, primary: str, secondary: str = ""
) -> tuple[bool, str]:
    """
    Mechanically check whether `output` satisfies the constraint.

    Returns (passed: bool, reason: str). reason explains the verdict either way.
    """
    out = (output or "").strip()
    if not out:
        return False, "empty output"

    if constraint_type == "word_count":
        n = int(primary)
        words = out.split()
        actual = len(words)
        ok = actual == n
        return ok, f"expected {n} words, got {actual}"

    if constraint_type == "list_length_letter":
        n = int(primary)
        letter = secondary.upper()
        # Parse list items: strip numbering / bullets, take non-empty lines
        items = []
        for line in out.split("\n"):
            cleaned = line.strip().lstrip("0123456789.- )*•").strip()
            if cleaned:
                items.append(cleaned)
        wrong_letter = [i for i in items if not i.upper().startswith(letter)]
        ok = len(items) == n and not wrong_letter
        return ok, (
            f"expected {n} items each starting with '{letter}'; "
            f"got {len(items)} items; "
            f"wrong-letter items: {wrong_letter[:3]}"
        )

    if constraint_type == "letter_exclusion":
        letter = primary.lower()
        present = letter in out.lower()
        return (
            not present,
            f"output contains letter '{letter}'" if present else f"no '{letter}' present",
        )

    if constraint_type == "alliteration":
        letter = primary.upper()
        words = re.findall(r"[A-Za-z']+", out)
        wrong = [w for w in words if not w.upper().startswith(letter)]
        ok = len(wrong) == 0 and len(words) > 0
        return ok, (
            f"non-alliterative words: {wrong[:5]}" if wrong else f"all words start with '{letter}'"
        )

    return False, f"unknown constraint_type: {constraint_type!r}"


# ---- Evaluator 1: constraint_satisfaction (programmatic, both modes) ----

def constraint_satisfaction(
    final_output: str, constraint_type: str, primary: str, secondary: str = ""
) -> EvalResult:
    ok, reason = verify_constraint(final_output, constraint_type, primary, secondary)
    return EvalResult(
        name="constraint_satisfaction",
        passed=ok,
        score=1.0 if ok else 0.0,
        reason=reason,
    )


# ---- Evaluator 2: retry_lift (programmatic, reflexion-only) ----

def retry_lift(
    first_output: str,
    final_output: str,
    constraint_type: str,
    primary: str,
    secondary: str = "",
) -> EvalResult:
    """
    -1 if retry made things worse, +1 if retry fixed an initial failure, 0 otherwise.

    Normalized to [0, 1] for averaging: 0.0 (hurt) / 0.5 (neutral) / 1.0 (helped).
    Pass = lift >= 0 (retry didn't hurt).
    """
    first_ok, _ = verify_constraint(first_output, constraint_type, primary, secondary)
    final_ok, _ = verify_constraint(final_output, constraint_type, primary, secondary)

    if first_ok and final_ok:
        lift, label = 0, "no-op (first attempt already passed; retry kept passing)"
    elif first_ok and not final_ok:
        lift, label = -1, "HURT (first attempt passed; retry broke it — perverse case)"
    elif not first_ok and final_ok:
        lift, label = 1, "HELPED (first attempt failed; retry fixed it)"
    else:
        lift, label = 0, "no-op (first attempt failed; retry still failed)"

    normalized = (lift + 1) / 2  # -1 → 0.0, 0 → 0.5, 1 → 1.0
    return EvalResult(
        name="retry_lift",
        passed=lift >= 0,
        score=normalized,
        reason=f"raw_lift={lift:+d} — {label}",
    )


# ---- Evaluator 3: self_critique_accuracy (programmatic, reflexion-only) ----

def self_critique_accuracy(
    attempts: list,  # list[Attempt] from agent.py — duck-typed: needs .output and .self_says_pass
    constraint_type: str,
    primary: str,
    secondary: str = "",
) -> EvalResult:
    """
    For each attempt the agent self-evaluated, did its verdict match the
    programmatic truth? Returns fraction-correct over all attempts.
    """
    if not attempts:
        return EvalResult(
            name="self_critique_accuracy",
            passed=False,
            score=0.0,
            reason="no attempts",
        )

    matches = 0
    mismatches: list[str] = []
    for i, a in enumerate(attempts, 1):
        programmatic_pass, _ = verify_constraint(
            a.output, constraint_type, primary, secondary
        )
        agent_says_pass = bool(getattr(a, "self_says_pass", False))
        if agent_says_pass == programmatic_pass:
            matches += 1
        else:
            mismatches.append(
                f"attempt {i}: agent said {'PASS' if agent_says_pass else 'FAIL'}, "
                f"actually {'PASS' if programmatic_pass else 'FAIL'}"
            )

    score = matches / len(attempts)
    return EvalResult(
        name="self_critique_accuracy",
        passed=score >= 0.7,
        score=score,
        reason=(
            f"{matches}/{len(attempts)} verdicts correct"
            + (f"; mismatches: {mismatches}" if mismatches else "")
        ),
    )
