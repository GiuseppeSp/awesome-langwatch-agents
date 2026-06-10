"""
Evaluators for the multi-agent pipeline.

Three scorers tuned for the fact-checked vs basic comparison:

    fact_coverage         — Programmatic. What fraction of the expected facts
                            (from the dataset row) are mentioned in the answer?
    hallucination_judge   — LLM-as-judge. Does the answer contain any claims
                            NOT supported by the dataset's labeled facts?
    answer_quality        — LLM-as-judge. Overall well-written and useful?
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from openai import OpenAI

import langwatch

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
_client = OpenAI()


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float
    reason: str = ""


# ---- Evaluator 1: fact_coverage (programmatic, lenient match) ----

def fact_coverage(answer: str, expected_facts: list[str]) -> EvalResult:
    """
    For each expected fact, check whether the answer contains either the
    literal phrase OR each space-separated token (case-insensitive). Returns
    the fraction matched. Passes at >= 0.7 — strict enough to catch obvious
    omissions, lenient enough to tolerate paraphrasing.
    """
    answer_lc = answer.lower()
    hits = []
    misses = []
    for fact in expected_facts:
        fact_lc = fact.lower().strip()
        if not fact_lc:
            continue
        # Lenient: exact phrase OR every token (for multi-word facts like
        # "July 20 1969", we accept the date even if the model writes "July
        # 20, 1969" with a comma).
        if fact_lc in answer_lc:
            hits.append(fact)
            continue
        tokens = [t for t in re.split(r"\s+", fact_lc) if t]
        if all(t in answer_lc for t in tokens):
            hits.append(fact)
        else:
            misses.append(fact)
    total = len(expected_facts)
    score = len(hits) / total if total else 0.0
    passed = score >= 0.7
    return EvalResult(
        name="fact_coverage",
        passed=passed,
        score=score,
        reason=(
            f"Hit {len(hits)}/{total}; missed: {misses}" if misses else "All facts covered"
        ),
    )


# ---- Evaluator 2: hallucination_judge (LLM-as-judge) ----

_HALLUCINATION_RUBRIC = """\
You are checking whether an assistant's answer contains hallucinations. You will
get the user's question, a list of LABELED FACTS (the dataset's ground truth),
and the assistant's answer. Determine whether the answer makes any factual
claim that is NOT supported by the labeled facts (or by widely-known general
knowledge).

Reply with ONLY a single integer 1-5 followed by a colon and a one-sentence reason.

5 — No hallucinations. Every specific claim aligns with the labeled facts.
4 — Minor unsupported detail (e.g. a phrasing choice) but nothing factually wrong.
3 — One unsupported specific claim that could be wrong.
2 — Multiple unsupported specific claims, OR one clearly-wrong fact.
1 — Multiple wrong facts or a confidently fabricated quotation/statistic.

Focus on dates, names, numbers, and places. Vague paraphrases are fine; invented
specifics are not.
"""


def hallucination_judge(query: str, expected_facts: list[str], answer: str) -> EvalResult:
    user = (
        f"Question: {query}\n\n"
        f"Labeled facts:\n" + "\n".join(f"- {f}" for f in expected_facts) + "\n\n"
        f"Assistant's answer:\n{answer}"
    )
    with langwatch.span(name="judge_hallucination", type="evaluation") as s:
        s.update(input=query)
        completion = _client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _HALLUCINATION_RUBRIC},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        raw = (completion.choices[0].message.content or "").strip()
        s.update(
            output=raw,
            metrics={
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
            },
        )
    return _parse_rubric_result("hallucination_judge", raw)


# ---- Evaluator 3: answer_quality (LLM-as-judge) ----

_QUALITY_RUBRIC = """\
You are scoring the overall quality of a multi-agent pipeline's answer to a
user question. Score 1-5 based on whether the answer is useful, well-structured,
and naturally written — independent of whether it's factually correct (that's a
different eval).

Reply with ONLY a single integer 1-5 followed by a colon and a one-sentence reason.

5 — Clear, concise, complete, naturally phrased.
4 — Good answer with minor stiffness or padding.
3 — Workable but awkward, vague, or missing structure.
2 — Hard to read or barely answers the question.
1 — Off-topic, incoherent, or empty.
"""


def answer_quality(query: str, answer: str) -> EvalResult:
    user = f"Question: {query}\n\nAnswer:\n{answer}"
    with langwatch.span(name="judge_quality", type="evaluation") as s:
        s.update(input=query)
        completion = _client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _QUALITY_RUBRIC},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        raw = (completion.choices[0].message.content or "").strip()
        s.update(
            output=raw,
            metrics={
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
            },
        )
    return _parse_rubric_result("answer_quality", raw)


# ---- Shared rubric parser ----

def _parse_rubric_result(name: str, raw: str) -> EvalResult:
    head, _, tail = raw.partition(":")
    try:
        score_int = int(head.strip())
    except ValueError:
        return EvalResult(name=name, passed=False, score=0.0, reason=f"Unparseable: {raw}")
    score_int = max(1, min(5, score_int))
    normalized = (score_int - 1) / 4
    return EvalResult(
        name=name,
        passed=score_int >= 4,
        score=normalized,
        reason=f"score={score_int}/5 — {tail.strip() or raw}",
    )
