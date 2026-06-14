"""
Chain-of-Thought agent — does making the model "think" before it answers help,
and is sampling many thoughts worth the cost?

Switch via env var:
    COT_MODE=direct           (default) — answer immediately, no reasoning.
    COT_MODE=cot                        — "think step by step", then answer.
    COT_MODE=self_consistency           — sample N CoT paths at higher
                                          temperature and majority-vote the
                                          final answers.

The variable under test is a single dial: how much compute the model spends
"thinking" before committing to an answer. direct spends none, cot spends one
reasoning pass, self_consistency spends N.

The PM question underneath isn't "does CoT help?" — it's "**does each added
increment of compute pay for itself?**" CoT is ~the same cost as direct;
self-consistency is N× the cost. If the extra samples rarely change the answer,
you are paying 5× for nothing. And worse: majority voting measures *agreement*,
not *correctness* — a model that is confidently wrong votes wrong 5/5.

Trace tree (one typed LangWatch span per node):

    chain_of_thought (workflow root)
    ├─ answer            (llm)    [direct / cot: a single pass]
    │
    └─ self_consistency:
       ├─ sample_1       (llm)
       ├─ sample_2       (llm)
       ├─ ...
       ├─ sample_N       (llm)
       └─ majority_vote  (span)   the aggregation + the winning answer
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration ----

COT_MODE = os.getenv("COT_MODE", "direct")  # "direct" | "cot" | "self_consistency"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))        # direct / cot
SC_SAMPLES = int(os.getenv("SC_SAMPLES", "5"))            # self_consistency
SC_TEMPERATURE = float(os.getenv("SC_TEMPERATURE", "0.8"))
SERVICE_NAME = "chain-of-thought"

os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- Prompts ----
#
# Both modes end with "Final answer: X" so the extractor is identical across
# modes — the only difference is whether reasoning is allowed in between.

_DIRECT_PROMPT = (
    "Answer the question. Do NOT explain or show any working. Respond with a "
    "single line in the form:\nFinal answer: <answer>"
)

_COT_PROMPT = (
    "Think step by step. Work through the problem explicitly, then end your "
    "response with a single line in the form:\nFinal answer: <answer>"
)


# ---- Answer extraction + canonicalization ----

def extract_final(output: str) -> str:
    """Pull the answer out of a model response, tolerating reasoning before it."""
    text = (output or "").strip()
    matches = list(re.finditer(r"final answer\s*:?\s*(.+)", text, flags=re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip().rstrip(".")
    # Fallback: last non-empty line.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def canonical(answer: str) -> str:
    """
    Normalize an answer to a vote key, WITHOUT knowing the expected type.

    Numbers (with $, %, commas, words stripped) collapse to a rounded float key
    so '0.05', '$0.05', '5 cents'->'5'... no — keep it simple: numeric tokens
    become a float key; everything else becomes lowercased alphanumerics.
    """
    raw = (answer or "").strip().lower()
    cleaned = raw.replace("$", "").replace(",", "").replace("%", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if m:
        try:
            return f"num:{round(float(m.group()), 4)}"
        except ValueError:
            pass
    return "txt:" + re.sub(r"[^a-z0-9]+", "", raw)


# ---- LLM call ----

def _ask(system: str, question: str, temperature: float) -> tuple[str, int, int]:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
    )
    msg = (completion.choices[0].message.content or "").strip()
    return msg, completion.usage.prompt_tokens, completion.usage.completion_tokens


# ---- Result type ----

@dataclass
class CoTResult:
    question: str
    mode: str
    final_answer: str = ""          # extracted answer text
    reasoning: str = ""             # full output (cot), empty-ish for direct
    samples: list[str] = field(default_factory=list)   # extracted answers (SC)
    vote_counts: dict = field(default_factory=dict)     # canonical key -> count
    llm_calls: int = 0

    @property
    def vote_agreement(self) -> float:
        """Fraction of SC samples that agreed with the winning answer (1.0 = unanimous)."""
        if not self.vote_counts:
            return 1.0
        return max(self.vote_counts.values()) / sum(self.vote_counts.values())


# ---- The agent ----

@langwatch.trace(name="chain_of_thought")
def run(question: str, *, mode: str = COT_MODE) -> CoTResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {question}")

    result = CoTResult(question=question, mode=mode)

    if mode in ("direct", "cot"):
        system = _DIRECT_PROMPT if mode == "direct" else _COT_PROMPT
        with langwatch.span(name="answer", type="llm") as s:
            s.update(input=question)
            out, pt, ct = _ask(system, question, TEMPERATURE)
            s.update(output=out[:500], metrics={"prompt_tokens": pt, "completion_tokens": ct})
        result.llm_calls = 1
        result.reasoning = out if mode == "cot" else ""
        result.final_answer = extract_final(out)

    elif mode == "self_consistency":
        extracted: list[str] = []
        for i in range(1, SC_SAMPLES + 1):
            with langwatch.span(name=f"sample_{i}", type="llm") as s:
                s.update(input=question)
                out, pt, ct = _ask(_COT_PROMPT, question, SC_TEMPERATURE)
                ans = extract_final(out)
                s.update(output=f"{out[:400]}\n--> {ans}",
                         metrics={"prompt_tokens": pt, "completion_tokens": ct})
            extracted.append(ans)
        result.llm_calls = SC_SAMPLES
        result.samples = extracted

        # Majority vote over canonical keys; map the winner back to a raw answer.
        counts = Counter(canonical(a) for a in extracted)
        result.vote_counts = dict(counts)
        winner_key, _ = counts.most_common(1)[0]
        winner_answer = next(a for a in extracted if canonical(a) == winner_key)
        result.final_answer = winner_answer

        with langwatch.span(name="majority_vote", type="span") as s:
            s.update(
                input=" | ".join(extracted),
                output=f"winner={winner_answer} (agreement={result.vote_agreement:.0%}, "
                       f"counts={dict(counts)})",
            )

    else:
        raise ValueError(f"unknown mode: {mode!r}")

    root.update(output=f"answer={result.final_answer} | calls={result.llm_calls}")
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "A bat and a ball cost 1.10 dollars in total. The bat costs 1.00 dollar "
        "more than the ball. How much does the ball cost in dollars?"
    )
    mode = os.getenv("COT_MODE", "direct")
    print(f"\n=== Question: {q}\n=== Mode: {mode}\n")
    res = run(q, mode=mode)
    if res.reasoning:
        print(f"Reasoning:\n{res.reasoning}\n")
    if res.samples:
        print("Samples:")
        for i, a in enumerate(res.samples, 1):
            print(f"  sample {i}: {a}")
        print(f"\nVote counts: {res.vote_counts}  (agreement {res.vote_agreement:.0%})")
    print(f"\n=== Final answer: {res.final_answer}  ({res.llm_calls} llm calls)")
