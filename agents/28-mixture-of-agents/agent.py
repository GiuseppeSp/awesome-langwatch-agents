"""
Mixture-of-agents — for a task where no single answer suffices (list a whole SET),
does combining several agents' drafts beat one agent — and is the win the ENSEMBLE
(just merging the drafts) or the SYNTHESIS (an aggregator that reconciles them)?

Switch via env var:
    MA_MODE=single   (default) — one agent lists the set. The baseline.
    MA_MODE=union              — N agents draft independently; take the UNION of their
                                 items (dedup), no aggregator. Ensemble, no synthesis.
    MA_MODE=moa                — N agents draft, then an AGGREGATOR LLM synthesizes one
                                 final list: keep every correct item, drop duplicates
                                 AND items that don't belong. Mixture-of-agents proper.

single → union → moa each adds exactly one thing, so the pair of steps isolates them:
union vs single = does ensembling more drafts add coverage; moa vs union = does the
aggregator add anything over a naive merge (its job is precision — pruning the wrong
items that N drafts accumulate). Drafts sample at MOA_TEMP > 0 so they actually differ.

Trace tree (typed LangWatch spans):

    mixture_of_agents (workflow root)
    ├─ single: draft (llm)
    ├─ union:  draft (llm) ×N -> merge (span, dedup in code)
    └─ moa:    draft (llm) ×N -> aggregate (llm)
"""

from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

import langwatch
from evals import dedup

load_dotenv()

# ---- Configuration ----

MA_MODE = os.getenv("MA_MODE", "single")  # single | union | moa
MODEL = os.getenv("MODEL", "gpt-4o-mini")
N_AGENTS = int(os.getenv("N_AGENTS", "3"))
MOA_TEMP = float(os.getenv("MOA_TEMP", "0.7"))  # drafts must diverge to add coverage

os.environ.setdefault("OTEL_SERVICE_NAME", "mixture-of-agents")
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- LLM helpers ----

_DRAFT_SYS = (
    "You answer a question that asks for a complete SET of items. List every item that "
    "belongs and nothing that doesn't.\n"
    'Respond with ONLY a JSON array of strings, e.g. ["item one", "item two"].'
)

_AGG_SYS = (
    "You are given several draft lists answering the same question, from different agents. "
    "Produce the single definitive answer: include EVERY item that genuinely belongs (union "
    "the drafts' correct items), remove duplicates, and remove any item that does NOT belong "
    "(an agent's mistake). Use your own knowledge to adjudicate.\n"
    'Respond with ONLY a JSON array of strings.'
)


def _parse_list(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("```"):  # strip markdown code fence (```json ... ```)
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    try:
        d = json.loads(raw)
        if isinstance(d, list):
            return [str(x).strip() for x in d if str(x).strip()]
        if isinstance(d, dict):  # tolerate {"items": [...]}
            for v in d.values():
                if isinstance(v, list):
                    return [str(x).strip() for x in v if str(x).strip()]
    except json.JSONDecodeError:
        pass
    # forgiving fallback: split on newlines / commas, strip bullets
    parts = [p.strip(" -*0123456789.\t") for p in raw.replace(",", "\n").splitlines()]
    return [p for p in parts if p]


def _draft(question: str, temperature: float, system: str = _DRAFT_SYS, user: str | None = None) -> list[str]:
    c = client.chat.completions.create(
        model=MODEL, temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user or question}],
    )
    return _parse_list(c.choices[0].message.content or "")


# ---- Result type ----

@dataclass
class MoAResult:
    mode: str
    items: list[str] = field(default_factory=list)
    drafts: list[list[str]] = field(default_factory=list)
    llm_calls: int = 0


# ---- The agent ----

@langwatch.trace(name="mixture_of_agents")
def run(question: str, *, mode: str = MA_MODE) -> MoAResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {question}")
    result = MoAResult(mode=mode)

    if mode == "single":
        with langwatch.span(name="draft", type="llm") as s:
            s.update(input=question)
            items = _draft(question, temperature=MOA_TEMP)
            s.update(output=f"{len(items)} items")
        result.items = dedup(items)
        result.llm_calls = 1
        root.update(output=f"[single] {len(result.items)} items")
        return result

    if mode not in ("union", "moa"):
        raise ValueError(f"unknown mode: {mode!r}")

    for i in range(N_AGENTS):
        with langwatch.span(name=f"draft_{i+1}", type="agent") as s:
            s.update(input=f"agent {i+1}")
            d = _draft(question, temperature=MOA_TEMP)
            s.update(output=f"{len(d)} items")
        result.drafts.append(d)
        result.llm_calls += 1

    if mode == "union":
        with langwatch.span(name="merge", type="span") as s:
            merged = dedup([it for d in result.drafts for it in d])
            s.update(input=f"{sum(len(d) for d in result.drafts)} items across {N_AGENTS} drafts",
                     output=f"{len(merged)} after dedup")
        result.items = merged
    else:  # moa
        with langwatch.span(name="aggregate", type="llm") as s:
            drafts_txt = "\n".join(f"Draft {i+1}: {json.dumps(d)}" for i, d in enumerate(result.drafts))
            user = f"Question: {question}\n\n{drafts_txt}"
            s.update(input=f"{N_AGENTS} drafts")
            agg = _draft(question, temperature=0.0, system=_AGG_SYS, user=user)
            s.update(output=f"{len(agg)} items")
        result.items = dedup(agg)
        result.llm_calls += 1

    root.update(output=f"[{mode}] {len(result.items)} items ({result.llm_calls} calls)")
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "List all the countries that border Germany."
    mode = os.getenv("MA_MODE", "single")
    print(f"\n=== {q}\n=== Mode: {mode}\n")
    res = run(q, mode=mode)
    if res.drafts:
        for i, d in enumerate(res.drafts, 1):
            print(f"  draft {i} ({len(d)}): {d}")
        print()
    print(f"final ({len(res.items)}): {res.items}")
    print(f"llm calls: {res.llm_calls}")
