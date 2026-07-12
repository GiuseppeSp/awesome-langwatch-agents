"""
Heterogeneous mixture-of-agents — #28 found that a homogeneous mixture (one model
sampled N times) collapses to a single agent, because same-model drafts make
CORRELATED errors. Its parting claim was that the famous MoA gains need MODEL
DIVERSITY. This agent tests exactly that, on the identical set-recall task and matcher.

Same three-mode shape as #28, but the drafters change:
    MA_MODE=single         (default) — one model lists the set. Baseline.
    MA_MODE=moa                      — N drafters -> an aggregator LLM synthesizes.
                                       run_eval calls this twice: with the SAME model
                                       N times (homogeneous, = #28) and with N DIFFERENT
                                       models (heterogeneous, the test).

The drafter models are passed in `models`. Homogeneous = [mini, mini, mini];
heterogeneous = [mini, gpt-3.5-turbo, gpt-4o]. The aggregator is held fixed (gpt-4o-mini)
so the ONLY thing that changes between homo and hetero is whether the drafts come from
one model or several — isolating diversity. Drafts sample at MOA_TEMP > 0 so even the
homogeneous drafts differ (giving it the same fair shot it got in #28).

Trace tree (typed LangWatch spans):

    heterogeneous_moa (workflow root)
    ├─ single: draft (llm, model=X)
    └─ moa:    draft_1..N (agent, one per model) -> aggregate (llm)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

import langwatch
from evals import dedup

load_dotenv()

# ---- Configuration ----

MA_MODE = os.getenv("MA_MODE", "single")
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
AGG_MODEL = os.getenv("AGG_MODEL", "gpt-4o-mini")
HETERO_MODELS = os.getenv("HETERO_MODELS", "gpt-4o-mini,gpt-3.5-turbo,gpt-4o").split(",")
MOA_TEMP = float(os.getenv("MOA_TEMP", "0.7"))

os.environ.setdefault("OTEL_SERVICE_NAME", "heterogeneous-moa")
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
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    try:
        d = json.loads(raw)
        if isinstance(d, list):
            return [str(x).strip() for x in d if str(x).strip()]
        if isinstance(d, dict):
            for v in d.values():
                if isinstance(v, list):
                    return [str(x).strip() for x in v if str(x).strip()]
    except json.JSONDecodeError:
        pass
    parts = [p.strip(" -*0123456789.\t") for p in raw.replace(",", "\n").splitlines()]
    return [p for p in parts if p]


def _draft(question: str, model: str, temperature: float,
           system: str = _DRAFT_SYS, user: str | None = None) -> list[str]:
    c = client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user or question}],
    )
    return _parse_list(c.choices[0].message.content or "")


# ---- Result type ----

@dataclass
class HMoAResult:
    mode: str
    label: str = ""                      # e.g. "single:gpt-4o" or "moa:hetero"
    items: list[str] = field(default_factory=list)
    drafts: list[tuple] = field(default_factory=list)  # [(model, items), ...]
    llm_calls: int = 0

    @property
    def draft_items(self) -> list[list[str]]:
        return [items for _, items in self.drafts]


# ---- The agent ----

@langwatch.trace(name="heterogeneous_moa")
def run(question: str, *, mode: str = MA_MODE, models: list[str] | None = None,
        label: str = "") -> HMoAResult:
    models = models or [PRIMARY_MODEL]
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[{mode}:{label or ','.join(models)}] {question}")
    result = HMoAResult(mode=mode, label=label or ",".join(models))

    if mode == "single":
        m = models[0]
        with langwatch.span(name="draft", type="llm") as s:
            s.update(input=f"{m}: {question}")
            items = _draft(question, m, temperature=MOA_TEMP)
            s.update(output=f"{len(items)} items")
        result.items = dedup(items)
        result.llm_calls = 1
        root.update(output=f"[single:{m}] {len(result.items)} items")
        return result

    if mode != "moa":
        raise ValueError(f"unknown mode: {mode!r}")

    for i, m in enumerate(models, 1):
        with langwatch.span(name=f"draft_{i}", type="agent") as s:
            s.update(input=f"agent {i} ({m})")
            d = _draft(question, m, temperature=MOA_TEMP)
            s.update(output=f"{m}: {len(d)} items")
        result.drafts.append((m, d))
        result.llm_calls += 1

    with langwatch.span(name="aggregate", type="llm") as s:
        drafts_txt = "\n".join(f"Draft {i+1}: {json.dumps(d)}" for i, (_, d) in enumerate(result.drafts))
        user = f"Question: {question}\n\n{drafts_txt}"
        s.update(input=f"{AGG_MODEL} over {len(result.drafts)} drafts")
        agg = _draft(question, AGG_MODEL, temperature=0.0, system=_AGG_SYS, user=user)
        s.update(output=f"{len(agg)} items")
    result.items = dedup(agg)
    result.llm_calls += 1

    root.update(output=f"[moa:{result.label}] {len(result.items)} items ({result.llm_calls} calls)")
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "List all the chemical elements whose symbol is a single letter."
    print(f"\n=== {q}\n")
    for mode, models, label in [
        ("single", ["gpt-4o-mini"], "gpt-4o-mini"),
        ("moa", ["gpt-4o-mini"] * 3, "homogeneous"),
        ("moa", HETERO_MODELS, "heterogeneous"),
    ]:
        r = run(q, mode=mode, models=models, label=label)
        print(f"[{label:>13}] {len(r.items)} items, {r.llm_calls} calls"
              + (f"  drafts: {[(m.split('-')[-1], len(d)) for m, d in r.drafts]}" if r.drafts else ""))
