"""
Evaluators for the mixture-of-agents agent.

Each question asks for a well-defined SET (e.g. "list all the countries that border
Germany"), so scoring is programmatic: recall (did we get the gold items?), precision
(are the items we listed actually correct?), and F1. Set membership is fuzzy — models
write "Czech Republic" for "Czechia", "Holland" for "Netherlands" — so ONE normalizer +
alias map is used everywhere: for scoring AND for the agent's union-dedup, so a mode
can never be helped or hurt by a stricter matcher on its side (the #20 lesson).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Variant spelling -> canonical, matched AFTER normalize(). Only the genuinely
# ambiguous ones; plain names normalize fine on their own.
ALIASES = {
    "czech republic": "czechia", "czech": "czechia",
    "holland": "netherlands", "the netherlands": "netherlands",
    "burma": "myanmar",
    "cesium": "caesium",
    "aluminum": "aluminium", "sulphur": "sulfur", "wolfram": "tungsten",
    "russian federation": "russia",
    "united states of america": "united states", "usa": "united states", "us": "united states",
    "united kingdom of great britain and northern ireland": "united kingdom", "uk": "united kingdom",
    "upenn": "pennsylvania", "penn": "pennsylvania", "university of pennsylvania": "pennsylvania",
    "newfoundland": "newfoundland and labrador",
    "pei": "prince edward island",
    "aspartate": "aspartic acid", "glutamate": "glutamic acid",
    "south korea": "south korea", "north korea": "north korea",
    "east timor": "timor leste",
}


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float
    reason: str = ""


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)  # drop parentheticals: "Burma (Myanmar)" -> "Burma"
    s = re.sub(r"^(the|republic of|kingdom of|state of)\s+", "", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canon(s: str) -> str:
    n = normalize(s)
    return ALIASES.get(n, n)


def dedup(items: list[str]) -> list[str]:
    """Case/alias-insensitive dedup, keeping first-seen original spelling."""
    seen, out = set(), []
    for it in items:
        k = canon(it)
        if k and k not in seen:
            seen.add(k)
            out.append(it)
    return out


def score_set(agent_items: list[str], gold_items: list[str]) -> dict:
    gold = {canon(g) for g in gold_items}
    got = {canon(a) for a in agent_items if a.strip()}
    hits = got & gold                       # correct items we listed
    extra = got - gold                      # items we listed that aren't in gold (false positives)
    missed = gold - got                     # gold items we didn't list
    recall = len(hits) / len(gold) if gold else 0.0
    precision = len(hits) / len(got) if got else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "recall": recall, "precision": precision, "f1": f1,
        "n_gold": len(gold), "n_listed": len(got),
        "hits": len(hits), "extra": sorted(extra), "missed": sorted(missed),
    }


def f1_result(agent_items: list[str], gold_items: list[str]) -> EvalResult:
    s = score_set(agent_items, gold_items)
    return EvalResult("f1", s["f1"] == 1.0, s["f1"],
                      f"R={s['recall']:.2f} P={s['precision']:.2f} F1={s['f1']:.2f} "
                      f"({s['hits']}/{s['n_gold']} hits, {len(s['extra'])} extra)")
