"""
Evaluators for the structured-extraction agent.

Fixed fields, known gold values → all programmatic, no judge noise (consistent
with the rest of the catalog). Each gold field is either a value or the marker
ABSENT (the bio never states it).

    parse_success    — Did the output yield a complete, usable 4-field record?
                       (schema: valid JSON with all keys; freeform: all four
                       'Field: value' lines found.) The headline reliability win.

    record_accuracy  — Are all four fields correct, INCLUDING correctly leaving an
                       ABSENT field null? Per-field then all-or-nothing. Both modes.

    field_accuracy   — Fraction of the 4 fields correct (partial credit), so the
                       gap between modes is visible even when no record is perfect.

(run_eval also reports `fabrication`: how many ABSENT fields got a made-up value —
the schema's slot-filling failure mode.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FIELDS = ["name", "role", "city", "years_experience"]


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float
    reason: str = ""


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s or "").lower()).strip()


def field_correct(extracted, gold: str, key: str) -> bool:
    if gold == "ABSENT":
        return extracted is None
    if extracted is None:
        return False
    if key == "years_experience":
        return str(extracted).strip() == str(gold).strip()
    g, e = _norm(gold), _norm(extracted)
    return g == e or g in e or e in g


# ---- Evaluator 1: parse_success ----

def parse_success(parse_ok: bool) -> EvalResult:
    return EvalResult("parse_success", parse_ok, 1.0 if parse_ok else 0.0,
                      "usable record" if parse_ok else "could not parse a complete record")


# ---- Evaluator 2 + 3: record / field accuracy ----

def field_accuracy(record: dict, gold: dict) -> EvalResult:
    hits = [field_correct(record.get(k), gold[k], k) for k in FIELDS]
    n = sum(hits)
    wrong = [k for k, ok in zip(FIELDS, hits) if not ok]
    return EvalResult("field_accuracy", n == len(FIELDS), n / len(FIELDS),
                      f"{n}/{len(FIELDS)} fields" + (f"; wrong: {wrong}" if wrong else ""))


def record_accuracy(record: dict, gold: dict) -> EvalResult:
    fa = field_accuracy(record, gold)
    ok = fa.score == 1.0
    return EvalResult("record_accuracy", ok, 1.0 if ok else 0.0, fa.reason)


# ---- fabrication: ABSENT gold fields that got a non-null value ----

def fabrication_count(record: dict, gold: dict) -> int:
    return sum(1 for k in FIELDS if gold[k] == "ABSENT" and record.get(k) is not None)
