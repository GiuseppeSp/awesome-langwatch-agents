"""
Structured-extraction agent — pull a fixed set of fields out of an unstructured
bio, either free-form or under a typed JSON schema.

Switch via env var:
    EX_MODE=freeform  (default) — ask for the fields in plain text, parse the
                                  reply heuristically. No format guarantee.
    EX_MODE=schema              — OpenAI JSON mode + a typed schema with an
                                  explicit "use null if not stated" rule; parse
                                  with json.loads.

Both extract the same four fields (name, role, city, years_experience) from the
same bios; the only difference is whether the output is constrained to a schema.
The questions: how much does the schema buy in *parse reliability* and accuracy,
and — since the schema asks for every field — does its slot-filling pressure make
the model FABRICATE values for fields the bio never states?

Trace tree (typed LangWatch spans):

    structured_extraction (workflow root)
    └─ extract (llm)   [freeform: plain completion | schema: JSON-mode completion]
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field as dc_field

from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration ----

EX_MODE = os.getenv("EX_MODE", "freeform")  # "freeform" | "schema"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

os.environ.setdefault("OTEL_SERVICE_NAME", "structured-extraction")
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()

FIELDS = ["name", "role", "city", "years_experience"]


# ---- Prompts ----

_FREEFORM_SYSTEM = (
    "Extract the person's full name, job role, city, and years of experience from the bio. "
    "Write one line per field as 'Field: value'. If the bio does not state a field, write "
    "'Field: not stated'."
)

_SCHEMA_SYSTEM = (
    "Extract information from the bio into a JSON object with EXACTLY these keys: "
    '"name" (string), "role" (string), "city" (string), "years_experience" (integer). '
    "Use null for any field the bio does not explicitly state. Do not guess or infer a value "
    "that is not in the text. Output only the JSON object."
)


# ---- Parsing helpers ----

_ABSENT = {"", "not stated", "null", "none", "n/a", "unknown", "not mentioned", "not specified"}


def _clean(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().strip("'\"").strip()
    return None if s.lower() in _ABSENT else s


def _to_int(v) -> int | None:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    m = re.search(r"-?\d+", str(v))
    return int(m.group()) if m else None


def _label_to_field(label: str) -> str | None:
    """Forgiving, variant-tolerant mapping: 'Full Name', '**Job Role**', 'Location', etc.
    Free-form parsing has to anticipate however the model phrases the label."""
    l = re.sub(r"[*_`#>\-•]", "", label).strip().lower()  # strip markdown/bullets
    if "year" in l or "experience" in l:
        return "years_experience"
    if "name" in l:
        return "name"
    if "role" in l or "title" in l or "position" in l or "job" in l:
        return "role"
    if "city" in l or "location" in l or "based" in l or "place" in l:
        return "city"
    return None


def _parse_freeform(text: str) -> tuple[dict, bool]:
    """Map 'Field: value' lines onto the schema. parse_ok = all four fields found."""
    rec: dict = {f: None for f in FIELDS}
    found = {f: False for f in FIELDS}
    for line in text.splitlines():
        if ":" not in line:
            continue
        label, _, value = line.partition(":")
        key = _label_to_field(label)
        if key and not found[key]:
            rec[key] = _to_int(value) if key == "years_experience" else _clean(value)
            found[key] = True
    return rec, all(found.values())


def _parse_schema(text: str) -> tuple[dict, bool]:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {f: None for f in FIELDS}, False
    rec = {
        "name": _clean(data.get("name")),
        "role": _clean(data.get("role")),
        "city": _clean(data.get("city")),
        "years_experience": _to_int(data.get("years_experience")),
    }
    return rec, all(k in data for k in FIELDS)


# ---- Result type ----

@dataclass
class ExtractResult:
    bio: str
    mode: str
    record: dict = dc_field(default_factory=dict)
    parse_ok: bool = False
    raw: str = ""
    llm_calls: int = 0


# ---- The agent ----

@langwatch.trace(name="structured_extraction")
def run(bio: str, *, mode: str = EX_MODE) -> ExtractResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {bio}")
    result = ExtractResult(bio=bio, mode=mode)

    if mode not in ("freeform", "schema"):
        raise ValueError(f"unknown mode: {mode!r}")

    system = _SCHEMA_SYSTEM if mode == "schema" else _FREEFORM_SYSTEM
    kwargs = {"response_format": {"type": "json_object"}} if mode == "schema" else {}

    with langwatch.span(name="extract", type="llm") as s:
        s.update(input=bio)
        c = client.chat.completions.create(
            model=MODEL, temperature=TEMPERATURE,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": bio}],
            **kwargs,
        )
        text = (c.choices[0].message.content or "").strip()
        s.update(output=text[:300], metrics={
            "prompt_tokens": c.usage.prompt_tokens, "completion_tokens": c.usage.completion_tokens})
    result.llm_calls += 1
    result.raw = text

    result.record, result.parse_ok = (
        _parse_schema(text) if mode == "schema" else _parse_freeform(text)
    )
    root.update(output=f"[{mode} parse_ok={result.parse_ok}] {result.record}")
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    bio = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Wen Li is a UX writer in Brightwater."
    mode = os.getenv("EX_MODE", "freeform")
    print(f"\n=== Bio: {bio}\n=== Mode: {mode}\n")
    res = run(bio, mode=mode)
    print(f"Raw output:\n{res.raw}\n")
    print(f"Parsed record (parse_ok={res.parse_ok}):")
    for f in FIELDS:
        print(f"  {f}: {res.record.get(f)!r}")
