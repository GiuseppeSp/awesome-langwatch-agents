"""
SQL agent — natural-language question -> SQL -> execute against a SQLite database.

Switch via env var:
    SQL_MODE=blind    (default) — write SQL with NO schema shown. The model must
                                  guess table and column names.
    SQL_MODE=grounded           — the schema (CREATE statements) is put in the
                                  prompt; the model writes SQL against the real names.

Same questions, same database, one LLM call each; the only difference is whether
the schema is in the prompt. The database uses deliberately non-obvious column
names (`handle` not `name`, `reward_gp`, `joined_year`), so blind SQL that guesses
conventional names errors out with "no such column". The experiment measures how
much of correctness is just *schema grounding* — and what fails even once the
schema is known.

Trace tree (typed LangWatch spans):

    sql_agent (workflow root)
    ├─ write_sql (llm)         NL -> SQL (schema in prompt only for grounded)
    └─ execute  (tool)         run the SQL against SQLite, capture rows or error
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration ----

SQL_MODE = os.getenv("SQL_MODE", "blind")  # "blind" | "grounded"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

os.environ.setdefault("OTEL_SERVICE_NAME", "sql-agent")
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- The database (fictional adventurers' guild; non-obvious column names) ----

SCHEMA = """\
CREATE TABLE members (id INTEGER PRIMARY KEY, handle TEXT, rank TEXT, joined_year INTEGER, home_city TEXT);
CREATE TABLE quests (id INTEGER PRIMARY KEY, title TEXT, reward_gp INTEGER, difficulty TEXT, region TEXT);
CREATE TABLE completions (member_id INTEGER, quest_id INTEGER, completed_year INTEGER);"""

_MEMBERS = [(1, "Ironwood", "Ranger", 1203, "Duskvale"), (2, "Marlow", "Knight", 1200, "Highmoor"),
            (3, "Sable", "Mage", 1203, "Duskvale"), (4, "Grix", "Rogue", 1198, "Fenwick"),
            (5, "Yara", "Cleric", 1201, "Highmoor"), (6, "Tobin", "Ranger", 1203, "Fenwick")]
_QUESTS = [(1, "The Sunken Vault", 500, "hard", "Frostpeak"), (2, "Hollow Root", 150, "easy", "Mirewood"),
           (3, "Ashfall Ridge", 800, "hard", "Frostpeak"), (4, "The Glass Bridge", 300, "medium", "Mirewood"),
           (5, "Wyrm's Hollow", 650, "hard", "Frostpeak"), (6, "Pilgrim's Path", 100, "easy", "Emberfell")]
_COMPLETIONS = [(1, 1, 1204), (1, 3, 1205), (1, 5, 1205), (2, 2, 1201), (2, 4, 1202), (3, 1, 1204),
                (3, 4, 1203), (4, 3, 1204), (5, 2, 1202), (5, 6, 1203), (6, 5, 1206), (6, 6, 1204)]


def build_db() -> sqlite3.Connection:
    """A fresh in-memory DB per call, so a stray model query can't corrupt others."""
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO members VALUES (?,?,?,?,?)", _MEMBERS)
    con.executemany("INSERT INTO quests VALUES (?,?,?,?,?)", _QUESTS)
    con.executemany("INSERT INTO completions VALUES (?,?,?)", _COMPLETIONS)
    con.commit()
    return con


def run_sql(sql: str) -> tuple[list, str | None]:
    """Execute SQL, return (rows, error). rows is a flat sorted list of cell strings."""
    try:
        con = build_db()
        cur = con.execute(sql)
        rows = cur.fetchall()
        con.close()
        return sorted(str(c) for row in rows for c in row), None
    except Exception as e:  # noqa: BLE001
        return [], str(e)


# ---- LLM ----

def _extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.+?)```", text, flags=re.DOTALL | re.IGNORECASE)
    sql = (m.group(1) if m else text).strip()
    return sql.rstrip(";").strip()


_SYSTEM = (
    "You are a SQLite expert. Write a single SQLite query that answers the question. "
    "Return ONLY the SQL, no explanation."
)


# ---- Result type ----

@dataclass
class SQLResult:
    question: str
    mode: str
    sql: str = ""
    rows: list = field(default_factory=list)
    error: str | None = None
    llm_calls: int = 0

    @property
    def executed_ok(self) -> bool:
        return self.error is None


# ---- The agent ----

@langwatch.trace(name="sql_agent")
def run(question: str, *, mode: str = SQL_MODE) -> SQLResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {question}")
    result = SQLResult(question=question, mode=mode)

    if mode not in ("blind", "grounded"):
        raise ValueError(f"unknown mode: {mode!r}")

    user = (f"SCHEMA:\n{SCHEMA}\n\nQUESTION: {question}" if mode == "grounded"
            else f"QUESTION: {question}")

    with langwatch.span(name="write_sql", type="llm") as s:
        s.update(input=user[-400:])
        c = client.chat.completions.create(
            model=MODEL, temperature=TEMPERATURE,
            messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        )
        raw = (c.choices[0].message.content or "").strip()
        s.update(output=raw[:300], metrics={
            "prompt_tokens": c.usage.prompt_tokens, "completion_tokens": c.usage.completion_tokens})
    result.llm_calls += 1
    result.sql = _extract_sql(raw)

    with langwatch.span(name="execute", type="tool") as s:
        s.update(input=result.sql)
        result.rows, result.error = run_sql(result.sql)
        s.update(output=(f"ERROR: {result.error}" if result.error else str(result.rows))[:400])

    root.update(output=f"[{mode} ok={result.executed_ok}] {result.rows or result.error}"[:200])
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What rank is the member whose handle is 'Ironwood'?"
    mode = os.getenv("SQL_MODE", "blind")
    print(f"\n=== Question: {q}\n=== Mode: {mode}\n")
    res = run(q, mode=mode)
    print(f"SQL:\n{res.sql}\n")
    print(f"Executed: {res.executed_ok}" + (f"  (ERROR: {res.error})" if res.error else ""))
    print(f"Result: {res.rows}")
