"""
Error-recovery agent — when a tool call fails, does the agent retry/repair, or
give up?

Switch via env var:
    RECOVER_MODE=no_retry  (default) — one tool attempt. On error, fail.
    RECOVER_MODE=retry               — on error, feed the error back to the LLM,
                                       let it fix the call and try again (up to
                                       MAX_ATTEMPTS), or decide to GIVE UP.

Both drive the SAME mock service. Its failures are deliberately classifiable:
  - clean resources succeed on the first call.
  - recoverable resources fail once with an error that CONTAINS the fix (an access
    token the agent can't guess up front but can copy from the error on retry).
  - unrecoverable resources fail on every call, and say so — a good retry loop
    should recognize that and give up instead of burning attempts.

The experiment measures where retrying pays (recoverable errors), where it's a
no-op (clean), and whether the loop wastes attempts on errors it can never fix
(unrecoverable).

Trace tree (typed LangWatch spans):

    error_recovery (workflow root)
    └─ attempt_k: decide (llm) -> call (tool)   [repeated in retry mode]
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration ----

RECOVER_MODE = os.getenv("RECOVER_MODE", "no_retry")  # "no_retry" | "retry"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "4"))

os.environ.setdefault("OTEL_SERVICE_NAME", "error-recovery")
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- The mock service (deterministic, classifiable failures) ----

_CLEAN = {"clock": "12:00", "status": "operational"}
_RECOVERABLE = {  # resource -> (required token, success value)
    "weather": ("9F3K", "18C"),
    "inventory": ("A7B2", "412 units"),
    "ledger": ("QX41", "balance 5,230"),
    "registry": ("ZK08", "record #77"),
}
_UNRECOVERABLE = {
    "ghost": "resource 'ghost' does not exist — no token or retry will make it appear",
    "sealed": "access to 'sealed' is permanently denied — this cannot be recovered",
    "void": "resource 'void' is permanently offline — retrying will not help",
}


def service(resource: str, token: str | None = None) -> tuple[str | None, str | None]:
    """Return (result, error). Deterministic."""
    r = (resource or "").strip().lower()
    if r in _CLEAN:
        return _CLEAN[r], None
    if r in _RECOVERABLE:
        need, value = _RECOVERABLE[r]
        if (token or "").strip() == need:
            return value, None
        return None, f"resource '{r}' requires a valid access token. Retry with token={need}"
    if r in _UNRECOVERABLE:
        return None, _UNRECOVERABLE[r]
    return None, f"unknown resource '{r}'"


# ---- LLM ----

_SYSTEM = (
    "You are an agent that fetches data from a service by calling a tool.\n"
    "The tool: service(resource, token=optional).\n"
    "Respond with ONE JSON object and nothing else:\n"
    '  {"action": "call", "resource": "<name>", "token": "<token or empty>"}\n'
    '  {"action": "give_up", "reason": "<why>"}\n'
    "If a previous call returned an error that tells you how to fix it (e.g. a token to use), "
    "call again WITH the fix. If the error says the resource is permanently unavailable / cannot "
    "be recovered, do NOT keep trying — respond with give_up."
)


def _chat(messages: list[dict]) -> tuple[str, int, int]:
    c = client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE,
        response_format={"type": "json_object"}, messages=messages,
    )
    return (c.choices[0].message.content or "").strip(), c.usage.prompt_tokens, c.usage.completion_tokens


# ---- Result type ----

@dataclass
class RecoverResult:
    task: str
    mode: str
    solved: bool = False
    gave_up: bool = False
    attempts: int = 0            # tool calls made
    llm_calls: int = 0
    result: str | None = None
    last_error: str | None = None
    trajectory: list[str] = field(default_factory=list)


# ---- The agent ----

@langwatch.trace(name="error_recovery")
def run(task: str, *, mode: str = RECOVER_MODE) -> RecoverResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {task}")
    result = RecoverResult(task=task, mode=mode)

    if mode not in ("no_retry", "retry"):
        raise ValueError(f"unknown mode: {mode!r}")

    max_attempts = 1 if mode == "no_retry" else MAX_ATTEMPTS
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": f"TASK: {task}"}]

    for attempt in range(1, max_attempts + 1):
        with langwatch.span(name=f"attempt_{attempt}", type="agent") as asp:
            with langwatch.span(name="decide", type="llm") as s:
                s.update(input=str(messages[-1])[:300])
                raw, pt, ct = _chat(messages)
                s.update(output=raw[:200], metrics={"prompt_tokens": pt, "completion_tokens": ct})
            result.llm_calls += 1
            messages.append({"role": "assistant", "content": raw})

            try:
                action = json.loads(raw)
            except json.JSONDecodeError:
                action = {"action": "call", "resource": raw}

            if action.get("action") == "give_up":
                result.gave_up = True
                asp.update(output=f"give_up: {action.get('reason', '')[:80]}")
                break

            resource = action.get("resource", "")
            token = action.get("token") or None
            with langwatch.span(name="call", type="tool") as s:
                s.update(input=f"service({resource}, token={token})")
                res, err = service(resource, token)
                s.update(output=(f"ERROR: {err}" if err else res))
            result.attempts += 1
            result.trajectory.append(f"service({resource}, token={token}) -> {res or 'ERROR: ' + (err or '')}")

            if err is None:
                result.solved = True
                result.result = res
                asp.update(output=f"solved: {res}")
                break
            result.last_error = err
            asp.update(output=f"error: {err[:80]}")
            # feed the error back for the next attempt (retry mode only continues the loop)
            messages.append({"role": "user", "content": f"The call failed with: {err}\nDecide the next action."})

    root.update(output=f"[{mode} solved={result.solved} attempts={result.attempts}{' gave_up' if result.gave_up else ''}] {result.result or result.last_error}"[:200])
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Fetch the current reading from the 'weather' resource."
    mode = os.getenv("RECOVER_MODE", "no_retry")
    print(f"\n=== Task: {q}\n=== Mode: {mode}\n")
    res = run(q, mode=mode)
    print("Trajectory:")
    for t in res.trajectory:
        print(f"  - {t}")
    if res.gave_up:
        print("  - GAVE UP")
    print(f"\nSolved: {res.solved}  |  attempts: {res.attempts}  |  llm calls: {res.llm_calls}")
    print(f"Result: {res.result or res.last_error}")
