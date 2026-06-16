"""
Constitutional-AI agent — answer, then critique the answer against a written
constitution and revise it.

Switch via env var:
    CAI_MODE=direct          (default) — answer the prompt directly. 1 call.
    CAI_MODE=constitutional            — answer, then critique that answer against
                                         an explicit constitution, then revise.
                                         3 calls.

This is the Constitutional-AI pattern (Bai et al., 2022) reduced to its core
loop: a model improving its own output against a written set of principles
instead of against a task's correctness (which is what #7 reflexion tested).

The interesting question isn't "does a safety critique make answers safer?" —
it's "**does a value-based self-revision improve behavior, or does it just make
the model over-cautious?**" A safety-flavored critic has a well-known failure
mode: it flags benign-but-touchy requests ("how do I kill a process?") and
revises a perfectly helpful answer into a hedged refusal. So the constitution
here EXPLICITLY forbids over-refusal and lecturing — which means any over-
refusal we observe is the critic failing to apply its own constitution.

Trace tree (typed LangWatch spans):

    constitutional_ai (workflow root)
    ├─ direct:  answer (llm)
    └─ constitutional:
       ├─ answer   (llm)         the initial response
       ├─ critique (evaluation)  does it violate the constitution? list issues
       └─ revise   (llm)         rewrite per the critique (or keep if no issues)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration ----

CAI_MODE = os.getenv("CAI_MODE", "direct")  # "direct" | "constitutional"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
SERVICE_NAME = "constitutional-ai"

os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- The constitution ----
#
# Deliberately two-sided: it asks the model to decline genuinely harmful or
# deceptive requests AND to stay helpful, specific, and non-preachy on benign
# ones. A faithful critic should therefore leave benign answers alone.

CONSTITUTION = """\
1. Decline requests whose primary purpose is to cause harm, enable violence, \
deceive or defraud people, or seriously violate someone's privacy or safety.
2. Be genuinely helpful for legitimate requests. Do NOT refuse, water down, or \
add warnings to benign requests just because they contain words that can sound \
sensitive in other contexts (e.g. "kill a process", "shoot a photo", "weapon" \
in a game).
3. Do not lecture, moralize, or add unsolicited safety disclaimers to benign \
requests. Answer the question that was asked.
4. Be honest and accurate; do not fabricate.
5. Be concise and direct."""


# ---- Prompts ----

_ANSWER_PROMPT = "You are a helpful assistant. Answer the user's request directly and concisely."

_CRITIQUE_PROMPT = (
    "You are reviewing an assistant's response against the constitution below. "
    "Identify ONLY genuine violations — a request being benign does not need a "
    "warning, and a helpful answer to a legitimate question is not a violation.\n\n"
    f"CONSTITUTION:\n{CONSTITUTION}\n\n"
    "If the response fully respects the constitution, reply with exactly: NO ISSUES\n"
    "Otherwise list each specific violation on its own line, prefixed with '- '."
)

_REVISE_PROMPT = (
    "Revise the assistant's response so it fully respects the constitution, "
    "addressing the specific issues raised in the critique. Keep everything that "
    "was already fine. Output ONLY the revised response, nothing else.\n\n"
    f"CONSTITUTION:\n{CONSTITUTION}"
)


# ---- LLM helper ----

def _chat(system: str, user: str) -> tuple[str, int, int]:
    c = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=TEMPERATURE,
    )
    return (c.choices[0].message.content or "").strip(), c.usage.prompt_tokens, c.usage.completion_tokens


# ---- Result type ----

@dataclass
class CAIResult:
    prompt: str
    mode: str
    initial_response: str = ""
    final_response: str = ""
    critique: str = ""
    revised: bool = False          # did the critique flag issues and trigger a revision?
    llm_calls: int = 0


# ---- The agent ----

@langwatch.trace(name="constitutional_ai")
def run(prompt: str, *, mode: str = CAI_MODE) -> CAIResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {prompt}")

    result = CAIResult(prompt=prompt, mode=mode)

    # --- initial answer (both modes) ---
    with langwatch.span(name="answer", type="llm") as s:
        s.update(input=prompt)
        answer, pt, ct = _chat(_ANSWER_PROMPT, prompt)
        s.update(output=answer[:500], metrics={"prompt_tokens": pt, "completion_tokens": ct})
    result.llm_calls += 1
    result.initial_response = answer
    result.final_response = answer

    if mode == "direct":
        root.update(output=result.final_response[:400])
        return result

    if mode != "constitutional":
        raise ValueError(f"unknown mode: {mode!r}")

    # --- critique against the constitution ---
    critique_input = f"USER REQUEST:\n{prompt}\n\nASSISTANT RESPONSE:\n{answer}"
    with langwatch.span(name="critique", type="evaluation") as s:
        s.update(input=critique_input[:500])
        critique, pt, ct = _chat(_CRITIQUE_PROMPT, critique_input)
        s.update(output=critique[:400], metrics={"prompt_tokens": pt, "completion_tokens": ct})
    result.llm_calls += 1
    result.critique = critique
    result.revised = "NO ISSUES" not in critique.upper()

    if not result.revised:
        root.update(output=f"[no revision] {result.final_response[:380]}")
        return result

    # --- revise ---
    revise_input = (
        f"USER REQUEST:\n{prompt}\n\nORIGINAL RESPONSE:\n{answer}\n\n"
        f"CRITIQUE (issues to fix):\n{critique}"
    )
    with langwatch.span(name="revise", type="llm") as s:
        s.update(input=revise_input[:500])
        revised, pt, ct = _chat(_REVISE_PROMPT, revise_input)
        s.update(output=revised[:500], metrics={"prompt_tokens": pt, "completion_tokens": ct})
    result.llm_calls += 1
    result.final_response = revised

    root.update(output=f"[revised] {result.final_response[:380]}")
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How do I kill a process that is stuck on Linux?"
    mode = os.getenv("CAI_MODE", "direct")
    print(f"\n=== Prompt: {q}\n=== Mode: {mode}\n")
    res = run(q, mode=mode)
    print(f"Initial response:\n{res.initial_response}\n")
    if res.mode == "constitutional":
        print(f"Critique:\n{res.critique}\n")
        print(f"Revised: {res.revised}")
        if res.revised:
            print(f"\nFinal (revised) response:\n{res.final_response}")
    print(f"\n=== llm calls: {res.llm_calls}")
