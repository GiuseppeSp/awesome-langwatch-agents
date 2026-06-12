"""
Reflexion-loop agent — self-critique and retry on constraint-satisfaction tasks.

Switch via env var:
    REFLEXION_MODE=single_pass  (default) — one attempt, no retry
    REFLEXION_MODE=reflexion              — up to MAX_ATTEMPTS, with self-critique
                                            between attempts

The reflexion loop:

    attempt_1: generate → self-critique
        if self says PASS  → done
        otherwise          → retry
    attempt_2: generate (with previous attempt + reflection in context) → critique
        if PASS → done; otherwise retry
    attempt_3: generate → critique → done (no more retries)

Each attempt is a typed LangWatch span (`agent`) under the workflow root, with
the generator LLM call (`llm`) and the self-critique LLM call (`evaluation`)
as nested children. The trace tree literally shows how many times the agent
"tried" — and whether each attempt PASS'd itself or not.

Temperature is slightly non-zero (0.3) in both modes so retries can produce
different outputs (a pure temp=0 retry would deterministically repeat). The
same temperature is used in single_pass so the comparison stays clean.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration ----

REFLEXION_MODE = os.getenv("REFLEXION_MODE", "single_pass")  # "single_pass" | "reflexion"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
SERVICE_NAME = "reflexion-loop"

os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- Prompts ----

_GENERATOR_PROMPT = (
    "You are a careful assistant. The user's task always contains an explicit "
    "constraint (a specific count, letter rule, or exclusion). Produce a response "
    "that satisfies the constraint EXACTLY. Be brief; do not preface or explain. "
    "Output only the requested content."
)

_CRITIC_PROMPT = (
    "You are reviewing your OWN previous answer to a task that had an explicit "
    "constraint. Check whether the answer satisfies the constraint EXACTLY (not "
    "approximately). Common pitfalls to check for:\n"
    "- Word counts: count the words; off-by-one is still failure.\n"
    "- Letter exclusions: scan every word for the forbidden letter.\n"
    "- Alliteration: every word must start with the letter, not most.\n"
    "- List length + letter: count items; check each starts with the letter.\n\n"
    "Reply with EITHER:\n"
    "- 'PASS' on the first line if the answer fully satisfies the constraint.\n"
    "- Otherwise, list specific issues, one per line, prefixed with '- '."
)


# ---- LLM helpers ----

def _generate(prompt: str, prev_output: str = "", prev_reflection: str = "") -> str:
    """Run the generator LLM. If prev_output + prev_reflection are provided, it's a retry."""
    if prev_output and prev_reflection:
        user_content = (
            f"Task: {prompt}\n\n"
            f"Your previous attempt:\n{prev_output}\n\n"
            f"Issues you identified with that attempt:\n{prev_reflection}\n\n"
            "Try again. Address the specific issues. Output only the new attempt."
        )
    else:
        user_content = prompt

    with langwatch.span(name="generate", type="llm") as s:
        s.update(input=user_content[:400])
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _GENERATOR_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=TEMPERATURE,
        )
        out = (completion.choices[0].message.content or "").strip()
        s.update(
            output=out[:400],
            metrics={
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
            },
        )
    return out


def _critique(prompt: str, attempt_output: str) -> tuple[str, bool]:
    """Run the self-critique LLM. Returns (full_reflection_text, self_says_pass)."""
    user_content = f"Task: {prompt}\n\nYour answer:\n{attempt_output}"

    with langwatch.span(name="self_critique", type="evaluation") as s:
        s.update(input=user_content[:400])
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _CRITIC_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
        )
        raw = (completion.choices[0].message.content or "").strip()
        s.update(
            output=raw[:400],
            metrics={
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
            },
        )

    first_line = raw.split("\n", 1)[0].strip().upper()
    self_says_pass = first_line == "PASS"
    return raw, self_says_pass


# ---- The pipeline ----

@dataclass
class Attempt:
    output: str
    reflection: str = ""
    self_says_pass: bool = False


@dataclass
class ReflexionResult:
    prompt: str
    mode: str
    attempts: list[Attempt] = field(default_factory=list)
    final_output: str = ""

    @property
    def num_attempts(self) -> int:
        return len(self.attempts)

    @property
    def first_attempt_output(self) -> str:
        return self.attempts[0].output if self.attempts else ""


@langwatch.trace(name="reflexion_loop")
def run(prompt: str, *, mode: str = REFLEXION_MODE) -> ReflexionResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {prompt}")

    result = ReflexionResult(prompt=prompt, mode=mode)

    if mode == "single_pass":
        with langwatch.span(name="attempt_1", type="agent") as s:
            output = _generate(prompt)
            s.update(output=output[:300])
        result.attempts.append(Attempt(output=output))
        result.final_output = output

    else:  # reflexion
        prev_output, prev_reflection = "", ""
        for n in range(1, MAX_ATTEMPTS + 1):
            with langwatch.span(name=f"attempt_{n}", type="agent") as s:
                output = _generate(prompt, prev_output, prev_reflection)
                reflection, says_pass = _critique(prompt, output)
                s.update(output=f"output: {output[:200]} | self_says_pass: {says_pass}")

            result.attempts.append(
                Attempt(output=output, reflection=reflection, self_says_pass=says_pass)
            )

            if says_pass or n == MAX_ATTEMPTS:
                break

            prev_output, prev_reflection = output, reflection

        result.final_output = result.attempts[-1].output

    root.update(output=result.final_output[:400])
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Write a single sentence about the moon using exactly 10 words."
    )
    mode = os.getenv("REFLEXION_MODE", "single_pass")
    print(f"\n=== Prompt: {q}\n=== Mode: {mode}\n")
    result = run(q, mode=mode)
    print(f"Attempts used: {result.num_attempts}")
    for i, a in enumerate(result.attempts, 1):
        print(f"\n--- Attempt {i} ---")
        print(f"Output: {a.output}")
        if a.reflection:
            print(f"Self-critique:\n{a.reflection}")
            print(f"self_says_pass: {a.self_says_pass}")
    print(f"\n=== Final output ===\n{result.final_output}")
