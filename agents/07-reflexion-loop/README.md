# 07 · Reflexion Loop

A self-critique-and-retry agent on constraint-satisfaction tasks. After producing an answer, the agent reviews its own output against the stated constraint, decides whether it satisfies it, and either commits or retries with the reflection added to the next prompt. Up to 3 attempts. This is the Reflexion pattern (Shinn et al., 2023) reduced to its core question: **does asking a model to grade its own work make its work better?**

The experiment is one variable: **single_pass vs reflexion**. Same model (gpt-4o-mini), same temperature, same 15 prompts. The only difference is whether a self-critique step runs between attempts. The headline question isn't "does reflexion help?" — it's "**is the model an accurate judge of its own output?**" If the answer is no, then no amount of retry budget can fix it.

This is the third agent in the catalog's verification arc:

> **#5** post-write verifier — can't catch upstream errors (scope problem).
> **#6** in-loop reasoning — talks model into bypassing tools (commitment problem).
> **#7** self-critique + retry — does the model know when it's wrong?

## The pipeline

```
single_pass mode:
    generate → done

reflexion mode:
    attempt_1: generate → self_critique
        self says PASS → done
        otherwise      → retry with reflection in context
    attempt_2: generate → self_critique
        self says PASS → done; otherwise retry
    attempt_3: generate → self_critique → done (no more retries)
```

Each attempt is a `langwatch.span(type="agent", name="attempt_N")` under the `reflexion_loop` workflow root, with the generator LLM call (`type="llm"`) and the self-critique call (`type="evaluation"`) as nested children. The trace tree shows exactly how many attempts the agent burned — and what its self-verdict was on each.

See [`agent.py`](agent.py) — ~200 lines, raw OpenAI + LangWatch, no agent framework.

## The dataset

15 prompts ([`dataset.csv`](dataset.csv)) across four constraint types, all mechanically verifiable:

| Constraint type | Example | Count |
|---|---|---|
| `word_count` | *"Write a sentence about the moon using exactly 10 words."* | 5 |
| `list_length_letter` | *"List exactly 7 European capital cities whose names start with B. One per line."* | 4 |
| `letter_exclusion` | *"Write a sentence about a thunderstorm that does NOT contain the letter A."* (a lipogram) | 3 |
| `alliteration` | *"Write a sentence about a forest where every word starts with F."* | 3 |

These are deliberately chosen for **non-trivial single-pass failure rates** — LLMs are notoriously bad at exact word counts, lipograms, and alliteration. If the dataset were all easy, single_pass would saturate and there'd be nothing to measure.

## The evaluators

Three scorers ([`evals.py`](evals.py)) — **all programmatic**, since the constraints are mechanically verifiable. This is the first agent in the catalog with zero LLM-judge evals, by design: judges add noise the eval doesn't need. The same verifier function that scores `constraint_satisfaction` also provides the ground truth that `self_critique_accuracy` measures the agent against.

| Evaluator | Type | Measures |
|---|---|---|
| `constraint_satisfaction` | Programmatic | Did the final output satisfy the constraint EXACTLY? Pass/fail. Applies to both modes. |
| `retry_lift` | Programmatic | Did the retry loop help or hurt? Per row: +1 if first failed but final passed (helped), 0 if no change, -1 if first passed but final failed (perverse — H3 hypothesis manifested). Normalized to [0, 1] for averaging (helped=1.0, neutral=0.5, hurt=0.0). Reflexion-only. |
| `self_critique_accuracy` | Programmatic | When the agent said `PASS` on its own answer, was the answer actually right? When it said `FAIL` and listed issues, were there real issues? Fraction-correct over all attempts. Reflexion-only. **This is the key metric** — it directly answers whether self-criticism is grounded. |

## The tuning experiment

> **single_pass vs reflexion on 15 mechanically-verifiable constraint prompts**

The driver ([`run_eval.py`](run_eval.py)) runs every row twice and prints the side-by-side comparison plus a per-row detail view showing `attempts` count, constraint pass/fail, retry lift sign, and self-critique accuracy.

### Three hypotheses to discriminate

| | Outcome | What it would teach |
|---|---|---|
| **H1** (textbook) | reflexion > single_pass on `constraint_satisfaction`, with positive `retry_lift` | The classic Reflexion paper finding — self-critique catches misses, retry fixes them. |
| **H2** (null) | reflexion ≈ single_pass; `self_critique_accuracy` is low | Agent has the same blind spots when grading itself as when generating — consistent bias. The retry budget burns tokens without changing outcomes. |
| **H3** (perverse) | reflexion < single_pass; `retry_lift` goes negative on some rows | Self-critique talks the model out of correct answers. Echoes agent #6's lesson: externalized reasoning commits the model to whichever conclusion it reaches first, including wrong ones. |

### Results

🚧 _Pending — baseline run not yet executed. Update once the run is complete._

## Quick start

```bash
cd agents/07-reflexion-loop
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY and LANGWATCH_API_KEY (Project API Key, type Service)

# Smoke test: one prompt, single_pass
python agent.py "Write a single sentence about the moon using exactly 10 words."

# Same prompt, reflexion (you'll see the attempts in the console output)
REFLEXION_MODE=reflexion python agent.py "Write a single sentence about the moon using exactly 10 words."

# Full comparison
python run_eval.py                  # both modes, all 15 rows
python run_eval.py --limit 3        # smoke test
```

If you hit the macOS Xcode-Python SSL issue (`Errno 2: No such file or directory` on `ssl.py`):

```bash
pip install certifi
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

## What you'll learn

How a self-critique loop looks in LangWatch when each attempt is its own typed span — and the deeper PM-relevant question: whether the foundational assumption behind reflexion-style patterns (that a model can grade its own work accurately enough for retry to be productive) actually holds on the kind of tasks teams ship in 2026. If `self_critique_accuracy` is high, the loop is well-founded. If it's low, no amount of retry budget can fix the agent — the retry just burns tokens chasing the same wrong verdict.

## Status

🚧 Code shipped — baseline run pending. The honest framing will land after the local run, depending on which of the three hypotheses the data supports.
