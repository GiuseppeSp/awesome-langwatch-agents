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

**Aggregate across 15 rows**

| mode | constraint_satisfaction | retry_lift | self_critique_accuracy |
|---|---|---|---|
| `single_pass` | 0.73 (11/15) | n/a | n/a |
| `reflexion` | **0.80 (12/15)** | 0.47 (slightly below neutral) | **0.40 (wrong 60% of the time)** |

The headline looks like a textbook H1 result — reflexion beats single-pass by 7 percentage points on the outcome metric. **That reading is wrong.** A careful look at the two orthogonal evaluators tells a different story.

### What's actually happening

**+7pp is one extra row at n=15.** Statistically a tie. With binomial noise that wide, you cannot conclude reflexion "works."

**The retry loop never productively fixed an initial failure.** Look at `retry_lift` per-row: 14 rows landed at exactly 0.50 (neutral — first attempt and final attempt had the same pass/fail status). One row landed at 0.00 (hurt — perverse). Zero rows landed at 1.00 (helped). The 7pp aggregate improvement does NOT come from retries fixing wrong outputs. It comes from the first attempts of reflexion-mode runs happening to score one row better than single-pass-mode runs — pure temperature variance at 0.3.

**The self-critique was wrong on 60% of rows.** And the errors aren't random — they cluster by constraint type.

### Three buckets of self-critic behavior

**Bucket A: false-negative miscounting bias** — 8 of 15 rows

The model wrote a correct answer, self-criticized it as wrong, burned both retries trying to "fix" a non-bug, and sometimes ended up at the correct answer anyway by luck.

| Row | constraint | self_critique_accuracy |
|---|---|---|
| Moon, 10 words | ✅ | 0.00 (wrong on all 3 attempts) |
| Old dog, 8 words | ✅ | 0.00 |
| 7 European capitals starting with B | ✅ | 0.00 |
| Road trip without letter S | ✅ | 0.00 |
| Beach, all words start with S | ✅ | 0.00 |
| City skyline, 15 words | ✅ | 0.33 |
| Child's first steps, 7 words | ✅ | 0.33 |
| Chess, all words start with P | ✅ | 0.33 |

The model **can write** correct answers on these constraint types but **cannot verify** them. The self-critic prompt explicitly says *"count the words; off-by-one is still failure."* The model still miscounted, repeatedly, even with the explicit instruction.

**Bucket B: critic is right** — 6 rows

Either the task is genuinely hard at temp=0.3 and the model correctly says FAIL each attempt (12-word winter sunrise, no-A thunderstorm), or the task is easy and the model correctly says PASS on attempt 1 and stops the loop (4 chemical elements starting with C — used only 1 attempt total).

**Bucket C: the perverse case** — 1 row 🚨

| Row | single_pass | reflexion | retry_lift |
|---|---|---|---|
| Forest, all words start with F | ✅ | ❌ | **0.00 (HURT)** |

The model wrote an F-alliterative sentence on its first attempt — correct. Self-critique falsely flagged issues that weren't there. The retry produced a non-alliterative sentence. The final output was wrong. **The loop actively broke a correct answer.** This is the H3 perverse outcome in its purest form, and it shipped in production scale this would be the failure that matters most: a correct response replaced with an incorrect one by the verification step that was supposed to make things safer.

### The lesson

> **Reflexion is only as good as the self-critic. If the model can't accurately judge its own output, the retry loop is at best wasted tokens and at worst actively destructive.**

On this dataset, the self-critic had **a systematic false-negative bias on word-count, lipogram, and alliteration tasks** — the exact constraint types where the LLM's well-known weakness at character-level reasoning shows up. The critic shares the producer's blind spots. The retry budget never had a chance to be productive on those rows.

### Where this fits in the catalog's verification arc

This is the third agent in a row to circle the same gravity: post-write verification, in-loop reasoning, and self-critique-with-retry all FAIL in different but related ways. Same meta-lesson, three angles:

| Agent | Verification pattern | How it fails |
|---|---|---|
| **#5** multi-agent-pipeline | Post-write fact-checker | Out-of-scope: catches what the writer fabricated, can't see what the planner *forgot* upstream |
| **#6** react-agent | In-loop externalized reasoning | Commitment: externalized "Thought:" gives the model rhetorical cover to skip verification tools |
| **#7** reflexion-loop | Self-critique + retry | Calibration: critic shares the producer's miscounting bias; retries amplify wrong verdicts into wasted attempts (or actively-broken outputs) |

**One unified PM lesson for the verification arc:**

> **Adding more verification steps doesn't reliably add safety. A verifier that shares the producer's blind spots cannot catch the producer's errors. A verifier that introduces *new* failure modes can break things the producer got right.**

For AI PMs designing safety-critical agent loops in 2026, the corollary is uncomfortable: **the existence of a verification step is not evidence that verification is happening**. The eval framework has to actively measure whether the verifier is grounded in reality — `self_critique_accuracy` in this experiment, `tool_call_efficiency` in agent #6, the fact-checker firing rate in agent #5. Without those orthogonal metrics, an "Aggregate score went up by 7pp after we added reflexion" PM update on a production agent would be celebrating temperature noise while the loop quietly destroys one row out of fifteen.

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

✅ Complete. Headline aggregate looks like H1 (reflexion +7pp on constraint satisfaction) but the orthogonal evaluators reveal it's actually H3 dressed up in temperature noise: `retry_lift` was neutral on 14 rows and HURT on 1 (forest F alliteration row — the canary perverse case), `self_critique_accuracy` was 0.40 (wrong on 60% of rows), and the critic's errors clustered on constraint types where character-level reasoning is the LLM's well-known weakness (word counts, lipograms, alliteration). This completes a three-agent verification arc (#5, #6, #7) where each pattern circles the same gravity: **adding more verification doesn't reliably add safety if the verifier shares blind spots with the producer**.
