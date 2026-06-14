# 09 · Chain-of-Thought

A chain-of-thought agent on a small mixed-reasoning dataset. The variable under test is a single dial — **how much compute the model spends "thinking" before it answers** — across three modes:

- **direct** — answer immediately, no reasoning. 1 LLM call.
- **cot** — "think step by step," then answer. 1 LLM call.
- **self_consistency** — sample 5 CoT paths at a higher temperature and majority-vote the final answers. 5 LLM calls.

The PM question underneath isn't "does CoT help?" — it's "**does each added increment of compute pay for itself?**" CoT costs about the same as direct; self-consistency costs 5×. If the extra samples rarely change the answer, you are paying 5× for nothing. And there's a sharper trap: **majority voting measures *agreement*, not *correctness*** — a model that is confidently wrong votes wrong 5/5, and self-consistency will hand you that wrong answer wearing a 100%-confidence badge.

This is the first agent in the catalog's reasoning/control-flow tier to test the two most-cited prompting techniques in the field — and the headline is that **on a 2026 model at realistic difficulty, both are largely saturated.**

## The pipeline

```
direct / cot:
    answer (one llm pass)

self_consistency:
    sample_1 ┐
    sample_2 │
    sample_3 ├─> majority_vote ─> winning answer
    sample_4 │
    sample_5 ┘
```

Each node is a typed LangWatch span under the `chain_of_thought` workflow root: `answer` (`llm`) for direct/cot, or `sample_1..N` (`llm`) plus a `majority_vote` (`span`) for self-consistency. The vote span records every sample's answer and the agreement rate, so the trace shows at a glance whether the five samples actually disagreed — or just paid five times to say the same thing.

Both prompts force a trailing `Final answer: X` line, so a single extractor parses every mode identically; the only difference between modes is whether reasoning is allowed before that line. See [`agent.py`](agent.py) — ~220 lines, raw OpenAI + LangWatch, no agent framework.

## The dataset

18 questions ([`dataset.csv`](dataset.csv)) across four bands, each with one exact answer so all scoring is mechanical:

| Band | Example | Count | Why it's here |
|---|---|---|---|
| `easy` | *"A box holds 12 eggs. How many in 8 boxes?"* | 5 | One step. Everything should ace these — a floor. |
| `multistep` | *"Tom has 48 marbles, gives a third away and 5 more. How many left?"* | 7 | Several chained operations — the classic place CoT is supposed to help. |
| `logic` | *"Tom > Sue > Ann in height. Who is shortest?"* | 3 | Transitive / syllogistic reasoning, no arithmetic. |
| `trick` | *"A bat and ball cost \$1.10, the bat is \$1 more than the ball. How much is the ball?"* | 3 | Cognitive-Reflection-Test items where the *intuitive* answer is wrong (\$0.10, not \$0.05). The textbook showcase for CoT. |

## The evaluators

Three scorers ([`evals.py`](evals.py)) — **all programmatic** (same noise-free choice as agents #7 and #8):

| Evaluator | Type | Measures |
|---|---|---|
| `answer_correctness` | Programmatic | Did the extracted final answer match ground truth (numeric tolerance, or normalized text for names/yes-no)? All modes. |
| `lift` | Programmatic | Paired comparison between two modes on the same row: +1 if the baseline was wrong but the treatment right (extra compute helped), 0 no change, −1 if it hurt. Used twice: `direct→cot` (is reasoning worth it?) and `cot→self_consistency` (is sampling worth it?). |
| `sample_efficiency` | Programmatic | LLM calls per run — the cost axis (direct=1, cot=1, sc=5). |

The driver also reports **self-consistency vote agreement** — what fraction of the 5 samples agreed — because self-consistency can only change an answer on rows where the samples *disagree*.

## The tuning experiment

> **direct vs cot vs self_consistency on 18 mixed-reasoning questions**

### Three hypotheses to discriminate

| | Outcome | What it would teach |
|---|---|---|
| **H1** (textbook) | cot > direct, and self_consistency > cot | More inference compute buys more accuracy, monotonically. The 2022 papers' result. |
| **H2** (saturation / null) | cot ≈ direct and/or sc ≈ cot | A 2026 model already reasons internally, so prompting for it adds little; and with no uncertainty left, voting changes nothing. The extra compute is a tax. |
| **H3** (perverse) | cot < direct (overthinking) or sc < cot (samples agree on a *wrong* answer) | More compute makes it worse — verbalizing talks the model out of a right answer, or majority vote launders a systematic bias into false confidence. |

### Results

**Aggregate across 18 rows**

| mode | answer_correctness | mean llm_calls |
|---|---|---|
| `direct` | 0.89 (16/18) | 1.0 |
| `cot` | **1.00 (18/18)** | 1.0 |
| `self_consistency` | 1.00 (18/18) | **5.0** |

**Correctness by band**

| mode | easy | multistep | logic | trick |
|---|---|---|---|---|
| `direct` | 5/5 | **5/7** | 3/3 | **3/3** |
| `cot` | 5/5 | **7/7** | 3/3 | 3/3 |
| `self_consistency` | 5/5 | 7/7 | 3/3 | 3/3 |

**Paired lifts**

| comparison | helped | hurt | neutral | mean lift |
|---|---|---|---|---|
| `direct → cot` | 2 | 0 | 16 | 0.56 |
| `cot → self_consistency` | 0 | 0 | 18 | 0.50 |

**Self-consistency vote agreement:** mean **100%**; rows where the 5 samples were *not* unanimous: **0 / 18.**

### What's actually happening

This is **H2 in its purest form for self-consistency, with a narrow real CoT win on the side** — and no H3 this run.

**1. The CoT showcase band is saturated.** The three Cognitive-Reflection-Test questions — bat-and-ball, the widget-machines, the lily pads — are the canonical demonstration that "think step by step" rescues a model from its wrong gut answer. `direct` got all three **right** (3/3). gpt-4o-mini no longer falls for the intuitive trap, because it already reasons internally before producing the "immediate" answer. The headline 2022 demo for CoT doesn't reproduce on a 2026 model.

**2. CoT's entire measurable benefit was 2 rows — both multi-step arithmetic.** The only place "show your work" flipped a wrong direct answer to right was the marbles problem (a third of 48, then minus 5) and the tank problem (200 − 35×4). Not logic, not tricks: multi-step arithmetic, where the model can do each step but occasionally drops one when answering in a single shot. That's the real, narrow, durable home of CoT — +11pp aggregate, all of it concentrated in one band.

**3. Self-consistency was a pure 5× tax: it changed nothing on all 18 rows.** Helped 0, hurt 0. The reason is in the agreement number: **the 5 samples were unanimous on every single row (100% mean agreement, 0/18 contested).** Self-consistency can only change an answer where the samples *disagree* — and once CoT is on, this model wasn't uncertain about anything in this set. We paid five times to hold a vote with no contested ballots.

### The lesson

> **More inference compute only buys accuracy where the model is actually uncertain — and you have to measure that uncertainty, not assume it.**

- **CoT** earns its keep on the narrow band where the model *can* do the work but won't reliably do it in one shot (multi-step arithmetic here). It's ~free, so keep it on — but don't expect the dramatic paper-sized lifts; on a capable model most of that lift has been absorbed into the base model.
- **Self-consistency** only helps where the samples disagree. Its cost is fixed at N×; its benefit is zero unless there is genuine sampling uncertainty to resolve. The honest pre-flight check is to **measure the disagreement rate first** — if your samples already agree, self-consistency is a pure cost line with no upside.

And the trap that makes self-consistency *dangerous* rather than merely wasteful: **it measures agreement, not correctness.** On this set 100% agreement coincided with 100% correctness, so it was harmless. But the exact same machinery, faced with a question the model is *systematically* wrong about, would return five identical wrong answers and a 100% agreement score — laundering a bias into the appearance of confidence. A PM watching only the agreement metric would read that as "high confidence, ship it."

This is the reasoning-tier echo of the catalog's recurring finding. A bolted-on "improvement" mechanism only works under a precondition that must be *measured*, not assumed: the self-critic must be calibrated (#7), the replan monitor must be calibrated (#8), and **self-consistency's samples must actually disagree (#9)**. When the precondition fails, the mechanism degrades from "free safety" to "pure cost" to "false confidence."

## Quick start

```bash
cd agents/09-chain-of-thought
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY and LANGWATCH_API_KEY (Project API Key, type Service)

# One question, each mode
python agent.py "Tom has 48 marbles. He gives a third to his sister and 5 to his friend. How many does he have left?"
COT_MODE=cot python agent.py "Tom has 48 marbles. He gives a third to his sister and 5 to his friend. How many does he have left?"
COT_MODE=self_consistency python agent.py "A bat and a ball cost 1.10 dollars in total. The bat costs 1.00 dollar more than the ball. How much does the ball cost?"

# Full comparison
python run_eval.py                          # all three modes, all 18 rows
python run_eval.py --mode self_consistency  # one mode (also prints vote agreement)
python run_eval.py --limit 3                # smoke test
```

If you hit the macOS Xcode-Python SSL issue (`Errno 2` on `ssl.py`):

```bash
pip install certifi
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

## What you'll learn

How direct / CoT / self-consistency look as LangWatch spans — and specifically how to *see*, in the `majority_vote` span, whether your five samples actually disagreed or just cost you five times to agree. The PM-relevant takeaway is that the two most-cited prompting techniques are not free accuracy: CoT is nearly free and helps narrowly, self-consistency is expensive and helps only under measurable sampling uncertainty, and the metric that decides whether to pay for it (sample disagreement) is the one you have to look at before, not after.

## Status

✅ Complete. CoT lifts aggregate correctness +11pp (89% → 100%) but the entire gain is 2 multi-step-arithmetic rows; the CRT "trick" band that is CoT's textbook showcase was already saturated in `direct` (3/3). Self-consistency was a pure 5× tax — helped 0, hurt 0 across all 18 rows — because the samples were unanimous everywhere (100% mean agreement, 0/18 contested). Opens the reasoning tier's lesson: extra inference compute only buys accuracy where the model is genuinely uncertain, and self-consistency measures agreement, not correctness.
