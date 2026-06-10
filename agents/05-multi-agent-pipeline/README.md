# 05 · Multi-agent Pipeline

A planner-writer pipeline with an optional fact-checker that can trigger one revision pass. This is the smallest interesting multi-agent shape: three specialized workers that each do one thing, wired into a single trace so you can see exactly which worker caused which problem.

The experiment is one variable: **basic vs fact_checked mode**. Same model (gpt-4o-mini), same 15 historical/scientific questions, same planner and writer. The fact_checked mode adds a third worker that inspects the writer's prose against the planner's fact list and can demand a revision. Does the extra verification step actually catch hallucinations and omitted facts — or just burn tokens?

## The three workers

| Worker | Role | LangWatch span type |
|---|---|---|
| `planner` | Reads the question, extracts 3-7 key facts a good answer must cover. Outputs a bullet list, no prose. | `agent` |
| `writer` | Reads the question + planner's facts, writes a 2-4 sentence answer that uses every fact, invents none. | `agent` |
| `fact_checker` | (fact_checked mode only) Reads the planner's facts + writer's prose, flags any unsupported claims or omitted required facts. Replies `OK` or a list of issues. | `evaluation` |
| `writer_revision` | (fact_checked mode only, runs only when checker found issues) Rewrites the answer addressing the specific issues. | `agent` |

The fact_checker uses `type="evaluation"` rather than `agent` so LangWatch visually distinguishes verification work from generation work in the trace tree — useful when you're scanning a busy dashboard.

## The pipeline

```
basic mode:
    planner  →  writer

fact_checked mode:
    planner  →  writer  →  fact_checker  →  (writer_revision if issues found)
```

Each worker is one LLM call wrapped in a typed `langwatch.span`. The orchestrator (`run`) is decorated with `@langwatch.trace(name="multi_agent_pipeline")` so the workflow root span ties everything together.

See [`agent.py`](agent.py) — ~150 lines, raw OpenAI + LangWatch, no framework.

## The dataset

15 historical and scientific Q&A rows ([`dataset.csv`](dataset.csv)) with semicolon-separated expected facts per question. Domains span moon landings, Manhattan Project, DNA discovery, Berlin Wall, Sputnik, Sistine Chapel, penicillin, Treaty of Versailles, Hubble, Origin of Species, Wright brothers, Curiosity rover, Marie Curie, Chernobyl, ISS.

The questions are deliberately the kind where the writer can easily either (a) omit a labeled fact, or (b) confidently fabricate a date / name / number — i.e., the exact failure modes a fact-checker is supposed to catch.

## The evaluators

Three scorers ([`evals.py`](evals.py)):

| Evaluator | Type | Measures |
|---|---|---|
| `fact_coverage` | Programmatic (token match) | Fraction of expected facts present in the answer. Passes at ≥0.7. Catches the omitted-fact failure mode the writer is prone to. |
| `hallucination_judge` | LLM-as-judge (1-5 rubric) | Are there factual claims in the answer NOT supported by the labeled fact list? Focus on dates, names, numbers, places. Passes at ≥4. |
| `answer_quality` | LLM-as-judge (1-5 rubric) | Is the answer well-written and useful, independent of correctness? Surfaces whether adding a revision pass makes prose worse (over-cautious, padded). |

Only the two judges cost LLM calls. `fact_coverage` is pure code.

## The tuning experiment

> **basic vs fact_checked pipeline on 15 questions where the writer can easily hallucinate or omit facts**

The driver ([`run_eval.py`](run_eval.py)) runs every row twice — once basic, once fact_checked — and prints a side-by-side comparison plus a per-row detail view that marks which fact_checked rows triggered a revision pass.

### Results

🚧 _Pending — baseline run not yet executed. Update once the run is complete._

## Quick start

```bash
cd agents/05-multi-agent-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY and LANGWATCH_API_KEY (Project API Key, type Service)

# Smoke test on one query
python agent.py "When did the Apollo 11 mission land on the moon?"

# Same query, fact-checked mode
PIPELINE_MODE=fact_checked python agent.py "When did the Apollo 11 mission land on the moon?"

# Full comparison
python run_eval.py                  # both modes, all 15 rows
python run_eval.py --limit 3        # smoke test
python run_eval.py --mode fact_checked
```

If you hit the macOS Xcode-Python SSL issue (`Errno 2: No such file or directory` on `ssl.py`):

```bash
pip install certifi
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

## What you'll learn

How LangWatch's nested span tree turns "the multi-agent pipeline got the wrong answer" into "the planner missed fact X, so the writer never had a chance" — and how a per-worker eval scheme lets you ask whether adding a verification worker is actually worth the extra tokens, or just makes you feel safer.

## Status

🚧 Code shipped — baseline run pending. Once the baseline lands, this section will get the same honest-finding framing as the rest of the catalog: whether the fact-checker measurably reduces hallucinations + omissions, or whether the basic pipeline was already at ceiling on this dataset.
