"""
Simple RAG agent — the most basic retrieve-then-generate pattern.

Pipeline:
    chunk corpus  ->  embed chunks  ->  embed query  ->  retrieve top-k  ->  synthesize

Every step is wrapped in a LangWatch span so the resulting trace tells the
whole story end-to-end (per-step latency, tokens, retrieved chunks, model
choice). No frameworks. ~150 lines of code, copy-paste-friendly.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration knobs (the tuning experiment changes these) ----

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))     # tokens-ish (we approximate via words)
TOP_K = int(os.getenv("TOP_K", "3"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
SYNTH_MODEL = os.getenv("SYNTH_MODEL", "gpt-4o-mini")
SERVICE_NAME = "simple-rag"

# ---- Setup ----

langwatch.setup(
    api_key=os.environ["LANGWATCH_API_KEY"],
    service_name=SERVICE_NAME,
)

client = OpenAI()


@dataclass
class Chunk:
    """A retrievable unit of the corpus."""

    source: str   # the entry title (e.g. "ReAct") — useful for retrieval evals
    text: str


@dataclass
class RagResult:
    """What one query returns. Mirrors the trace's top-level output."""

    question: str
    answer: str
    retrieved: list[dict]   # [{source, text, score}]
    config: dict             # the knobs in effect at query time


# ---- Corpus loading + chunking ----

def load_corpus(path: str = "corpus.md") -> list[Chunk]:
    """
    Parse corpus.md into one Chunk per `## ` section. The markdown structure
    is the canonical source of truth for entry boundaries.
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()
    # Split on level-2 headings, dropping the preamble before the first one.
    sections = re.split(r"\n## ", content)[1:]
    chunks: list[Chunk] = []
    for section in sections:
        title, _, body = section.partition("\n")
        chunks.append(Chunk(source=title.strip(), text=body.strip()))
    return chunks


def rechunk(chunks: list[Chunk], chunk_size: int) -> list[Chunk]:
    """
    Re-split each entry into fixed-size word chunks. Word-based (not token-based)
    chunking keeps the example dependency-light. Each subchunk preserves its
    source title so retrieval evaluation can check provenance.
    """
    out: list[Chunk] = []
    for chunk in chunks:
        words = chunk.text.split()
        if len(words) <= chunk_size:
            out.append(chunk)
            continue
        for i in range(0, len(words), chunk_size):
            window = " ".join(words[i : i + chunk_size])
            out.append(Chunk(source=chunk.source, text=window))
    return out


# ---- The RAG pipeline ----

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@langwatch.trace(name="simple_rag_query")
def query(
    question: str,
    corpus: list[Chunk],
    *,
    chunk_size: int = CHUNK_SIZE,
    top_k: int = TOP_K,
) -> RagResult:
    """
    Run one query end-to-end.

    Each pipeline step is a LangWatch span so the trace shows the full timeline.
    The root span receives the question as input and the final answer as output;
    children record per-step inputs, outputs, and token usage.
    """
    root = langwatch.get_current_trace().root_span
    root.update(input=question, metadata={"chunk_size": chunk_size, "top_k": top_k})

    # 1. Re-chunk at the requested size. This is data prep, not an LLM call.
    with langwatch.span(name="chunk_corpus", type="span") as s:
        chunks = rechunk(corpus, chunk_size)
        s.update(
            output=f"{len(chunks)} chunks at size {chunk_size}",
            metadata={"chunk_count": len(chunks)},
        )

    # 2. Embed every chunk. This is expensive — in production you'd cache.
    with langwatch.span(name="embed_corpus", type="rag") as s:
        s.update(input=f"{len(chunks)} chunks", metadata={"model": EMBED_MODEL})
        chunk_texts = [c.text for c in chunks]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk_texts)
        chunk_embs = np.array([e.embedding for e in resp.data])
        s.update(
            metrics={
                "prompt_tokens": resp.usage.prompt_tokens,
                "total_tokens": resp.usage.total_tokens,
            },
        )

    # 3. Embed the question.
    with langwatch.span(name="embed_query", type="rag") as s:
        s.update(input=question, metadata={"model": EMBED_MODEL})
        q_resp = client.embeddings.create(model=EMBED_MODEL, input=question)
        q_emb = np.array(q_resp.data[0].embedding)
        s.update(
            metrics={
                "prompt_tokens": q_resp.usage.prompt_tokens,
                "total_tokens": q_resp.usage.total_tokens,
            },
        )

    # 4. Cosine similarity → top-k.
    with langwatch.span(name="retrieve", type="rag") as s:
        scores = np.array([_cosine(q_emb, ce) for ce in chunk_embs])
        top_idx = np.argsort(-scores)[:top_k]
        retrieved = [
            {
                "source": chunks[i].source,
                "text": chunks[i].text,
                "score": float(scores[i]),
            }
            for i in top_idx
        ]
        s.update(
            output=[{"source": r["source"], "score": round(r["score"], 3)} for r in retrieved],
            metadata={"top_k": top_k, "candidate_count": len(chunks)},
        )

    # 5. Stuff retrieved chunks into the prompt and synthesize.
    context = "\n\n".join(f"[{r['source']}]\n{r['text']}" for r in retrieved)
    synth_prompt = (
        "Answer the question using only the context below. "
        "If the answer is not present, say 'I don't have enough information.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    with langwatch.span(name="synthesize", type="llm") as s:
        s.update(input=question, metadata={"model": SYNTH_MODEL})
        completion = client.chat.completions.create(
            model=SYNTH_MODEL,
            messages=[{"role": "user", "content": synth_prompt}],
            temperature=0,
        )
        answer = completion.choices[0].message.content or ""
        s.update(
            output=answer,
            metrics={
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            },
        )

    root.update(output=answer)

    return RagResult(
        question=question,
        answer=answer,
        retrieved=retrieved,
        config={"chunk_size": chunk_size, "top_k": top_k, "embed_model": EMBED_MODEL, "synth_model": SYNTH_MODEL},
    )


# ---- CLI ----

if __name__ == "__main__":
    import sys

    corpus = load_corpus()
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What does ReAct stand for?"
    result = query(question, corpus)
    print(f"\nQuestion: {result.question}\n")
    print(f"Answer: {result.answer}\n")
    print(f"Retrieved (top {len(result.retrieved)}):")
    for r in result.retrieved:
        print(f"  - {r['source']} (score {r['score']:.3f})")
    print(f"\nConfig: {result.config}")
