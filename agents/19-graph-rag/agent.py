"""
Graph-RAG agent — retrieve over a knowledge graph by traversing relationships,
instead of keyword-matching flat passages.

Switch via env var:
    GRAPH_MODE=flat   (default) — keyword top-k over the passages, then generate.
                                  (Baseline RAG: similarity over independent chunks.)
    GRAPH_MODE=graph            — link the query to graph entities, traverse their
                                  connected subgraph (and, for global questions, the
                                  whole graph), serialize it, then generate.

Both modes feed the generator text and make ONE LLM call; the only difference is
how the context is assembled. Flat retrieval is capped at k passages chosen by
surface overlap. Graph traversal assembles the connected, complete subgraph the
question touches — which is the only way to answer questions whose answer depends
on the graph's *structure*: multi-hop chains, and global counts that span more
edges than any top-k could hold.

Trace tree (typed LangWatch spans):

    graph_rag (workflow root)
    ├─ flat:   retrieve (rag, keyword top-k)        -> generate (llm)
    └─ graph:  traverse (rag, entity-linked subgraph) -> generate (llm)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration ----

GRAPH_MODE = os.getenv("GRAPH_MODE", "flat")  # "flat" | "graph"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
TOP_K = int(os.getenv("TOP_K", "3"))
HOPS = int(os.getenv("HOPS", "3"))            # graph traversal radius from seed nodes

os.environ.setdefault("OTEL_SERVICE_NAME", "graph-rag")
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- Corpus + graph ----

def _load_passages(filename: str) -> list[str]:
    text = (Path(__file__).parent / filename).read_text(encoding="utf-8")
    return [line[2:].strip() for line in text.splitlines() if line.strip().startswith("- ")]


CORPUS = _load_passages("corpus.md")


def _parse_edges(passages: list[str]) -> list[tuple[str, str, str]]:
    """Turn each passage into a (subject, relation, object) edge."""
    edges = []
    for p in passages:
        t = p.rstrip(".")
        if m := re.match(r"(.+?) belongs to the (.+)", t):
            edges.append((m.group(1), "belongs to", m.group(2)))
        elif m := re.match(r"The (.+?) is located in (.+)", t):
            edges.append((m.group(1), "is located in", m.group(2)))
        elif m := re.match(r"(.+?) is governed by the (.+)", t):
            edges.append((m.group(1), "is governed by", m.group(2)))
        elif m := re.match(r"(.+?) specializes in (.+)", t):
            edges.append((m.group(1), "specializes in", m.group(2)))
    return edges


EDGES = _parse_edges(CORPUS)
NODES = sorted({e[0] for e in EDGES} | {e[2] for e in EDGES}, key=len, reverse=True)  # longest first


def _edge_text(e: tuple[str, str, str]) -> str:
    s, r, o = e
    article = "the " if r in ("belongs to", "is governed by") else ""
    return f"{s} {r} {article}{o}."


# ---- Flat retrieval (keyword overlap with light stemming) ----

_STOP = set("the a an of to in on and or is are was were be by for with at from as it its what which "
            "where who how many does do you your i can that this these those study work scholar city body".split())


def _stem(w: str) -> str:
    return w[:-1] if w.endswith("s") and len(w) > 3 else w


def _tokens(s: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"\w+", s.lower()) if w not in _STOP}


def retrieve_flat(query: str, k: int = TOP_K) -> list[str]:
    q = _tokens(query)
    scored = [(len(q & _tokens(p)), p) for p in CORPUS]
    scored = [(s, p) for s, p in scored if s > 0]
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:k]]


# ---- Graph traversal ----

def _seed_nodes(query: str) -> list[str]:
    """Entities named in the query (longest-match first to prefer 'Vellum Academy' over 'Vellum')."""
    found, ql = [], query.lower()
    for n in NODES:
        if n.lower() in ql and not any(n.lower() in f.lower() and n != f for f in found):
            found.append(n)
    return found


def _neighborhood(seeds: list[str], hops: int) -> list[tuple[str, str, str]]:
    """All edges within `hops` of any seed node (BFS over the undirected graph)."""
    frontier = set(seeds)
    seen_nodes = set(seeds)
    collected: list[tuple[str, str, str]] = []
    for _ in range(hops):
        nxt = set()
        for e in EDGES:
            s, r, o = e
            if s in frontier or o in frontier:
                if e not in collected:
                    collected.append(e)
                for endpoint in (s, o):
                    if endpoint not in seen_nodes:
                        seen_nodes.add(endpoint); nxt.add(endpoint)
        if not nxt:
            break
        frontier = nxt
    return collected


def traverse_graph(query: str) -> tuple[list[str], str]:
    """Return (serialized passages, scope) for the query's relevant subgraph."""
    seeds = _seed_nodes(query)
    if seeds:
        edges = _neighborhood(seeds, HOPS)
        scope = f"subgraph around {seeds}"
    else:
        edges = list(EDGES)      # global question with no named entity -> whole graph
        scope = "whole graph"
    return [_edge_text(e) for e in edges], scope


# ---- LLM ----

def _chat(system: str, user: str) -> tuple[str, int, int]:
    c = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=TEMPERATURE,
    )
    return (c.choices[0].message.content or "").strip(), c.usage.prompt_tokens, c.usage.completion_tokens


_GEN_SYSTEM = (
    "Answer the question using ONLY the provided facts. For 'how many' questions, count carefully "
    "and reply with just the number. Otherwise reply with a short phrase or name. If the facts do "
    "not contain the answer, reply exactly: I don't know."
)


def _generate(query: str, passages: list[str]) -> tuple[str, int, int]:
    ctx = "\n".join(f"- {p}" for p in passages) if passages else "(no facts)"
    return _chat(_GEN_SYSTEM, f"FACTS:\n{ctx}\n\nQUESTION: {query}")


# ---- Result type ----

@dataclass
class GraphResult:
    query: str
    mode: str
    answer: str = ""
    scope: str = ""
    llm_calls: int = 0
    retrieved: list[str] = field(default_factory=list)


# ---- The agent ----

@langwatch.trace(name="graph_rag")
def run(query: str, *, mode: str = GRAPH_MODE) -> GraphResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {query}")
    result = GraphResult(query=query, mode=mode)

    if mode == "flat":
        with langwatch.span(name="retrieve", type="rag") as s:
            s.update(input=query)
            passages = retrieve_flat(query)
            s.update(output="\n".join(passages)[:500])
        result.retrieved = passages
        result.scope = f"top-{TOP_K} passages"
    elif mode == "graph":
        with langwatch.span(name="traverse", type="rag") as s:
            s.update(input=query)
            passages, scope = traverse_graph(query)
            s.update(output=f"[{scope}, {len(passages)} edges]\n" + "\n".join(passages)[:500])
        result.retrieved = passages
        result.scope = scope
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    with langwatch.span(name="generate", type="llm") as s:
        s.update(input=query)
        ans, pt, ct = _generate(query, result.retrieved)
        s.update(output=ans[:300], metrics={"prompt_tokens": pt, "completion_tokens": ct})
    result.llm_calls += 1
    result.answer = ans

    root.update(output=f"[{mode} {len(result.retrieved)} facts] {ans[:140]}")
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How many scholars belong to the Vellum Academy?"
    mode = os.getenv("GRAPH_MODE", "flat")
    print(f"\n=== Query: {q}\n=== Mode: {mode}\n")
    res = run(q, mode=mode)
    print(f"Context ({res.scope}, {len(res.retrieved)} facts):")
    for p in res.retrieved:
        print(f"  - {p}")
    print(f"\nAnswer: {res.answer}")
    print(f"=== llm calls: {res.llm_calls}")
