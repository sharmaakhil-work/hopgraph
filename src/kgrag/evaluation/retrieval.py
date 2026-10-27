"""Phase 2 gate: measure vector retrieval before building anything on top of it."""

import json
import random
import statistics
import time
from pathlib import Path

from pydantic import BaseModel

from kgrag.embeddings import Embedder
from kgrag.llm import LLM
from kgrag.models import Chunk
from kgrag.retrieval.vector_retriever import VectorRetriever


class DraftQuery(BaseModel):
    question: str
    self_contained: bool


LABEL_SYSTEM = """You write evaluation queries for a retrieval system over SEC filings. Given one chunk, write \
one question a financial analyst might ask that this chunk answers. Name the company explicitly. Don't copy \
phrases verbatim; paraphrase so lexical overlap alone doesn't find it. Set self_contained=false if the chunk \
is boilerplate, a table fragment, or can't answer a specific question."""


def draft_labels(llm: LLM, model: str, chunks: list[Chunk], n: int, out: Path, seed: int = 0) -> int:
    rng = random.Random(seed)
    pool = [c for c in chunks if len(c.text) > 1200]
    written = 0
    with out.open("w") as f:
        for c in rng.sample(pool, min(n * 2, len(pool))):
            q, _ = llm.structured(
                model=model,
                system=LABEL_SYSTEM,
                output=DraftQuery,
                purpose="draft_retrieval_labels",
                user=f"Company: {c.ticker} ({c.form}, {c.section_path})\n\n{c.text}",
                max_tokens=300,
                cache_system=False,
            )
            if not q.self_contained:
                continue
            f.write(json.dumps({"query": q.question, "relevant": [c.id], "verified": False}) + "\n")
            written += 1
            if written >= n:
                break
    return written


def load_labels(path: Path, allow_unverified: bool = False) -> list[dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    return [r for r in rows if allow_unverified or r.get("verified")]


def recall_at_k(
    retriever: VectorRetriever,
    embedder: Embedder,
    labels: list[dict],
    ks: tuple[int, ...] = (5, 10, 20),
    efs: tuple[int, ...] = (20, 40, 80, 160),
) -> list[dict]:
    """Relevance recall@k against labels, plus ANN recall (HNSW top-k vs exact top-k) per ef_search."""
    kmax = max(ks)
    qvecs = [embedder.embed_query(lab["query"]) for lab in labels]
    exact = [retriever.search_ids(v, kmax, exact=True) for v in qvecs]
    rows = []
    for ef in (None, *efs):
        recalls = {k: [] for k in ks}
        ann = {k: [] for k in ks}
        lat = []
        for lab, v, ex in zip(labels, qvecs, exact, strict=True):
            t0 = time.perf_counter()
            ids = ex if ef is None else retriever.search_ids(v, kmax, ef=ef)
            lat.append((time.perf_counter() - t0) * 1000)
            rel = set(lab["relevant"])
            for k in ks:
                recalls[k].append(len(set(ids[:k]) & rel) / len(rel))
                ann[k].append(len(set(ids[:k]) & set(ex[:k])) / k)
        rows.append(
            {
                "ef_search": "exact" if ef is None else ef,
                **{f"recall@{k}": statistics.mean(recalls[k]) for k in ks},
                **{f"ann_recall@{k}": statistics.mean(ann[k]) for k in ks},
                "p50_ms": statistics.median(lat) if ef is not None else None,
            }
        )
    return rows


def retrieval_report(rows: list[dict], n_labels: int, ks: tuple[int, ...] = (5, 10, 20)) -> str:
    head = ["ef_search", *[f"recall@{k}" for k in ks], *[f"ann_recall@{k}" for k in ks], "p50 ms"]
    lines = [
        f"# Vector retrieval ({n_labels} labeled queries)",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for r in rows:
        cells = [
            str(r["ef_search"]),
            *[f"{r[f'recall@{k}']:.3f}" for k in ks],
            *[f"{r[f'ann_recall@{k}']:.3f}" for k in ks],
            "-" if r["p50_ms"] is None else f"{r['p50_ms']:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "`exact` is a sequential scan (ground-truth kNN). `ann_recall` is the overlap of HNSW top-k "
        "with exact top-k; `recall` is against the labeled relevant chunks.",
    ]
    return "\n".join(lines)
