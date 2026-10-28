import json
import statistics
from typing import Annotated

import typer

from kgrag.config import get_settings

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="Knowledge-graph + vector RAG over SEC filings"
)


def _llm(label: str, max_usd: float | None = None):
    import anthropic

    from kgrag.llm import LLM, CostMeter

    s = get_settings()
    return LLM(
        anthropic.Anthropic(),
        CostMeter(max_usd=max_usd, ledger_path=s.reports_dir / "spend.jsonl", label=label),
    )


def _docs_and_chunks(limit: int | None):
    from kgrag.corpus.edgar import load_documents
    from kgrag.ingest.chunk import chunk_document

    s = get_settings()
    docs = load_documents(s.raw_dir, limit)
    if not docs:
        raise typer.Exit("No filings in data/raw. Run `kgrag download` first.")
    chunks = [c for d in docs for c in chunk_document(d, s.chunk_chars, s.chunk_overlap)]
    return docs, chunks


def _embedder():
    from kgrag.embeddings import BgeEmbedder

    s = get_settings()
    return BgeEmbedder(s.embed_model, s.embed_dim)


def _observed_output_tokens() -> float | None:
    ledger = get_settings().reports_dir / "spend.jsonl"
    if not ledger.exists():
        return None
    outs = [r["output"] for line in ledger.open() if (r := json.loads(line))["purpose"] == "extract"]
    return statistics.mean(outs) if len(outs) >= 10 else None


@app.command()
def download(
    tickers: Annotated[str | None, typer.Option(help="Comma-separated tickers")] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N tickers")] = None,
):
    """Fetch the latest 10-K (+ Exhibit 21) and DEF 14A for each company from SEC EDGAR."""
    from kgrag.corpus.edgar import DEFAULT_TICKERS
    from kgrag.corpus.edgar import download as dl

    s = get_settings()
    chosen = tickers.split(",") if tickers else DEFAULT_TICKERS
    paths = dl(chosen[:limit] if limit else chosen, s.raw_dir, s.edgar_user_agent)
    typer.echo(f"{len(paths)} filings in {s.raw_dir}")


@app.command("reparse")
def reparse_cmd():
    """Re-split downloaded filings into sections from the cached text (no network)."""
    from kgrag.corpus.edgar import reparse

    for d in reparse(get_settings().raw_dir):
        typer.echo(
            f"{d.doc_id:28s} "
            + ", ".join(f"{s.path.split('/')[-1]}: {len(s.text) // 1000}k" for s in d.sections)
        )


@app.command()
def estimate(limit: Annotated[int | None, typer.Option(help="Only the first N filings")] = None):
    """Estimate Claude extraction cost for the corpus before spending anything."""
    from kgrag.ingest.cost import estimate as est

    s = get_settings()
    docs, chunks = _docs_and_chunks(limit)
    e = est(
        _llm("estimate"), s.extract_model, docs, chunks, output_tokens_per_chunk=_observed_output_tokens()
    )
    typer.echo(f"model {e.model}: {len(docs)} filings, {e.chunks} chunks")
    typer.echo(
        f"system prompt {e.system_tokens} tokens ({'cached' if e.system_cached else 'below cache minimum'}), "
        f"~{e.mean_user_tokens:.0f} input + {e.output_tokens_per_chunk:.0f} output tokens per chunk"
    )
    for doc_id, usd in sorted(e.per_doc.items()):
        typer.echo(f"  {doc_id:40s} ${usd:.3f}")
    typer.echo(
        f"TOTAL ${e.total_usd:.2f} (Batch API: ${e.batch_usd:.2f}); mean ${e.total_usd / len(docs):.3f}/filing"
    )


@app.command()
def ingest(
    limit: Annotated[int | None, typer.Option(help="Only the first N filings")] = None,
    max_usd: Annotated[float | None, typer.Option(help="Hard spend cap for this run")] = None,
    batch: Annotated[bool, typer.Option(help="Use the Message Batches API (50% cheaper, async)")] = False,
    workers: int = 4,
):
    """Chunk filings and extract entities/relationships with Claude (cached by document hash)."""
    from kgrag.ingest.chunk import write_chunks
    from kgrag.ingest.extract import ExtractionCache, Extractor

    s = get_settings()
    docs, chunks = _docs_and_chunks(limit)
    write_chunks(chunks, s.chunks_path)
    cap = max_usd if max_usd is not None else s.max_usd
    ex = Extractor(
        _llm("ingest", cap),
        s.extract_model,
        ExtractionCache(s.cache_dir, s.extract_model),
        s.reports_dir / "extract_failures.jsonl",
    )
    typer.echo(f"{len(docs)} filings, {len(chunks)} chunks, cap ${cap:.2f}")
    stats = ex.run_batch(docs, chunks) if batch else ex.run(docs, chunks, workers)
    typer.echo(
        f"cached {stats.cached}, extracted {stats.extracted}, failed {stats.failed}, spent ${stats.usd:.2f}"
    )


@app.command("tune-resolution")
def tune_resolution():
    """Pick the entity-resolution threshold on eval/resolution_pairs.jsonl."""
    from kgrag.corpus.edgar import load_ticker_map
    from kgrag.ingest.resolve import edgar_anchors, tune_threshold

    s = get_settings()
    pairs = [json.loads(line) for line in (s.eval_dir / "resolution_pairs.jsonl").open() if line.strip()]
    rows = tune_threshold(
        pairs,
        _embedder(),
        [round(0.80 + 0.01 * i, 2) for i in range(19)],
        edgar_anchors(load_ticker_map(s.raw_dir)),
    )
    typer.echo("threshold precision recall f1")
    for r in rows:
        typer.echo(
            f"{r['threshold']:.2f}      {r['precision']:.3f}     {r['recall']:.3f}  {r['f1']:.3f}"
            f"{'' if r['eligible'] else '  (precision < 0.95)'}"
        )
    best = max((r for r in rows if r["eligible"]), key=lambda r: (r["f1"], -r["threshold"]), default=None)
    if best:
        typer.echo(f"recommended RESOLUTION_THRESHOLD={best['threshold']:.2f}")
        s.reports_dir.mkdir(parents=True, exist_ok=True)
        (s.reports_dir / "resolution_tuning.json").write_text(
            json.dumps({"rows": rows, "best": best}, indent=1)
        )


@app.command()
def resolve(
    limit: Annotated[int | None, typer.Option()] = None,
    rebuild_graph: Annotated[bool, typer.Option(help="Delete the Neo4j graph before writing")] = False,
):
    """Resolve entities across chunks and write the graph to Neo4j (MERGE, idempotent)."""
    from kgrag.corpus.edgar import load_ticker_map
    from kgrag.db import neo4j_driver
    from kgrag.ingest.extract import ExtractionCache, load_extractions
    from kgrag.ingest.graph_writer import write_graph
    from kgrag.ingest.resolve import Resolver

    s = get_settings()
    docs, chunks = _docs_and_chunks(limit)
    extractions = load_extractions(ExtractionCache(s.cache_dir, s.extract_model), docs, chunks)
    typer.echo(f"{len(extractions)}/{len(chunks)} chunks have extractions")
    resolver = Resolver(_embedder(), s.resolution_threshold, load_ticker_map(s.raw_dir))
    res = resolver.resolve(extractions, chunks, {d.ticker: d.company_name for d in docs})
    res.save(s.resolution_path)
    typer.echo(json.dumps(res.stats, indent=1))
    with neo4j_driver(s) as driver:
        counts = write_graph(driver, res, chunks, rebuild=rebuild_graph)
    typer.echo(f"neo4j: {counts['entities']} entities, {counts['relations']} relationships")


@app.command()
def index(limit: Annotated[int | None, typer.Option()] = None):
    """Embed chunks and entity names into pgvector (HNSW)."""
    import time

    from kgrag.db import init_pg_schema, pg_pool
    from kgrag.ingest.resolve import Resolution
    from kgrag.ingest.vector_writer import index_chunks, index_entities

    s = get_settings()
    _, chunks = _docs_and_chunks(limit)
    res = Resolution.load(s.resolution_path) if s.resolution_path.exists() else None
    emb = _embedder()
    pool = pg_pool(s)
    init_pg_schema(pool, s.embed_dim)
    t0 = time.perf_counter()
    n = index_chunks(pool, emb, chunks, res)
    m = index_entities(pool, emb, res) if res else 0
    secs = time.perf_counter() - t0
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    (s.reports_dir / "index_stats.json").write_text(json.dumps({"chunks": n, "entities": m, "seconds": secs}))
    typer.echo(f"embedded {n} new chunks, {m} entities in {secs:.0f}s")


@app.command("eval-retrieval")
def eval_retrieval(
    draft: Annotated[int, typer.Option(help="Draft N labeled queries with Claude instead of evaluating")] = 0,
    allow_unverified: bool = False,
):
    """Recall@k across ef_search values on eval/retrieval_labels.jsonl."""
    from kgrag.db import pg_pool
    from kgrag.evaluation.retrieval import draft_labels, load_labels, recall_at_k, retrieval_report
    from kgrag.ingest.chunk import read_chunks
    from kgrag.retrieval.vector_retriever import VectorRetriever

    s = get_settings()
    if draft:
        out = s.eval_dir / "retrieval_labels.draft.jsonl"
        n = draft_labels(_llm("eval"), s.extract_model, read_chunks(s.chunks_path), draft, out)
        typer.echo(
            f"wrote {n} drafts to {out}. Review each, set verified=true, and save as retrieval_labels.jsonl"
        )
        return
    labels = load_labels(s.eval_dir / "retrieval_labels.jsonl", allow_unverified)
    if not labels:
        raise typer.Exit("No verified labels. Use --draft N first, or --allow-unverified for a smoke test.")
    emb = _embedder()
    rows = recall_at_k(VectorRetriever(pg_pool(s), emb), emb, labels)
    report = retrieval_report(rows, len(labels))
    (s.reports_dir / "retrieval.md").write_text(report)
    typer.echo(report)


@app.command("draft-questions")
def draft_questions_cmd(per_category: int = 15):
    """Draft the stratified benchmark set (needs human verification before use)."""
    from kgrag.db import neo4j_driver, pg_pool
    from kgrag.evaluation.questions import draft_questions

    s = get_settings()
    out = s.eval_dir / "questions.draft.jsonl"
    with neo4j_driver(s) as driver:
        counts = draft_questions(_llm("eval"), s.judge_model, driver, pg_pool(s), out, per_category)
    typer.echo(f"{counts} -> {out}. Verify each against its support_chunk_ids, then save as questions.jsonl")


def _pipeline():
    import anthropic

    from kgrag.db import neo4j_driver, pg_pool
    from kgrag.pipeline import Pipeline
    from kgrag.retrieval.entity_link import EntityLinker

    s = get_settings()
    pool = pg_pool(s)
    emb = _embedder()
    return Pipeline(
        s,
        anthropic.Anthropic(),
        neo4j_driver(s),
        pool,
        emb,
        EntityLinker.from_pg(pool, emb, s.link_threshold),
    )


@app.command()
def ask(question: str, mode: Annotated[str, typer.Option(help="auto|graph|vector|hybrid")] = "auto"):
    """Answer one question from the command line."""
    r = _pipeline().answer(question, mode=mode)
    typer.echo(
        f"route={r.route} conf={r.route_confidence} outcome={r.outcome} {r.latency_ms}ms ${r.cost_usd:.4f}\n"
    )
    typer.echo(r.answer)
    for c in r.citations:
        typer.echo(f"  [{c.chunk_id}] {c.ticker} {c.section}: {c.snippet[:120]}...")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run("kgrag.api.app:app", host=host, port=port)


@app.command()
def bench(
    limit: Annotated[int | None, typer.Option(help="First N questions (smoke test)")] = None,
    allow_unverified: bool = False,
):
    """Benchmark KG-RAG against the vector-only baseline; writes the report and updates the README."""
    from pathlib import Path

    from kgrag.corpus.edgar import load_documents
    from kgrag.evaluation import benchmark as b
    from kgrag.evaluation.questions import load_questions

    s = get_settings()
    path = s.eval_dir / ("questions.draft.jsonl" if allow_unverified else "questions.jsonl")
    questions = load_questions(path, allow_unverified)[:limit]
    if not questions:
        raise typer.Exit(f"No questions in {path}. Run `kgrag draft-questions` and verify them.")
    rows = b.run(_pipeline(), _llm("judge"), s.judge_model, questions, s.reports_dir / "benchmark.jsonl")
    summary = b.summarize(rows, b.ingestion_costs(s.reports_dir / "spend.jsonl"))
    (s.reports_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=1))
    table = b.markdown_table(summary)
    b.chart(summary, Path("docs/accuracy_by_hop.png"))
    b.update_readme(Path("README.md"), table, "docs/accuracy_by_hop.png", len(load_documents(s.raw_dir)))
    typer.echo(table)
