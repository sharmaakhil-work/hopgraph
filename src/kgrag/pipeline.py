import time
from dataclasses import asdict
from datetime import date

from pydantic import BaseModel

from kgrag.answer.context import build_context
from kgrag.answer.generate import generate_answer
from kgrag.config import Settings
from kgrag.llm import LLM, CostMeter
from kgrag.models import Passage
from kgrag.retrieval.entity_link import EntityLinker
from kgrag.retrieval.graph_retriever import GraphResult, GraphRetriever
from kgrag.retrieval.router import Route, route
from kgrag.retrieval.vector_retriever import VectorRetriever
from kgrag.runlog import log_query

SNIPPET_CHARS = 400


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    ticker: str
    section: str
    filing_date: date | None
    snippet: str


class ClaimOut(BaseModel):
    text: str
    citations: list[str]


class QueryResult(BaseModel):
    question: str
    system: str
    route: str
    route_confidence: float | None
    route_forced: bool
    answerable: bool
    answer: str
    claims: list[ClaimOut]
    citations: list[Citation]
    graph_facts: list[str]
    graph_chains: list[str]
    graph_plans: list[dict]
    linked_entities: list[dict]
    retrieved_ids: list[str]
    citations_valid: bool
    regenerations: int
    outcome: str
    latency_ms: int
    cost_usd: float


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        anthropic_client,
        driver,
        pool,
        embedder,
        linker: EntityLinker,
        log: bool = True,
    ):
        self.s = settings
        self.client = anthropic_client
        self.driver = driver
        self.pool = pool
        self.vectors = VectorRetriever(pool, embedder, ef=settings.hnsw_ef_search)
        self.linker = linker
        self.log = log

    def answer(self, question: str, mode: str = "auto", system: str = "kgrag") -> QueryResult:
        t0 = time.perf_counter()
        llm = LLM(
            self.client, CostMeter(ledger_path=self.s.reports_dir / "spend.jsonl", label=f"query:{system}")
        )
        s = self.s

        decision_conf = None
        forced = mode != "auto"
        if forced:
            effective = Route(mode)
        else:
            decision, effective, _ = route(llm, s.router_model, question, s.router_confidence_threshold)
            decision_conf = decision.confidence

        graph = GraphResult()
        passages: list[Passage] = []
        outcome = "ok"
        if effective in (Route.GRAPH, Route.HYBRID):
            graph = GraphRetriever(self.driver, self.linker, llm, s.router_model, s.max_graph_facts).retrieve(
                question
            )
        if effective in (Route.VECTOR, Route.HYBRID):
            passages = self.vectors.search(question, s.vector_top_k)
        if effective == Route.HYBRID and graph.linked:
            # graph -> text: passages that mention the linked entities
            passages += self.vectors.passages_for_entities(
                question, [e.id for e in graph.linked], max(2, s.vector_top_k // 2)
            )
        if effective == Route.GRAPH and not graph.verbalized.facts:
            outcome = "graph_empty_fallback_vector"
            passages = self.vectors.search(question, s.vector_top_k)

        cited = self.vectors.fetch(graph.verbalized.chunk_ids)
        ctx = build_context(graph.verbalized, passages, cited)
        gen = generate_answer(llm, s.answer_model, s.answer_effort, question, ctx)
        if not gen.citations_valid:
            outcome = "citation_validation_failed"
        elif not gen.draft.answerable:
            outcome = "refused"

        cited_ids = list(dict.fromkeys(c for claim in gen.draft.claims for c in claim.citations))
        citations = [
            Citation(
                chunk_id=cid,
                doc_id=p.doc_id,
                ticker=p.ticker,
                section=p.section_path,
                filing_date=p.filing_date,
                snippet=p.text[:SNIPPET_CHARS],
            )
            for cid in cited_ids
            if (p := ctx.passages.get(cid))
        ]
        result = QueryResult(
            question=question,
            system=system,
            route=effective.value,
            route_confidence=decision_conf,
            route_forced=forced,
            answerable=gen.draft.answerable,
            answer=gen.text,
            claims=[ClaimOut(text=c.text, citations=c.citations) for c in gen.draft.claims],
            citations=citations,
            graph_facts=[f.text for f in graph.verbalized.facts],
            graph_chains=graph.verbalized.chains,
            graph_plans=[p.model_dump(mode="json") for p in graph.plans],
            linked_entities=[asdict(e) for e in graph.linked],
            retrieved_ids=sorted(ctx.allowed_ids),
            citations_valid=gen.citations_valid,
            regenerations=gen.regenerations,
            outcome=outcome,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            cost_usd=round(llm.meter.spent_usd, 6),
        )
        if self.log:
            log_query(self.pool, result.model_dump(mode="json"))
        return result
