from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from hopgraph.pipeline import Pipeline, QueryResult


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Literal["auto", "graph", "vector", "hybrid"] = "auto"


def build_pipeline() -> Pipeline:
    import anthropic

    from hopgraph.config import get_settings
    from hopgraph.db import neo4j_driver, pg_pool
    from hopgraph.embeddings import BgeEmbedder
    from hopgraph.retrieval.entity_link import EntityLinker

    s = get_settings()
    pool = pg_pool(s)
    embedder = BgeEmbedder(s.embed_model, s.embed_dim)
    linker = EntityLinker.from_pg(pool, embedder, s.link_threshold)
    return Pipeline(s, anthropic.Anthropic(), neo4j_driver(s), pool, embedder, linker)


def create_app(pipeline_factory=build_pipeline) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline_factory()
        yield

    app = FastAPI(title="hopgraph", version="0.1.0", lifespan=lifespan)

    def get_pipeline(request: Request) -> Pipeline:
        return request.app.state.pipeline

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResult)
    async def query(body: QueryRequest, pipeline: Pipeline = Depends(get_pipeline)) -> QueryResult:
        return await run_in_threadpool(pipeline.answer, body.question, body.mode)

    @app.get("/chunks/{chunk_id}")
    def chunk(chunk_id: str, pipeline: Pipeline = Depends(get_pipeline)) -> dict:
        found = pipeline.vectors.fetch([chunk_id])
        if chunk_id not in found:
            raise HTTPException(404, "chunk not found")
        return found[chunk_id].model_dump(mode="json")

    return app


app = create_app()
