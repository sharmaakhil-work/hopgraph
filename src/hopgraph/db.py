from contextlib import contextmanager

import psycopg
from neo4j import Driver, GraphDatabase
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from hopgraph.config import Settings

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id             text PRIMARY KEY,
    doc_id         text NOT NULL,
    company_ticker text NOT NULL,
    form           text NOT NULL,
    section_path   text NOT NULL,
    filing_date    date,
    text           text NOT NULL,
    entity_ids     text[] NOT NULL DEFAULT '{{}}',
    embedding      vector({dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_entity_ids ON chunks USING gin (entity_ids);

CREATE TABLE IF NOT EXISTS entities (
    id        text PRIMARY KEY,
    type      text NOT NULL,
    name      text NOT NULL,
    aliases   text[] NOT NULL DEFAULT '{{}}',
    mentions  int NOT NULL DEFAULT 0,
    embedding vector({dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS entities_embedding_hnsw ON entities
    USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS query_log (
    id              bigserial PRIMARY KEY,
    ts              timestamptz NOT NULL DEFAULT now(),
    system          text NOT NULL,
    question        text NOT NULL,
    route           text,
    route_confidence real,
    route_forced    boolean NOT NULL DEFAULT false,
    graph_plans     jsonb,
    linked_entities jsonb,
    retrieved_ids   text[],
    answerable      boolean,
    citations_valid boolean,
    regenerations   int,
    latency_ms      int,
    cost_usd        real,
    outcome         text,
    answer          text
);
"""


def pg_pool(settings: Settings) -> ConnectionPool:
    ensure_extension(settings)
    return ConnectionPool(settings.pg_dsn, configure=register_vector, min_size=1, max_size=8, open=True)


def ensure_extension(settings: Settings) -> None:
    # register_vector needs the extension to exist before any pooled connection is configured
    with psycopg.connect(settings.pg_dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


def init_pg_schema(pool: ConnectionPool, dim: int) -> None:
    with pool.connection() as conn:
        conn.execute(SCHEMA_SQL.format(dim=dim))


def neo4j_driver(settings: Settings) -> Driver:
    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))


@contextmanager
def ef_search(conn: psycopg.Connection, value: int | None, exact: bool = False):
    """Per-transaction HNSW knobs. `exact=True` forces a sequential scan for ground-truth kNN."""
    with conn.transaction():
        if exact:
            conn.execute("SET LOCAL enable_indexscan = off")
        elif value is not None:
            conn.execute(f"SET LOCAL hnsw.ef_search = {int(value)}")
        yield conn
