"""Embed chunks and entity names into pgvector. Chunk ids are the same keys used in Neo4j."""

from psycopg_pool import ConnectionPool

from hopgraph.embeddings import Embedder
from hopgraph.ingest.resolve import Resolution, normalize
from hopgraph.models import Chunk

BATCH = 128


def index_chunks(
    pool: ConnectionPool, embedder: Embedder, chunks: list[Chunk], resolution: Resolution | None
) -> int:
    mentions = resolution.mentions if resolution else {}
    with pool.connection() as conn:
        existing = {r[0] for r in conn.execute("SELECT id FROM chunks").fetchall()}
    todo = [c for c in chunks if c.id not in existing]
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        vecs = embedder.embed_passages([c.text for c in batch])
        with pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (id, doc_id, company_ticker, form, section_path, filing_date, text, entity_ids, "
                "embedding) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO UPDATE SET "
                "text = EXCLUDED.text, entity_ids = EXCLUDED.entity_ids, embedding = EXCLUDED.embedding",
                [
                    (
                        c.id,
                        c.doc_id,
                        c.ticker,
                        c.form,
                        c.section_path,
                        c.filing_date,
                        c.text,
                        mentions.get(c.id, []),
                        v,
                    )
                    for c, v in zip(batch, vecs, strict=True)
                ],
            )
        print(f"  embedded {min(i + BATCH, len(todo))}/{len(todo)} chunks")
    # entity mentions can change after re-resolution without the chunk text changing
    if mentions:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "UPDATE chunks SET entity_ids = %s WHERE id = %s",
                [(ids, cid) for cid, ids in mentions.items()],
            )
    return len(todo)


def index_entities(pool: ConnectionPool, embedder: Embedder, resolution: Resolution) -> int:
    ents = resolution.entities
    for i in range(0, len(ents), 512):
        batch = ents[i : i + 512]
        vecs = embedder.embed_names([normalize(e["name"]) for e in batch])
        with pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO entities (id, type, name, aliases, mentions, embedding) VALUES (%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, aliases = EXCLUDED.aliases, "
                "type = EXCLUDED.type, mentions = EXCLUDED.mentions, embedding = EXCLUDED.embedding",
                [
                    (e["id"], e["type"], e["name"], e["aliases"], e["mentions"], v)
                    for e, v in zip(batch, vecs, strict=True)
                ],
            )
    return len(ents)
