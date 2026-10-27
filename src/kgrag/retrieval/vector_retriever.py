from psycopg_pool import ConnectionPool

from kgrag.db import ef_search
from kgrag.embeddings import Embedder
from kgrag.models import Passage

_COLS = "id, doc_id, company_ticker, form, section_path, filing_date, text, entity_ids"


def _passage(row, score: float) -> Passage:
    return Passage(
        chunk_id=row[0],
        doc_id=row[1],
        ticker=row[2],
        form=row[3],
        section_path=row[4],
        filing_date=row[5],
        text=row[6],
        entity_ids=list(row[7] or []),
        score=score,
    )


class VectorRetriever:
    def __init__(self, pool: ConnectionPool, embedder: Embedder, ef: int = 80):
        self.pool = pool
        self.embedder = embedder
        self.ef = ef

    def search(self, query: str, k: int, ef: int | None = None, exact: bool = False) -> list[Passage]:
        qv = self.embedder.embed_query(query)
        with self.pool.connection() as conn, ef_search(conn, ef or self.ef, exact=exact):
            rows = conn.execute(
                f"SELECT {_COLS}, 1 - (embedding <=> %s) AS score FROM chunks ORDER BY embedding <=> %s LIMIT %s",
                (qv, qv, k),
            ).fetchall()
        return [_passage(r[:8], float(r[8])) for r in rows]

    def search_ids(self, query_vec, k: int, ef: int | None = None, exact: bool = False) -> list[str]:
        with self.pool.connection() as conn, ef_search(conn, ef or self.ef, exact=exact):
            rows = conn.execute(
                "SELECT id FROM chunks ORDER BY embedding <=> %s LIMIT %s", (query_vec, k)
            ).fetchall()
        return [r[0] for r in rows]

    def passages_for_entities(self, query: str, entity_ids: list[str], k: int) -> list[Passage]:
        """Graph -> text: passages that mention any of the given entities, ranked by similarity to the query."""
        if not entity_ids:
            return []
        qv = self.embedder.embed_query(query)
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"SELECT {_COLS}, 1 - (embedding <=> %s) AS score FROM chunks WHERE entity_ids && %s "
                "ORDER BY embedding <=> %s LIMIT %s",
                (qv, entity_ids, qv, k),
            ).fetchall()
        return [_passage(r[:8], float(r[8])) for r in rows]

    def fetch(self, chunk_ids: list[str]) -> dict[str, Passage]:
        if not chunk_ids:
            return {}
        with self.pool.connection() as conn:
            rows = conn.execute(f"SELECT {_COLS} FROM chunks WHERE id = ANY(%s)", (chunk_ids,)).fetchall()
        return {r[0]: _passage(r, 0.0) for r in rows}
