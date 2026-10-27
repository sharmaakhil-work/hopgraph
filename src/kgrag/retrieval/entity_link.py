"""Resolve entity mentions from a question to graph node ids."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from psycopg_pool import ConnectionPool

from kgrag.embeddings import Embedder
from kgrag.ingest.resolve import acronyms, looks_like_acronym, normalize

_SCAN_STOP = {"the", "company", "companies", "inc", "board", "market", "data", "who", "which", "what"}


@dataclass
class LinkedEntity:
    mention: str
    id: str
    name: str
    type: str
    method: str
    score: float


class EntityLinker:
    def __init__(self, entities: list[dict], vectors: np.ndarray, embedder: Embedder, threshold: float):
        self.entities = entities
        self.vectors = vectors
        self.embedder = embedder
        self.threshold = threshold
        self.by_norm: dict[str, list[int]] = defaultdict(list)
        self.by_acronym: dict[str, list[int]] = defaultdict(list)
        for i, e in enumerate(entities):
            for name in [e["name"], *e.get("aliases", [])]:
                self.by_norm[normalize(name)].append(i)
                for a in acronyms(name):
                    self.by_acronym[a].append(i)

    @classmethod
    def from_pg(cls, pool: ConnectionPool, embedder: Embedder, threshold: float) -> "EntityLinker":
        with pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, type, name, aliases, mentions, embedding FROM entities"
            ).fetchall()
        entities = [
            {"id": r[0], "type": r[1], "name": r[2], "aliases": list(r[3]), "mentions": r[4]} for r in rows
        ]
        vectors = np.stack([np.asarray(r[5], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 1))
        return cls(entities, vectors, embedder, threshold)

    def _pick(self, idxs: list[int], type_hint: str | None) -> int:
        cands = list(dict.fromkeys(idxs))
        if type_hint:
            typed = [i for i in cands if self.entities[i]["type"] == type_hint]
            cands = typed or cands
        # prefer companies for ambiguous names, then the most-mentioned entity
        return max(
            cands, key=lambda i: (self.entities[i]["type"] == "Company", self.entities[i].get("mentions", 0))
        )

    def _result(self, mention: str, i: int, method: str, score: float) -> LinkedEntity:
        e = self.entities[i]
        return LinkedEntity(
            mention=mention, id=e["id"], name=e["name"], type=e["type"], method=method, score=score
        )

    def link(self, mention: str, type_hint: str | None = None) -> LinkedEntity | None:
        if not mention or not self.entities:
            return None
        norm = normalize(mention)
        if norm in self.by_norm:
            return self._result(mention, self._pick(self.by_norm[norm], type_hint), "alias", 1.0)
        if looks_like_acronym(mention) and mention.lower() in self.by_acronym:
            return self._result(
                mention, self._pick(self.by_acronym[mention.lower()], type_hint), "acronym", 1.0
            )
        q = self.embedder.embed_names([norm])[0]
        sims = self.vectors @ q
        if type_hint:
            mask = np.array([e["type"] == type_hint for e in self.entities])
            if mask.any():
                sims = np.where(mask, sims, -1.0)
        best = int(np.argmax(sims))
        if sims[best] >= self.threshold:
            return self._result(mention, best, "embedding", float(sims[best]))
        return None

    def scan(self, question: str, max_ngram: int = 6) -> list[LinkedEntity]:
        """Find known names/aliases appearing verbatim in the question (longest match wins)."""
        toks = normalize(question).split()
        found: list[LinkedEntity] = []
        used = [False] * len(toks)
        for n in range(min(max_ngram, len(toks)), 0, -1):
            for s in range(len(toks) - n + 1):
                if any(used[s : s + n]):
                    continue
                gram = " ".join(toks[s : s + n])
                if gram in _SCAN_STOP or len(gram) < 3 or gram not in self.by_norm:
                    continue
                found.append(self._result(gram, self._pick(self.by_norm[gram], None), "scan", 1.0))
                used[s : s + n] = [True] * n
        return found
