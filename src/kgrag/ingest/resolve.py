"""Entity resolution: collapse "Taiwan Semiconductor Manufacturing Company Limited", "TSMC" and "Taiwan
Semiconductor Manufacturing Co., Ltd." into one node with an alias list.

Merge rules, cheapest and safest first:
  1. identical normalized name (case, punctuation, legal suffixes removed)
  2. same EDGAR ticker anchor
  3. acronym of the other name ("TSMC", "U.S.")
  4. descriptor rule: same name plus generic words merges ("Cadence Design Systems"); plus distinguishing
     words blocks, even against a high embedding score ("Intel Foundry", "Lam Research International")
  5. embedding cosine >= threshold within the same type (Persons also need a shared company context)
"""

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kgrag.embeddings import Embedder
from kgrag.models import Chunk
from kgrag.ontology import ChunkExtraction, EntityType, RelType

LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "llc",
    "plc",
    "nv",
    "sa",
    "ag",
    "gmbh",
    "bv",
    "lp",
    "llp",
    "se",
    "kk",
    "holdings",
    "holding",
    "the",
}
_CONNECTORS = {"and", "of", "the", "for"}
_ACRONYM_SKIP = _CONNECTORS | {"inc", "incorporated", "ltd", "limited", "llc", "plc", "corp"}
GENERIC_DESCRIPTORS = {
    "design",
    "systems",
    "system",
    "technology",
    "technologies",
    "tech",
    "semiconductor",
    "semiconductors",
    "group",
    "processor",
    "processors",
    "platform",
    "platforms",
    "computer",
    "computers",
    "electronics",
    "devices",
    "solutions",
    "software",
    "networks",
    "products",
    "family",
    "series",
    "chips",
    "gpus",
    "cpus",
}
DISTINGUISHING_WORDS = {
    "international",
    "foundry",
    "foundries",
    "services",
    "financial",
    "finance",
    "capital",
    "cloud",
    "labs",
    "ventures",
    "biologics",
    "telecom",
    "bank",
    "insurance",
    "energy",
    "health",
    "pharmaceuticals",
    "japan",
    "china",
    "europe",
    "korea",
    "taiwan",
    "usa",
    "america",
    "india",
    "germany",
    "ireland",
    "singapore",
    "israel",
    "uk",
    "netherlands",
    "fund",
    "trust",
    "foundation",
}
SYMMETRIC = {RelType.COMPETES_WITH, RelType.PARTNERS_WITH}
ACRONYM_TYPES = {EntityType.COMPANY, EntityType.REGULATOR, EntityType.TECHNOLOGY, EntityType.LOCATION}


def _tokens(name: str) -> list[str]:
    s = unicodedata.normalize("NFKC", name).casefold().replace("&", " and ")
    s = re.sub(r"[’'`]s\b", "", s)
    s = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b)", "", s)  # n.v. -> nv, u.s. -> us
    return re.sub(r"[^\w\s]", " ", s).split()


def normalize(name: str) -> str:
    toks = _tokens(name)
    while toks and toks[0] == "the":
        toks = toks[1:]
    while len(toks) > 1 and toks[-1] in LEGAL_SUFFIXES:
        toks = toks[:-1]
    return " ".join(toks)


def acronyms(name: str) -> set[str]:
    raw = [t for t in _tokens(name) if t not in _CONNECTORS]  # keeps "inc": ADI = Analog Devices Inc
    full = [t for t in raw if t not in _ACRONYM_SKIP]
    core = normalize(name).split()
    return {"".join(t[0] for t in toks) for toks in (raw, full, core) if len(toks) >= 2}


def looks_like_acronym(name: str) -> bool:
    s = name.strip().replace(".", "")
    return 2 <= len(s) <= 6 and " " not in s and sum(ch.isupper() for ch in s) >= 2


def acronym_key(name: str) -> str:
    return name.strip().replace(".", "").lower()


def acronym_match(a: str, b: str) -> bool:
    return (looks_like_acronym(a) and acronym_key(a) in acronyms(b)) or (
        looks_like_acronym(b) and acronym_key(b) in acronyms(a)
    )


def descriptor_rule(a_norm: str, b_norm: str) -> str | None:
    """'merge' when one name is the other plus generic descriptors ("Cadence" / "Cadence Design Systems",
    "Supermicro" / "Super Micro Computer"); 'block' when the extra words name a different entity
    ("Intel" / "Intel Foundry", "Lam Research" / "Lam Research International"); None if neither applies."""
    short, long = sorted((a_norm.split(), b_norm.split()), key=len)
    if not short or not long:
        return None
    rest = None
    if long[: len(short)] == short:
        rest = long[len(short) :]
    else:
        joined = "".join(short)
        for k in range(1, len(long) + 1):
            if "".join(long[:k]) == joined:
                rest = long[k:]
                break
    if rest is None:
        return None
    if not rest or all(t in GENERIC_DESCRIPTORS for t in rest):
        return "merge"
    if any(t in DISTINGUISHING_WORDS for t in rest):
        return "block"
    return None


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80]


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[max(ra, rb)] = min(ra, rb)
        return True


@dataclass
class MentionGroup:
    type: EntityType
    norm: str
    surfaces: Counter = field(default_factory=Counter)
    chunk_ids: set[str] = field(default_factory=set)
    context: set[str] = field(default_factory=set)  # tickers / company norms a Person is tied to
    ticker: str | None = None


@dataclass
class Resolution:
    entities: list[dict]
    relations: list[dict]
    mentions: dict[str, list[str]]  # chunk_id -> entity ids
    stats: dict

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "entities": self.entities,
                    "relations": self.relations,
                    "mentions": self.mentions,
                    "stats": self.stats,
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "Resolution":
        d = json.loads(path.read_text())
        return cls(d["entities"], d["relations"], d["mentions"], d["stats"])


def edgar_anchors(ticker_map: dict[str, dict]) -> dict[str, str]:
    """normalized EDGAR company title -> ticker"""
    anchors: dict[str, str] = {}
    for ticker, info in ticker_map.items():
        anchors.setdefault(normalize(info["title"]), ticker)
    return anchors


def same_entity(
    a: str,
    b: str,
    etype: EntityType,
    embedder: Embedder,
    threshold: float,
    anchors: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Pairwise decision used for threshold tuning; mirrors the clustering rules."""
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True, "exact"
    if anchors and na in anchors and anchors.get(na) == anchors.get(nb):
        return True, "anchor"
    if etype in ACRONYM_TYPES and acronym_match(a, b):
        return True, "acronym"
    if etype != EntityType.PERSON and (rule := descriptor_rule(na, nb)):
        return rule == "merge", f"descriptor:{rule}"
    va, vb = embedder.embed_names([na, nb])
    sim = float(va @ vb)
    return sim >= threshold, f"embedding:{sim:.3f}"


class Resolver:
    def __init__(self, embedder: Embedder, threshold: float, ticker_map: dict[str, dict] | None = None):
        self.embedder = embedder
        self.threshold = threshold
        self.anchors = edgar_anchors(ticker_map or {})

    def _groups(
        self, extractions: dict[str, ChunkExtraction], chunks: dict[str, Chunk], filer_norms: dict[str, str]
    ) -> dict[tuple[EntityType, str], MentionGroup]:
        groups: dict[tuple[EntityType, str], MentionGroup] = {}
        for cid, x in extractions.items():
            chunk = chunks.get(cid)
            types = {e.name: e.type for e in x.entities}
            for e in x.entities:
                norm = normalize(e.name)
                if not norm:
                    continue
                g = groups.setdefault((e.type, norm), MentionGroup(type=e.type, norm=norm))
                g.surfaces[e.name.strip()] += 1
                g.chunk_ids.add(cid)
                if e.type == EntityType.PERSON and chunk:
                    g.context.add(filer_norms.get(chunk.ticker, chunk.ticker.lower()))
            for r in x.relations:
                if types.get(r.source) == EntityType.PERSON and types.get(r.target) == EntityType.COMPANY:
                    key = (EntityType.PERSON, normalize(r.source))
                    if key in groups:
                        groups[key].context.add(normalize(r.target))
        for g in groups.values():
            if g.type == EntityType.COMPANY:
                g.ticker = self.anchors.get(g.norm)
        return groups

    def resolve(
        self, extractions: dict[str, ChunkExtraction], chunks: list[Chunk], filer_names: dict[str, str]
    ) -> Resolution:
        chunk_map = {c.id: c for c in chunks}
        filer_norms = {t: normalize(n) for t, n in filer_names.items()}
        groups = self._groups(extractions, chunk_map, filer_norms)
        keys = list(groups)
        uf = UnionFind(len(keys))
        merges: Counter = Counter()

        # 2. ticker anchors
        by_ticker: dict[str, list[int]] = defaultdict(list)
        for i, k in enumerate(keys):
            if groups[k].ticker:
                by_ticker[groups[k].ticker].append(i)
        for ids in by_ticker.values():
            for j in ids[1:]:
                merges["anchor"] += uf.union(ids[0], j)

        by_type: dict[EntityType, list[int]] = defaultdict(list)
        for i, k in enumerate(keys):
            by_type[k[0]].append(i)

        # 3. acronyms
        for etype, ids in by_type.items():
            if etype not in ACRONYM_TYPES:
                continue
            acro_index: dict[str, list[int]] = defaultdict(list)
            for i in ids:
                for s in groups[keys[i]].surfaces:
                    for a in acronyms(s):
                        acro_index[a].append(i)
            for i in ids:
                for s in groups[keys[i]].surfaces:
                    if looks_like_acronym(s):
                        for j in acro_index.get(acronym_key(s), []):
                            if j != i:
                                merges["acronym"] += uf.union(i, j)

        # 3b. descriptor rule, bucketed by the first 4 letters of the space-free name
        blocked: set[tuple[int, int]] = set()
        for etype, ids in by_type.items():
            if etype == EntityType.PERSON:
                continue
            buckets: dict[str, list[int]] = defaultdict(list)
            for i in ids:
                buckets[keys[i][1].replace(" ", "")[:4]].append(i)
            for bucket in buckets.values():
                for x, a in enumerate(bucket):
                    for b in bucket[x + 1 :]:
                        rule = descriptor_rule(keys[a][1], keys[b][1])
                        if rule == "merge":
                            merges["descriptor"] += uf.union(a, b)
                        elif rule == "block":
                            blocked.add((min(a, b), max(a, b)))

        # 4. embeddings, blocked by type
        for etype, ids in by_type.items():
            if len(ids) < 2:
                continue
            vecs = self.embedder.embed_names([keys[i][1] for i in ids])
            for start in range(0, len(ids), 1024):
                sims = vecs[start : start + 1024] @ vecs.T
                rows, cols = np.nonzero(sims >= self.threshold)
                for r, c in zip(rows, cols, strict=True):
                    a, b = ids[start + r], ids[c]
                    if a >= b or (a, b) in blocked:
                        continue
                    if etype == EntityType.PERSON and not self._person_compatible(
                        groups[keys[a]], groups[keys[b]]
                    ):
                        continue
                    merges["embedding"] += uf.union(a, b)

        # clusters -> entities
        clusters: dict[int, list[int]] = defaultdict(list)
        for i in range(len(keys)):
            clusters[uf.find(i)].append(i)
        group_to_entity: dict[tuple[EntityType, str], str] = {}
        entities: list[dict] = []
        for members in clusters.values():
            surfaces: Counter = Counter()
            ticker = None
            chunk_ids: set[str] = set()
            for i in members:
                g = groups[keys[i]]
                surfaces.update(g.surfaces)
                ticker = ticker or g.ticker
                chunk_ids |= g.chunk_ids
            name = self._canonical(surfaces)
            etype = keys[members[0]][0]
            eid = f"{etype.value.lower()}:{slug(normalize(name)) or slug(name)}"
            aliases = sorted({s for s in surfaces if s != name} | ({ticker} if ticker else set()))
            entities.append(
                {
                    "id": eid,
                    "type": etype.value,
                    "name": name,
                    "aliases": aliases,
                    "ticker": ticker,
                    "mentions": len(chunk_ids),
                }
            )
            for i in members:
                group_to_entity[keys[i]] = eid

        relations, mentions = self._relations(extractions, group_to_entity)
        stats = {
            "mention_groups": len(keys),
            "entities": len(entities),
            "relations": len(relations),
            "merges": dict(merges),
            "threshold": self.threshold,
            "by_type": dict(Counter(e["type"] for e in entities)),
        }
        return Resolution(entities, relations, mentions, stats)

    @staticmethod
    def _person_compatible(a: MentionGroup, b: MentionGroup) -> bool:
        # Same surname and at least one shared company; two unrelated "John Smith"s stay separate.
        return a.norm.split()[-1] == b.norm.split()[-1] and bool(a.context & b.context)

    @staticmethod
    def _canonical(surfaces: Counter) -> str:
        ranked = sorted(surfaces.items(), key=lambda kv: (-kv[1], -len(kv[0])))
        mixed = [s for s, _ in ranked if not s.isupper()]
        return mixed[0] if mixed else ranked[0][0]

    @staticmethod
    def _relations(
        extractions: dict[str, ChunkExtraction], group_to_entity: dict[tuple[EntityType, str], str]
    ) -> tuple[list[dict], dict[str, list[str]]]:
        edges: dict[tuple[str, str, str], dict] = {}
        mentions: dict[str, list[str]] = {}
        for cid, x in extractions.items():
            types = {e.name: e.type for e in x.entities}
            ids = {e.name: group_to_entity.get((e.type, normalize(e.name))) for e in x.entities}
            mentions[cid] = sorted({i for i in ids.values() if i})
            for r in x.relations:
                src, dst = ids.get(r.source), ids.get(r.target)
                if not src or not dst or src == dst or r.source not in types:
                    continue
                rtype = r.type
                if rtype == RelType.CUSTOMER_OF:  # store every supply relationship one way
                    rtype, src, dst = RelType.SUPPLIES, dst, src
                if rtype in SYMMETRIC and src > dst:
                    src, dst = dst, src
                key = (src, rtype.value, dst)
                e = edges.setdefault(
                    key,
                    {
                        "src": src,
                        "type": rtype.value,
                        "dst": dst,
                        "chunk_ids": [],
                        "confidence": 0.0,
                        "role": None,
                        "year": None,
                        "evidence": [],
                    },
                )
                if cid not in e["chunk_ids"]:
                    e["chunk_ids"].append(cid)
                e["confidence"] = max(e["confidence"], r.confidence)
                e["role"] = e["role"] or r.role
                e["year"] = e["year"] or r.year
                if len(e["evidence"]) < 3:
                    e["evidence"].append(r.evidence)
        return list(edges.values()), mentions


def tune_threshold(
    pairs: list[dict],
    embedder: Embedder,
    thresholds: list[float],
    anchors: dict[str, str] | None = None,
    min_precision: float = 0.95,
) -> list[dict]:
    """Precision/recall of the pairwise merge decision at each threshold on hand-labeled pairs."""
    rows = []
    for t in thresholds:
        tp = fp = fn = tn = 0
        for p in pairs:
            pred, _ = same_entity(p["a"], p["b"], EntityType(p["type"]), embedder, t, anchors)
            tp += pred and p["same"]
            fp += pred and not p["same"]
            fn += (not pred) and p["same"]
            tn += (not pred) and not p["same"]
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": t,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "eligible": precision >= min_precision,
            }
        )
    return rows
