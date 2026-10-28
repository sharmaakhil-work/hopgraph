from dataclasses import dataclass, field

from hopgraph.models import Passage
from hopgraph.retrieval.verbalize import Verbalized

PASSAGE_CHARS = 1800


@dataclass
class Context:
    text: str
    allowed_ids: set[str]
    passages: dict[str, Passage] = field(default_factory=dict)
    n_facts: int = 0

    @property
    def empty(self) -> bool:
        return not self.allowed_ids


def build_context(
    graph: Verbalized | None, passages: list[Passage], cited_chunks: dict[str, Passage]
) -> Context:
    """Two labeled blocks. Graph facts cite the chunk that justified the edge; passages are cited by their own id.

    `cited_chunks` are the texts of chunks referenced by graph facts (fetched from pgvector through the shared
    chunk id); they back the citation snippets but aren't repeated in the prompt unless also retrieved.
    """
    lines: list[str] = []
    allowed: set[str] = set()
    n_facts = 0
    if graph and (graph.facts or graph.summaries):
        lines.append(
            "=== GRAPH FACTS (from the knowledge graph; each fact cites the filing chunk it was extracted from) ==="
        )
        for f in graph.facts:
            ids = [c for c in f.chunk_ids if c in cited_chunks]
            if not ids:
                continue
            lines.append(f"{' '.join(f'[{c}]' for c in ids)} {f.text}")
            allowed.update(ids)
            n_facts += 1
        for s in graph.summaries:
            lines.append(f"(aggregate) {s}")
        if graph.chains:
            lines.append("Connection chains found (structure only; cite the facts above):")
            lines.extend(f"- {c}" for c in graph.chains)
        lines.append("")

    unique: dict[str, Passage] = {}
    for p in passages:
        unique.setdefault(p.chunk_id, p)
    if unique:
        lines.append("=== RETRIEVED PASSAGES (from filing text) ===")
        for p in unique.values():
            header = f"[{p.chunk_id}] ({p.ticker} {p.form} {p.filing_date or ''}, {p.section_path})"
            lines.append(f"{header}\n{p.text[:PASSAGE_CHARS]}\n")
            allowed.add(p.chunk_id)

    sources = {**{k: v for k, v in cited_chunks.items() if k in allowed}, **unique}
    return Context(text="\n".join(lines).strip(), allowed_ids=allowed, passages=sources, n_facts=n_facts)
