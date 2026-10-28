"""Turn graph paths into readable, cited statements. Raw triples in a prompt produce awkward answers."""

from dataclasses import dataclass, field

PHRASES = {
    "SUPPLIES": "{a} supplies {b}",
    "COMPETES_WITH": "{a} competes with {b}",
    "SUBSIDIARY_OF": "{a} is a subsidiary of {b}",
    "ACQUIRED": "{a} acquired {b}",
    "PARTNERS_WITH": "{a} partners with {b}",
    "EXECUTIVE_OF": "{a} is an executive of {b}",
    "DIRECTOR_OF": "{a} is a director of {b}",
    "FORMERLY_AT": "{a} previously worked at {b}",
    "MANUFACTURES": "{a} makes {b}",
    "USES_TECHNOLOGY": "{a} uses {b}",
    "TARGETS_MARKET": "{a} serves the {b} market",
    "LOCATED_IN": "{a} is located in {b}",
    "REGULATED_BY": "{a} is subject to {b}",
}
MAX_CITES_PER_FACT = 3


@dataclass
class GraphFact:
    key: tuple[str, str, str]
    text: str
    chunk_ids: list[str]


@dataclass
class Verbalized:
    facts: list[GraphFact] = field(default_factory=list)
    chains: list[str] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)

    @property
    def chunk_ids(self) -> list[str]:
        return list(dict.fromkeys(c for f in self.facts for c in f.chunk_ids))


def edge_sentence(rel: dict, names: dict[str, str]) -> str:
    a, b = names.get(rel["src"], rel["src"]), names.get(rel["dst"], rel["dst"])
    text = PHRASES.get(rel["type"], "{a} " + rel["type"].lower().replace("_", " ") + " {b}").format(a=a, b=b)
    if rel["type"] == "EXECUTIVE_OF" and rel.get("role"):
        text = f"{a} is {rel['role']} of {b}"
    if rel["type"] == "ACQUIRED" and rel.get("year"):
        text += f" in {rel['year']}"
    return text + "."


def chain_text(nodes: list[dict], rels: list[dict]) -> str:
    parts = [nodes[0]["name"]]
    for i, r in enumerate(rels):
        forward = r["src"] == nodes[i]["id"]
        parts.append(f"-[{r['type']}]->" if forward else f"<-[{r['type']}]-")
        parts.append(nodes[i + 1]["name"])
    return " ".join(parts)


def verbalize(rows: list[dict], aggregate: bool = False, max_facts: int = 40) -> Verbalized:
    """rows: [{nodes: [...], rels: [...], degree?: int}] as returned by the Cypher templates."""
    out = Verbalized()
    seen: dict[tuple[str, str, str], GraphFact] = {}
    for row in rows:
        nodes, rels = row["nodes"], row["rels"]
        names = {n["id"]: n["name"] for n in nodes}
        if len(rels) > 1:
            out.chains.append(chain_text(nodes, rels))
        for r in rels:
            key = (r["src"], r["type"], r["dst"])
            if key in seen or len(seen) >= max_facts:
                continue
            seen[key] = GraphFact(
                key, edge_sentence(r, names), list(r.get("chunk_ids") or [])[:MAX_CITES_PER_FACT]
            )
    out.facts = list(seen.values())
    out.chains = list(dict.fromkeys(out.chains))[:20]
    if aggregate and rows:
        if "degree" in rows[0]:
            ranking = {}
            for row in rows:
                ranking.setdefault(row["nodes"][0]["name"], row["degree"])
            out.summaries.append(
                "Ranking by number of matching relationships in the graph: "
                + "; ".join(f"{n}: {d}" for n, d in ranking.items())
            )
        else:
            anchor = rows[0]["nodes"][0]["name"]
            others = {row["nodes"][-1]["name"] for row in rows}
            out.summaries.append(
                f"The graph has {len(others)} matching relationships for {anchor}: "
                + ", ".join(sorted(others))
            )
    return out
