"""Parameterized Cypher template library.

The model never writes Cypher. It picks a template id and fills typed parameters; entity mentions are
resolved to node ids in code, and values are passed as query parameters. Relationship types and hop counts
can't be Cypher parameters, so they're checked against the RelType enum / clamped to integers before being
spliced into the query text. That's the only string interpolation in this module.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field

from kgrag.ontology import EntityType, RelType

MAX_HOPS = 3
ROW_LIMIT = 50
DOMAIN_RELS = [r for r in RelType if r != RelType.CUSTOMER_OF]  # CUSTOMER_OF is stored as reversed SUPPLIES


class TemplateId(StrEnum):
    NEIGHBORS = "neighbors"
    TWO_HOP = "two_hop"
    PATH_BETWEEN = "path_between"
    SHARED_NEIGHBORS = "shared_neighbors"
    BOARD_INTERLOCKS = "board_interlocks"
    SUPPLY_CHAIN = "supply_chain"
    COUNT_BY_REL = "count_by_rel"
    TOP_BY_DEGREE = "top_by_degree"
    SUBSIDIARIES = "subsidiaries"
    ACQUISITIONS = "acquisitions"
    ENTITY_OVERVIEW = "entity_overview"


class Direction(StrEnum):
    OUT = "out"
    IN = "in"
    BOTH = "both"


class GraphParams(BaseModel):
    entity_a: str | None = Field(default=None, description="Entity mention from the question")
    entity_b: str | None = None
    rel_types: list[RelType] = Field(default_factory=list)
    rel_1: RelType | None = None
    rel_2: RelType | None = None
    direction: Direction | None = None
    max_hops: int | None = None
    since_year: int | None = None
    entity_type: str | None = Field(default=None, description="For top_by_degree: Company, Person, ...")


class GraphPlan(BaseModel):
    template_id: TemplateId
    params: GraphParams


class GraphPlans(BaseModel):
    plans: list[GraphPlan] = Field(
        description="1-3 template calls whose results together answer the question"
    )


class TemplateError(ValueError):
    pass


def rel_pattern(rels: list[RelType] | None) -> str:
    """Allowlisted relationship-type alternation for a Cypher pattern."""
    chosen = rels or DOMAIN_RELS
    out = []
    for r in chosen:
        if not isinstance(r, RelType):
            r = RelType(r)  # raises ValueError for anything not in the ontology
        out.append(RelType.SUPPLIES.value if r == RelType.CUSTOMER_OF else r.value)
    return "|".join(dict.fromkeys(out))


def clamp_hops(h: int | None, default: int = 2) -> int:
    return max(1, min(MAX_HOPS, int(h if h is not None else default)))


def _arrow(direction: Direction | None, rels: str, hops: str = "") -> tuple[str, str]:
    body = f"[:{rels}{hops}]"
    if direction == Direction.OUT:
        return f"-{body}->", ""
    if direction == Direction.IN:
        return f"<-{body}-", ""
    return f"-{body}-", ""


PATH_RETURN = (
    "RETURN [n IN nodes(p) | {id: n.id, name: n.name, type: n.type}] AS nodes, "
    "[r IN relationships(p) | {type: type(r), src: startNode(r).id, dst: endNode(r).id, "
    "chunk_ids: r.chunk_ids, role: r.role, year: r.year, confidence: r.confidence}] AS rels"
)


@dataclass(frozen=True)
class Template:
    id: TemplateId
    description: str
    needs: tuple[str, ...]  # resolved entity slots: "a", "b"
    build: Callable[[GraphParams], str]
    aggregate: bool = False


def _neighbors(p: GraphParams) -> str:
    arrow, _ = _arrow(p.direction, rel_pattern(p.rel_types))
    return f"MATCH p = (a:Entity {{id: $a}}){arrow}(b:Entity) {PATH_RETURN} LIMIT {ROW_LIMIT}"


def _two_hop(p: GraphParams) -> str:
    r1 = rel_pattern([p.rel_1] if p.rel_1 else None)
    r2 = rel_pattern([p.rel_2] if p.rel_2 else None)
    return (
        f"MATCH p = (a:Entity {{id: $a}})-[:{r1}]-(m:Entity)-[:{r2}]-(b:Entity) WHERE b <> a "
        f"{PATH_RETURN} LIMIT {ROW_LIMIT}"
    )


def _path_between(p: GraphParams) -> str:
    hops = clamp_hops(p.max_hops, 3)
    rels = rel_pattern(p.rel_types)
    return (
        f"MATCH (a:Entity {{id: $a}}), (b:Entity {{id: $b}}) "
        f"MATCH p = allShortestPaths((a)-[:{rels}*..{hops}]-(b)) {PATH_RETURN} LIMIT {ROW_LIMIT}"
    )


def _shared_neighbors(p: GraphParams) -> str:
    rels = rel_pattern(p.rel_types)
    return (
        f"MATCH p = (a:Entity {{id: $a}})-[:{rels}]-(m:Entity)-[:{rels}]-(b:Entity {{id: $b}}) "
        f"{PATH_RETURN} LIMIT {ROW_LIMIT}"
    )


def _board_interlocks(p: GraphParams) -> str:
    return (
        "MATCH p = (a:Entity {id: $a})<-[:DIRECTOR_OF|EXECUTIVE_OF]-(:Person)"
        "-[:DIRECTOR_OF|EXECUTIVE_OF|FORMERLY_AT]->(o:Company) WHERE o <> a "
        f"{PATH_RETURN} LIMIT {ROW_LIMIT}"
    )


def _supply_chain(p: GraphParams) -> str:
    hops = clamp_hops(p.max_hops, 2)
    if p.direction == Direction.OUT:  # downstream: who a supplies, and who they supply
        pattern = f"(a:Entity {{id: $a}})-[:SUPPLIES*1..{hops}]->(b:Entity)"
    else:  # upstream: a's suppliers and their suppliers
        pattern = f"(b:Entity)-[:SUPPLIES*1..{hops}]->(a:Entity {{id: $a}})"
    return f"MATCH p = {pattern} {PATH_RETURN} LIMIT {ROW_LIMIT}"


def _subsidiaries(p: GraphParams) -> str:
    hops = clamp_hops(p.max_hops, 2)
    return f"MATCH p = (s:Entity)-[:SUBSIDIARY_OF*1..{hops}]->(a:Entity {{id: $a}}) {PATH_RETURN} LIMIT {ROW_LIMIT}"


def _acquisitions(p: GraphParams) -> str:
    return (
        "MATCH p = (a:Entity {id: $a})-[r:ACQUIRED]->(t:Entity) "
        f"WHERE $since IS NULL OR r.year IS NULL OR r.year >= $since {PATH_RETURN} LIMIT {ROW_LIMIT}"
    )


def _top_by_degree(p: GraphParams) -> str:
    rels = rel_pattern(p.rel_types)
    arrow, _ = _arrow(p.direction, rels)
    return (
        f"MATCH (a:Entity){arrow}(b:Entity) WHERE $etype IS NULL OR a.type = $etype "
        "WITH a, count(DISTINCT b) AS degree ORDER BY degree DESC LIMIT 10 "
        f"MATCH p = (a){arrow}(b:Entity) WITH a, degree, collect(p)[..5] AS ps "
        "UNWIND ps AS p WITH p, degree "
        f"{PATH_RETURN}, degree"
    )


def _overview(p: GraphParams) -> str:
    rels = rel_pattern(None)
    return f"MATCH p = (a:Entity {{id: $a}})-[:{rels}]-(b:Entity) {PATH_RETURN} LIMIT {ROW_LIMIT}"


TEMPLATES: dict[TemplateId, Template] = {
    t.id: t
    for t in [
        Template(
            TemplateId.NEIGHBORS,
            "Direct relationships of one entity, optionally filtered by rel_types/direction "
            "(e.g. 'Who supplies NVIDIA?' -> rel_types=[SUPPLIES], direction=in).",
            ("a",),
            _neighbors,
        ),
        Template(
            TemplateId.TWO_HOP,
            "Two-step chain from entity_a via rel_1 then rel_2 (e.g. 'Which companies "
            "compete with NVIDIA's suppliers?' -> rel_1=SUPPLIES, rel_2=COMPETES_WITH).",
            ("a",),
            _two_hop,
        ),
        Template(
            TemplateId.PATH_BETWEEN,
            "How two entities are connected, up to max_hops (<=3).",
            ("a", "b"),
            _path_between,
        ),
        Template(
            TemplateId.SHARED_NEIGHBORS,
            "Entities directly related to both entity_a and entity_b (common suppliers, shared directors).",
            ("a", "b"),
            _shared_neighbors,
        ),
        Template(
            TemplateId.BOARD_INTERLOCKS,
            "Other companies where entity_a's directors/officers also serve or worked.",
            ("a",),
            _board_interlocks,
        ),
        Template(
            TemplateId.SUPPLY_CHAIN,
            "Multi-tier supply chain of entity_a: direction=in for upstream suppliers "
            "(default), out for downstream customers; max_hops<=3.",
            ("a",),
            _supply_chain,
        ),
        Template(
            TemplateId.COUNT_BY_REL,
            "Count an entity's relationships of rel_types (e.g. 'How many subsidiaries...').",
            ("a",),
            _neighbors,
            aggregate=True,
        ),
        Template(
            TemplateId.TOP_BY_DEGREE,
            "Corpus-wide ranking: which entities have the most rel_types "
            "relationships (e.g. 'Which company has the most suppliers named?').",
            (),
            _top_by_degree,
            aggregate=True,
        ),
        Template(
            TemplateId.SUBSIDIARIES,
            "Subsidiaries of entity_a, optionally nested (max_hops).",
            ("a",),
            _subsidiaries,
        ),
        Template(
            TemplateId.ACQUISITIONS,
            "Companies acquired by entity_a, optionally since_year.",
            ("a",),
            _acquisitions,
        ),
        Template(
            TemplateId.ENTITY_OVERVIEW,
            "All direct relationships of entity_a when nothing narrower fits.",
            ("a",),
            _overview,
        ),
    ]
}


def build_query(plan: GraphPlan, a_id: str | None, b_id: str | None) -> tuple[str, dict]:
    t = TEMPLATES[plan.template_id]
    if "a" in t.needs and not a_id:
        raise TemplateError(f"{t.id}: entity_a could not be resolved")
    if "b" in t.needs and not b_id:
        raise TemplateError(f"{t.id}: entity_b could not be resolved")
    if "b" in t.needs and a_id == b_id:
        raise TemplateError(f"{t.id}: entity_a and entity_b resolved to the same node")
    etype = next(
        (e.value for e in EntityType if e.value.lower() == (plan.params.entity_type or "").lower()), None
    )
    params = {"a": a_id, "b": b_id, "since": plan.params.since_year, "etype": etype}
    return t.build(plan.params), params


def template_catalog() -> str:
    return "\n".join(
        f"- {t.id.value} (needs: {', '.join(t.needs) or 'none'}): {t.description}" for t in TEMPLATES.values()
    )
