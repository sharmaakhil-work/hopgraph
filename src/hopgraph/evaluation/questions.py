"""Draft a stratified benchmark set. Every draft is written with verified=false and must be checked by a human
against the cited source text before it counts.

Two generators, so the set isn't built only from the graph it's testing:
  - graph paths (1-3 hops, aggregations), seeded from Neo4j
  - chunk pairs: two chunks from different filings that mention the same entity, independent of extracted edges
"""

import json
import random
from pathlib import Path

from neo4j import Driver
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from hopgraph.llm import LLM
from hopgraph.retrieval.cypher_templates import DOMAIN_RELS
from hopgraph.retrieval.verbalize import edge_sentence

CATEGORIES = ["single_hop", "two_hop", "three_hop", "aggregation", "out_of_scope"]

OUT_OF_SCOPE = [
    "What was NVIDIA's closing stock price on March 3, 2019?",
    "Who won the 2022 FIFA World Cup?",
    "What revenue will Intel report in fiscal 2031?",
    "What is the home address of AMD's CEO?",
    "Which semiconductor company will TSMC acquire next year?",
    "What is Micron's internal code name for its 2029 DRAM node?",
    "How many employees does Samsung's Pyeongtaek fab cafeteria serve daily?",
    "What did Qualcomm's board discuss in its closed session last Tuesday?",
    "Which company will have the largest market cap in 2035?",
    "What is the recipe for Apple's employee cafeteria lasagna?",
    "What is the private cell phone number of Microsoft's CFO?",
    "Which of these companies is the best investment right now?",
    "What is the population of Austin, Texas in 2040?",
    "What were Broadcom's quarterly earnings in 1995?",
    "What is Oracle's unannounced next product?",
]


class DraftQuestion(BaseModel):
    question: str
    gold_answer: str
    gold_entities: list[str]
    natural: bool


QUESTION_SYSTEM = """You write benchmark questions for a question-answering system over SEC filings.

You get a set of facts (each with the verbatim evidence from the filing) that form a chain or a group. Write ONE \
natural question an analyst might ask that requires ALL of the facts to answer:
- For a chain A -> B -> C: name only A in the question and describe the path so the answer is C. Don't name \
the intermediate entities.
- For a group of facts about one entity: ask for the full list or the count.
- For two passages: ask something that needs information from both.
gold_answer: short, complete answer. gold_entities: the entity names that must appear in a correct answer.
Set natural=false if the facts are too generic or odd to make a sensible question."""


def _paths(driver: Driver, hops: int, n: int, rng: random.Random) -> list[dict]:
    rels = "|".join(r.value for r in DOMAIN_RELS)
    pattern = "(n0:Entity)" + "".join(f"-[r{i}:{rels}]-(n{i + 1}:Entity)" for i in range(hops))
    distinct = " AND ".join(f"n{i} <> n{j}" for i in range(hops + 1) for j in range(i + 1, hops + 1))
    conf = " AND ".join(f"r{i}.confidence >= 0.85" for i in range(hops))
    # sample start nodes first, then take one path per start; ORDER BY rand() over all 3-hop paths explodes
    q = (
        f"MATCH (n0:Entity) WHERE COUNT {{ (n0)-[:{rels}]-() }} > 0 WITH n0, rand() AS o ORDER BY o LIMIT $starts "
        f"CALL (n0) {{ MATCH p = {pattern} WHERE {distinct} AND {conf} RETURN p LIMIT 1 }} "
        "RETURN [x IN nodes(p) | {id: x.id, name: x.name, aliases: x.aliases}] AS nodes, "
        "[x IN relationships(p) | {type: type(x), src: startNode(x).id, dst: endNode(x).id, "
        "chunk_ids: x.chunk_ids, evidence: x.evidence}] AS rels"
    )
    with driver.session() as s:
        rows = [r.data() for r in s.run(q, starts=n * 20)]
    rng.shuffle(rows)
    return rows[: n * 3]


def _groups(driver: Driver, n: int) -> list[dict]:
    rels = "|".join(r.value for r in DOMAIN_RELS)
    q = (
        f"MATCH (a:Entity)-[r:{rels}]-(b:Entity) WITH a, type(r) AS t, collect({{r: r, b: b}}) AS xs "
        "WHERE size(xs) >= 3 AND size(xs) <= 12 WITH a, t, xs, rand() AS o ORDER BY o LIMIT $n "
        "RETURN a {.id, .name, .aliases} AS a, t, "
        "[x IN xs | {name: x.b.name, aliases: x.b.aliases, src: startNode(x.r).id, dst: endNode(x.r).id, "
        "chunk_ids: x.r.chunk_ids, evidence: x.r.evidence}] AS items"
    )
    with driver.session() as s:
        return [r.data() for r in s.run(q, n=n * 2)]


def _chunk_pairs(pool: ConnectionPool, n: int) -> list[tuple]:
    with pool.connection() as conn:
        return conn.execute(
            "SELECT a.id, a.company_ticker, a.text, b.id, b.company_ticker, b.text, e.name "
            "FROM chunks a JOIN chunks b ON a.doc_id <> b.doc_id AND a.entity_ids && b.entity_ids AND a.id < b.id "
            "JOIN LATERAL (SELECT name FROM entities WHERE id = ANY(a.entity_ids) AND id = ANY(b.entity_ids) "
            "AND type = 'Company' LIMIT 1) e ON true ORDER BY random() LIMIT %s",
            (n * 2,),
        ).fetchall()


def _draft(llm: LLM, model: str, facts: str) -> DraftQuestion:
    q, _ = llm.structured(
        model=model,
        system=QUESTION_SYSTEM,
        user=facts,
        output=DraftQuestion,
        purpose="draft_questions",
        max_tokens=600,
        cache_system=False,
    )
    return q


def _aliases(names: list[str], alias_lists: list[list[str] | None]) -> list[list[str]]:
    return [[n, *(a or [])] for n, a in zip(names, alias_lists, strict=True)]


def draft_questions(
    llm: LLM,
    model: str,
    driver: Driver,
    pool: ConnectionPool,
    out: Path,
    per_category: int = 15,
    seed: int = 0,
) -> dict[str, int]:
    rng = random.Random(seed)
    items: list[dict] = []
    counts = dict.fromkeys(CATEGORIES, 0)

    def add(category: str, q: DraftQuestion, support: list[str], gold_aliases: list[list[str]], source: str):
        if not q.natural or counts[category] >= per_category:
            return
        counts[category] += 1
        items.append(
            {
                "id": f"{category}-{counts[category]:02d}",
                "category": category,
                "question": q.question,
                "gold_answer": q.gold_answer,
                "gold_entities": [
                    a
                    for a in gold_aliases
                    if any(x.lower() in [g.lower() for g in q.gold_entities] for x in a)
                ]
                or [[g] for g in q.gold_entities],
                "support_chunk_ids": support,
                "source": source,
                "verified": False,
            }
        )

    for hops, category in ((1, "single_hop"), (2, "two_hop"), (3, "three_hop")):
        for path in _paths(driver, hops, per_category, rng):
            if counts[category] >= per_category:
                break
            names = {x["id"]: x["name"] for x in path["nodes"]}
            facts = "\n".join(
                f'- {edge_sentence(r, names)} Evidence: "{(r.get("evidence") or [""])[0]}"'
                for r in path["rels"]
            )
            end = path["nodes"][-1]
            support = [c for r in path["rels"] for c in (r["chunk_ids"] or [])[:1]]
            add(
                category,
                _draft(llm, model, f"Chain starting at {path['nodes'][0]['name']}:\n{facts}"),
                support,
                _aliases([end["name"]], [end.get("aliases")]),
                "graph_path",
            )

    for g in _groups(driver, per_category):
        if counts["aggregation"] >= per_category:
            break
        names = {
            g["a"]["id"]: g["a"]["name"],
            **{i["src"]: i["name"] for i in g["items"]},
            **{i["dst"]: i["name"] for i in g["items"]},
        }
        names[g["a"]["id"]] = g["a"]["name"]  # the anchor wins over item names keyed by the same id
        facts = "\n".join(f"- {edge_sentence({'type': g['t'], **i}, names)}" for i in g["items"])
        support = [c for i in g["items"] for c in (i["chunk_ids"] or [])[:1]]
        add(
            "aggregation",
            _draft(llm, model, f"Group about {g['a']['name']} ({g['t']}):\n{facts}"),
            support,
            _aliases([i["name"] for i in g["items"]], [i.get("aliases") for i in g["items"]]),
            "graph_group",
        )

    # chunk pairs count toward two_hop, using whatever quota the graph paths left
    for a_id, a_t, a_text, b_id, b_t, b_text, ent in _chunk_pairs(pool, per_category):
        if counts["two_hop"] >= per_category:
            break
        facts = f"Shared entity: {ent}\nPassage 1 ({a_t}):\n{a_text[:1500]}\n\nPassage 2 ({b_t}):\n{b_text[:1500]}"
        add("two_hop", _draft(llm, model, facts), [a_id, b_id], [], "chunk_pair")

    for i, q in enumerate(OUT_OF_SCOPE[:per_category], 1):
        items.append(
            {
                "id": f"out_of_scope-{i:02d}",
                "category": "out_of_scope",
                "question": q,
                "gold_answer": "REFUSE",
                "gold_entities": [],
                "support_chunk_ids": [],
                "source": "hand",
                "verified": True,
            }
        )
        counts["out_of_scope"] += 1

    with out.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return counts


def load_questions(path: Path, allow_unverified: bool = False) -> list[dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    return [r for r in rows if allow_unverified or r.get("verified")]
