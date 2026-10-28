"""Runs against live Neo4j + Postgres. Opt in with HOPGRAPH_INTEGRATION=1.

All rows use `test:` / `c_test` ids and are removed afterwards, so it is safe to run next to a real graph.
"""

import os
from datetime import date

import pytest

from hopgraph.config import get_settings
from hopgraph.ingest.graph_writer import write_graph
from hopgraph.ingest.resolve import Resolution
from hopgraph.models import Chunk
from hopgraph.ontology import RelType
from hopgraph.retrieval.cypher_templates import TEMPLATES, GraphParams, GraphPlan, TemplateId, build_query
from hopgraph.retrieval.verbalize import verbalize

pytestmark = pytest.mark.skipif(os.environ.get("HOPGRAPH_INTEGRATION") != "1", reason="needs live databases")


def _chunk(i: int) -> Chunk:
    return Chunk(
        id=f"c_test{i}",
        doc_id="test:doc",
        ticker="TST",
        form="10-K",
        section_path="10-K/Item 1",
        filing_date=date(2026, 1, 1),
        ordinal=i,
        text=f"chunk {i}",
    )


ENTITIES = [
    ("test:nvidia", "Company", "NVIDIA"),
    ("test:tsmc", "Company", "TSMC"),
    ("test:asml", "Company", "ASML"),
    ("test:msft", "Company", "Microsoft"),
    ("test:amd", "Company", "AMD"),
    ("test:jane", "Person", "Jane Doe"),
    ("test:sub", "Company", "NVIDIA Israel"),
    ("test:mellanox", "Company", "Mellanox"),
]
RELATIONS = [
    ("test:tsmc", "SUPPLIES", "test:nvidia", "c_test1"),
    ("test:asml", "SUPPLIES", "test:tsmc", "c_test2"),
    ("test:nvidia", "SUPPLIES", "test:msft", "c_test3"),
    ("test:tsmc", "SUPPLIES", "test:amd", "c_test4"),
    ("test:amd", "COMPETES_WITH", "test:nvidia", "c_test5"),
    ("test:jane", "DIRECTOR_OF", "test:nvidia", "c_test6"),
    ("test:jane", "DIRECTOR_OF", "test:msft", "c_test6"),
    ("test:sub", "SUBSIDIARY_OF", "test:nvidia", "c_test7"),
    ("test:nvidia", "ACQUIRED", "test:mellanox", "c_test8"),
]


@pytest.fixture(scope="module")
def driver():
    from hopgraph.db import neo4j_driver

    d = neo4j_driver(get_settings())
    res = Resolution(
        entities=[
            {"id": i, "type": t, "name": n, "aliases": [], "ticker": None, "mentions": 1}
            for i, t, n in ENTITIES
        ],
        relations=[
            {
                "src": s,
                "type": t,
                "dst": o,
                "chunk_ids": [c],
                "confidence": 0.9,
                "role": None,
                "year": 2020 if t == "ACQUIRED" else None,
                "evidence": ["e"],
            }
            for s, t, o, c in RELATIONS
        ],
        mentions={f"c_test{i}": ["test:nvidia"] for i in range(1, 9)},
        stats={},
    )
    chunks = [_chunk(i) for i in range(1, 9)]
    write_graph(d, res, chunks)
    write_graph(d, res, chunks)  # idempotent: a second run must not duplicate anything
    yield d
    with d.session() as s:
        s.run(
            "MATCH (n) WHERE n.id STARTS WITH 'test:' OR n.id STARTS WITH 'c_test' DETACH DELETE n"
        ).consume()
    d.close()


def _run(driver, tid: TemplateId, a=None, b=None, **params) -> list[dict]:
    q, p = build_query(GraphPlan(template_id=tid, params=GraphParams(**params)), a, b)
    with driver.session() as s:
        return s.execute_read(lambda tx: [r.data() for r in tx.run(q, p)])


def test_merge_is_idempotent(driver):
    with driver.session() as s:
        n = s.run("MATCH (e:Entity) WHERE e.id STARTS WITH 'test:' RETURN count(e) AS n").single()["n"]
        r = s.run(
            "MATCH (:Entity {id:'test:tsmc'})-[r:SUPPLIES]->(:Entity {id:'test:nvidia'}) "
            "RETURN count(r) AS n, r.chunk_ids AS ids"
        ).single()
    assert n == len(ENTITIES)
    assert r["n"] == 1 and r["ids"] == ["c_test1"]


def test_every_template_executes(driver):
    for tid in TEMPLATES:
        rows = _run(
            driver, tid, "test:nvidia", "test:amd", rel_types=[RelType.SUPPLIES], entity_type="company"
        )
        assert isinstance(rows, list), tid


def test_supply_chain_upstream_two_hops(driver):
    v = verbalize(_run(driver, TemplateId.SUPPLY_CHAIN, "test:nvidia", direction="in", max_hops=2))
    texts = {f.text for f in v.facts}
    assert "TSMC supplies NVIDIA." in texts and "ASML supplies TSMC." in texts
    assert "ASML -[SUPPLIES]-> TSMC -[SUPPLIES]-> NVIDIA" in v.chains


def test_shared_neighbors_and_interlocks(driver):
    shared = _run(
        driver, TemplateId.SHARED_NEIGHBORS, "test:nvidia", "test:amd", rel_types=[RelType.SUPPLIES]
    )
    assert {r["nodes"][1]["name"] for r in shared} == {"TSMC"}
    inter = verbalize(_run(driver, TemplateId.BOARD_INTERLOCKS, "test:nvidia"))
    assert "Jane Doe is a director of Microsoft." in {f.text for f in inter.facts}


def test_path_between_and_aggregates(driver):
    path = _run(driver, TemplateId.PATH_BETWEEN, "test:asml", "test:msft", max_hops=3)
    assert path and len(path[0]["rels"]) == 3
    subs = _run(driver, TemplateId.SUBSIDIARIES, "test:nvidia")
    assert [r["nodes"][0]["name"] for r in subs] == ["NVIDIA Israel"]
    acq = _run(driver, TemplateId.ACQUISITIONS, "test:nvidia", since_year=2019)
    assert [r["nodes"][-1]["name"] for r in acq] == ["Mellanox"]
    top = _run(driver, TemplateId.TOP_BY_DEGREE, rel_types=[RelType.SUPPLIES], entity_type="Company")
    assert top and "degree" in top[0]
