"""Write the resolved graph to Neo4j. Every statement is MERGE, so re-running ingestion is idempotent."""

from collections import defaultdict

from neo4j import Driver

from kgrag.ingest.resolve import Resolution
from kgrag.models import Chunk
from kgrag.ontology import EntityType, RelType

CONSTRAINTS = [
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
]

BATCH = 500


def _batches(rows: list[dict], size: int = BATCH):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def write_graph(driver: Driver, resolution: Resolution, chunks: list[Chunk], rebuild: bool = False) -> dict:
    with driver.session() as s:
        if rebuild:
            s.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS").consume()
        for stmt in CONSTRAINTS:
            s.run(stmt).consume()

        doc_rows = {
            c.doc_id: {
                "id": c.doc_id,
                "ticker": c.ticker,
                "form": c.form,
                "filing_date": c.filing_date.isoformat(),
            }
            for c in chunks
        }
        for rows in _batches(list(doc_rows.values())):
            s.run(
                "UNWIND $rows AS r MERGE (d:Document {id: r.id}) "
                "SET d.ticker = r.ticker, d.form = r.form, d.filing_date = date(r.filing_date)",
                rows=rows,
            ).consume()
        chunk_rows = [
            {"id": c.id, "doc_id": c.doc_id, "section": c.section_path, "ordinal": c.ordinal} for c in chunks
        ]
        for rows in _batches(chunk_rows):
            s.run(
                "UNWIND $rows AS r MERGE (c:Chunk {id: r.id}) SET c.section = r.section, c.ordinal = r.ordinal "
                "WITH c, r MATCH (d:Document {id: r.doc_id}) MERGE (c)-[:PART_OF]->(d)",
                rows=rows,
            ).consume()

        by_type: dict[str, list[dict]] = defaultdict(list)
        for e in resolution.entities:
            by_type[e["type"]].append(e)
        for etype, rows_all in by_type.items():
            label = EntityType(etype).value  # allowlisted label; labels can't be parameters
            for rows in _batches(rows_all):
                s.run(
                    f"UNWIND $rows AS r MERGE (e:Entity {{id: r.id}}) SET e:{label}, e.type = r.type, "
                    "e.name = r.name, e.aliases = r.aliases, e.ticker = r.ticker, e.mentions = r.mentions",
                    rows=rows,
                ).consume()

        mention_rows = [
            {"chunk": cid, "entity": eid} for cid, eids in resolution.mentions.items() for eid in eids
        ]
        for rows in _batches(mention_rows, 2000):
            s.run(
                "UNWIND $rows AS r MATCH (e:Entity {id: r.entity}), (c:Chunk {id: r.chunk}) "
                "MERGE (e)-[:MENTIONED_IN]->(c)",
                rows=rows,
            ).consume()

        rels_by_type: dict[str, list[dict]] = defaultdict(list)
        for r in resolution.relations:
            rels_by_type[r["type"]].append(r)
        for rtype, rows_all in rels_by_type.items():
            rel = RelType(rtype).value
            for rows in _batches(rows_all):
                s.run(
                    f"UNWIND $rows AS r MATCH (a:Entity {{id: r.src}}), (b:Entity {{id: r.dst}}) "
                    f"MERGE (a)-[x:{rel}]->(b) "
                    "ON CREATE SET x.chunk_ids = [], x.confidence = 0.0 "
                    "SET x.chunk_ids = [c IN x.chunk_ids WHERE NOT c IN r.chunk_ids] + r.chunk_ids, "
                    "    x.confidence = CASE WHEN r.confidence > x.confidence THEN r.confidence ELSE x.confidence END, "
                    "    x.role = coalesce(r.role, x.role), x.year = coalesce(r.year, x.year), "
                    "    x.evidence = r.evidence",
                    rows=rows,
                ).consume()

        counts = s.run(
            "MATCH (e:Entity) WITH count(e) AS entities "
            "OPTIONAL MATCH (:Entity)-[r]->(:Entity) WHERE type(r) <> 'MENTIONED_IN' "
            "RETURN entities, count(r) AS relations"
        ).single()
    return {"entities": counts["entities"], "relations": counts["relations"]}
