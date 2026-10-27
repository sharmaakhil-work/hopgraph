from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool


def log_query(pool: ConnectionPool, r: dict) -> None:
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO query_log (system, question, route, route_confidence, route_forced, graph_plans, "
            "linked_entities, retrieved_ids, answerable, citations_valid, regenerations, latency_ms, cost_usd, "
            "outcome, answer) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                r["system"],
                r["question"],
                r["route"],
                r["route_confidence"],
                r["route_forced"],
                Jsonb(r["graph_plans"]),
                Jsonb(r["linked_entities"]),
                r["retrieved_ids"],
                r["answerable"],
                r["citations_valid"],
                r["regenerations"],
                r["latency_ms"],
                r["cost_usd"],
                r["outcome"],
                r["answer"],
            ),
        )
