import pytest
from pydantic import ValidationError

from hopgraph.ontology import RelType
from hopgraph.retrieval.cypher_templates import (
    MAX_HOPS,
    TEMPLATES,
    GraphParams,
    GraphPlan,
    TemplateError,
    TemplateId,
    build_query,
    clamp_hops,
    rel_pattern,
)


def plan(tid: TemplateId, **params) -> GraphPlan:
    return GraphPlan(template_id=tid, params=GraphParams(**params))


def test_every_template_builds_and_passes_entities_as_parameters():
    for tid, t in TEMPLATES.items():
        q, params = build_query(plan(tid, entity_a="x", entity_b="y"), "company:a", "company:b")
        assert "company:a" not in q and "company:b" not in q
        if "a" in t.needs:
            assert "$a" in q and params["a"] == "company:a"


def test_injection_through_rel_type_is_rejected_by_schema():
    with pytest.raises(ValidationError):
        GraphParams(rel_types=["SUPPLIES]->() DETACH DELETE n //"])
    with pytest.raises(ValueError):
        rel_pattern(["SUPPLIES|DETACH"])


def test_injection_through_entity_is_just_a_parameter():
    evil = "x'}) DETACH DELETE n //"
    q, params = build_query(plan(TemplateId.NEIGHBORS, entity_a=evil), evil, None)
    assert evil not in q
    assert params["a"] == evil


def test_hops_are_clamped():
    assert clamp_hops(50) == MAX_HOPS
    assert clamp_hops(-3) == 1
    q, _ = build_query(plan(TemplateId.PATH_BETWEEN, max_hops=99), "a", "b")
    assert f"*..{MAX_HOPS}]" in q


def test_customer_of_maps_to_stored_supplies_direction():
    assert rel_pattern([RelType.CUSTOMER_OF, RelType.SUPPLIES]) == "SUPPLIES"


def test_unresolved_entity_raises():
    with pytest.raises(TemplateError):
        build_query(plan(TemplateId.SHARED_NEIGHBORS), "company:a", None)


def test_queries_are_read_only():
    for tid in TEMPLATES:
        q, _ = build_query(plan(tid, rel_types=[RelType.SUPPLIES]), "a", "b")
        assert not any(w in q.upper() for w in ("CREATE", "MERGE", "DELETE", "SET ", "REMOVE", "LOAD CSV"))


def test_same_node_for_both_endpoints_is_rejected():
    with pytest.raises(TemplateError):
        build_query(plan(TemplateId.PATH_BETWEEN), "company:a", "company:a")


def test_entity_type_is_normalized_to_ontology_label():
    _, params = build_query(plan(TemplateId.TOP_BY_DEGREE, entity_type="company"), None, None)
    assert params["etype"] == "Company"
    _, params = build_query(plan(TemplateId.TOP_BY_DEGREE, entity_type="Spaceship"), None, None)
    assert params["etype"] is None
