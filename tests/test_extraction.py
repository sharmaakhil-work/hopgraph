import pytest

from hopgraph.llm import BudgetExceeded, StructuredOutputError, strict_schema
from hopgraph.ontology import ChunkExtraction, validate_extraction

CHUNK_ID = "c_abc"
TEXT = "We rely on Taiwan Semiconductor Manufacturing Company Limited to manufacture our GPUs."


def extraction(**rel_overrides) -> dict:
    rel = {
        "type": "SUPPLIES",
        "source": "TSMC Limited",
        "target": "NVIDIA",
        "source_chunk_id": CHUNK_ID,
        "confidence": 0.95,
        "evidence": "rely on Taiwan Semiconductor Manufacturing Company Limited",
        "role": None,
        "year": None,
    } | rel_overrides
    return {
        "entities": [
            {"name": "TSMC Limited", "type": "Company"},
            {"name": "NVIDIA", "type": "Company"},
            {"name": "GPU", "type": "Product"},
        ],
        "relations": [rel],
    }


def check(x: dict) -> list[str]:
    return validate_extraction(ChunkExtraction.model_validate(x), CHUNK_ID, TEXT)


def test_valid_extraction_passes():
    assert check(extraction()) == []


def test_disallowed_type_pair_rejected():
    errors = check(extraction(type="SUPPLIES", target="GPU"))
    assert any("not allowed" in e for e in errors)


def test_unknown_endpoint_rejected():
    assert any("must exactly match" in e for e in check(extraction(source="Samsung")))


def test_wrong_chunk_id_and_invented_evidence_rejected():
    errors = check(extraction(source_chunk_id="c_other", evidence="Samsung makes our chips"))
    assert any("source_chunk_id" in e for e in errors)
    assert any("verbatim" in e for e in errors)


def test_strict_schema_closes_objects_and_requires_all_fields():
    schema = strict_schema(ChunkExtraction)
    rel = schema["$defs"]["ExtractedRelation"]
    assert rel["additionalProperties"] is False
    assert set(rel["required"]) == set(rel["properties"])
    assert "default" not in str(schema)


def test_retry_feeds_errors_back_then_succeeds(fake_llm):
    llm = fake_llm([extraction(target="GPU"), extraction()])
    x, usage = llm.structured(
        model="claude-haiku-4-5",
        system="s",
        user="u",
        output=ChunkExtraction,
        purpose="extract",
        validator=lambda x: validate_extraction(x, CHUNK_ID, TEXT),
    )
    assert x.relations[0].target == "NVIDIA"
    requests = llm.client.messages.requests
    assert len(requests) == 2
    assert "failed validation" in requests[1]["messages"][-1]["content"]
    assert usage.input_tokens == 2000


def test_invalid_json_exhausts_retries(fake_llm):
    llm = fake_llm(["not json", "{}", '{"entities": []}'])
    with pytest.raises(StructuredOutputError):
        llm.structured(model="claude-haiku-4-5", system="s", user="u", output=ChunkExtraction, purpose="t")


def test_budget_cap_stops_calls(fake_llm):
    llm = fake_llm([extraction()] * 5, max_usd=0.001)
    llm.structured(model="claude-haiku-4-5", system="s", user="u", output=ChunkExtraction, purpose="t")
    with pytest.raises(BudgetExceeded):
        llm.structured(model="claude-haiku-4-5", system="s", user="u", output=ChunkExtraction, purpose="t")


def test_haiku_gets_no_effort_param(fake_llm):
    llm = fake_llm([extraction()])
    llm.structured(
        model="claude-haiku-4-5", system="s", user="u", output=ChunkExtraction, purpose="t", effort="low"
    )
    assert "effort" not in llm.client.messages.requests[0]["output_config"]
