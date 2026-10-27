from datetime import date

from fastapi.testclient import TestClient

from kgrag.api.app import create_app
from kgrag.corpus.edgar import extract_director_bios, split_10k_items
from kgrag.evaluation.judge import entity_f1
from kgrag.ingest.chunk import chunk_document, chunk_id
from kgrag.models import Document, Passage, Section
from kgrag.pipeline import QueryResult

FILLER = "The Company designs accelerated computing platforms. " * 30
TENK = f"""
Table of Contents
Item 1. Business 4
Item 1A. Risk Factors 12
Item 7. Management's Discussion 40

PART I
Item 1. Business
{FILLER}
Item 1A. Risk Factors
We rely on TSMC. {FILLER}
Item 1B. Unresolved Staff Comments
None.
Item 7. Management's Discussion and Analysis
{FILLER}
Item 8. Financial Statements
"""


def test_10k_items_skip_table_of_contents():
    sections = {s.path: s.text for s in split_10k_items(TENK)}
    assert set(sections) == {"10-K/Item 1", "10-K/Item 1A", "10-K/Item 7"}
    assert "We rely on TSMC" in sections["10-K/Item 1A"]
    assert "Item 1B" not in sections["10-K/Item 1A"]


def test_director_bios_filter():
    bio = "Jane Example has served as a director since 2019 and serves on the board of Fabrikam. " * 4
    text = f"{bio}\n\nThe annual meeting will be held virtually and shareholders may vote online. " * 1
    out = extract_director_bios(text)
    assert len(out) == 1 and "Jane Example" in out[0].text and "annual meeting" not in out[0].text


def test_chunk_ids_are_stable():
    doc = Document(
        doc_id="NVDA-10-K-2026-02-25",
        ticker="NVDA",
        company_name="NVIDIA CORP",
        cik=1,
        form="10-K",
        filing_date=date(2026, 2, 25),
        url="u",
        sections=[Section(path="10-K/Item 1", text=FILLER * 5)],
    )
    a, b = chunk_document(doc, 1000, 100), chunk_document(doc, 1000, 100)
    assert [c.id for c in a] == [c.id for c in b]
    assert a[0].id == chunk_id(doc.doc_id, "10-K/Item 1", 0)
    assert len({c.id for c in a}) == len(a) > 1


def test_entity_f1_uses_aliases():
    gold = [["Taiwan Semiconductor Manufacturing Company", "TSMC"], ["Samsung Electronics"]]
    assert entity_f1(gold, ["TSMC", "Samsung Electronics"]) == 1.0
    assert 0 < entity_f1(gold, ["TSMC", "Intel"]) < 1
    assert entity_f1(gold, []) == 0.0


class StubPipeline:
    def __init__(self):
        self.vectors = self

    def fetch(self, ids):
        return {
            i: Passage(
                chunk_id=i,
                doc_id="d",
                ticker="NVDA",
                form="10-K",
                section_path="s",
                filing_date=None,
                text="t",
                score=0,
            )
            for i in ids
            if i == "c_1"
        }

    def answer(self, question, mode="auto", system="kgrag"):
        return QueryResult(
            question=question,
            system=system,
            route="graph" if mode == "auto" else mode,
            route_confidence=0.9,
            route_forced=mode != "auto",
            answerable=True,
            answer="TSMC supplies NVIDIA. [c_1]",
            claims=[{"text": "TSMC supplies NVIDIA.", "citations": ["c_1"]}],
            citations=[],
            graph_facts=[],
            graph_chains=[],
            graph_plans=[],
            linked_entities=[],
            retrieved_ids=["c_1"],
            citations_valid=True,
            regenerations=0,
            outcome="ok",
            latency_ms=5,
            cost_usd=0.001,
        )


def test_api_endpoints():
    with TestClient(create_app(StubPipeline)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        r = client.post("/query", json={"question": "Who supplies NVIDIA?", "mode": "graph"})
        assert r.status_code == 200 and r.json()["route"] == "graph"
        assert client.post("/query", json={"question": "hi", "mode": "sql"}).status_code == 422
        assert client.get("/chunks/c_1").status_code == 200
        assert client.get("/chunks/c_missing").status_code == 404
