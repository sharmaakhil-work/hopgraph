from datetime import date

from fastapi.testclient import TestClient

from hopgraph.api.app import create_app
from hopgraph.corpus.edgar import extract_director_bios, split_10k_items
from hopgraph.evaluation.judge import entity_f1
from hopgraph.ingest.chunk import chunk_document, chunk_id
from hopgraph.models import Document, Passage, Section
from hopgraph.pipeline import QueryResult

FILLER = "The Company designs accelerated computing platforms. " * 150
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


def test_running_page_headers_do_not_split_an_item():
    pages = "\n".join(f"PART I\nItem 1\n{FILLER}" for _ in range(3))
    text = f"Item 1. Business\n{pages}\nItem 1A. Risk Factors\n{FILLER}\nItem 7. MD&A\n{FILLER}\nItem 8. Statements\n"
    sections = {s.path: s.text for s in split_10k_items(text)}
    assert sections["10-K/Item 1"].count("accelerated computing") >= 3 * 150


def test_filing_without_item_headings_falls_back_to_narrative():
    text = "Fundamentals of our business. " + FILLER * 5 + "\nPart I Item 1. Business: page 4\n"
    (section,) = split_10k_items(text)
    assert section.path == "10-K/Narrative (no Item headings)" and section.text.startswith("Fundamentals")


def test_director_bios_found_in_prose_and_label_layouts():
    noise = "The annual meeting will be held virtually and shareholders may vote online. " * 80
    prose = "Jane Example has served as a director since 2019 and sits on the board of Fabrikam."
    labels = "John Sample\n\nDirector Since:\n 2021\n\nAge:\n 58\n\nMr. Sample is CEO of Contoso."
    out = extract_director_bios(f"{noise}\n\n{prose}\n\n{noise}\n\n{labels}\n\n{noise}", after=200, before=50)
    assert len(out) == 1
    assert "Jane Example" in out[0].text and "Mr. Sample is CEO of Contoso" in out[0].text
    assert len(out[0].text) < len(noise)


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

    def answer(self, question, mode="auto", system="hopgraph"):
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
