from datetime import date

from kgrag.answer.citations import AnswerDraft, citation_errors
from kgrag.answer.context import build_context
from kgrag.answer.generate import generate_answer
from kgrag.embeddings import HashEmbedder
from kgrag.models import Passage
from kgrag.retrieval.entity_link import EntityLinker
from kgrag.retrieval.router import Route, route
from kgrag.retrieval.verbalize import verbalize

ROWS = [
    {
        "nodes": [
            {"id": "company:tsmc", "name": "TSMC", "type": "Company"},
            {"id": "company:nvidia", "name": "NVIDIA", "type": "Company"},
            {"id": "company:microsoft", "name": "Microsoft", "type": "Company"},
        ],
        "rels": [
            {"type": "SUPPLIES", "src": "company:tsmc", "dst": "company:nvidia", "chunk_ids": ["c_1"]},
            {
                "type": "SUPPLIES",
                "src": "company:nvidia",
                "dst": "company:microsoft",
                "chunk_ids": ["c_2", "c_3"],
            },
        ],
    }
]


def passage(cid: str, text: str = "text") -> Passage:
    return Passage(
        chunk_id=cid,
        doc_id="d",
        ticker="NVDA",
        form="10-K",
        section_path="10-K/Item 1",
        filing_date=date(2026, 1, 1),
        text=text,
        score=0.9,
    )


def test_verbalize_paths_into_cited_sentences_and_chains():
    v = verbalize(ROWS)
    assert [f.text for f in v.facts] == ["TSMC supplies NVIDIA.", "NVIDIA supplies Microsoft."]
    assert v.facts[1].chunk_ids == ["c_2", "c_3"]
    assert v.chains == ["TSMC -[SUPPLIES]-> NVIDIA -[SUPPLIES]-> Microsoft"]


def test_verbalize_dedupes_edges_across_paths():
    assert len(verbalize(ROWS + ROWS).facts) == 2


def test_context_labels_blocks_and_limits_allowed_ids():
    v = verbalize(ROWS)
    cited = {"c_1": passage("c_1"), "c_2": passage("c_2")}  # c_3 not found in pgvector
    ctx = build_context(v, [passage("c_9"), passage("c_9")], cited)
    assert "=== GRAPH FACTS" in ctx.text and "=== RETRIEVED PASSAGES" in ctx.text
    assert ctx.allowed_ids == {"c_1", "c_2", "c_9"}
    assert ctx.text.count("[c_9]") == 1


def test_citation_validator():
    ok = AnswerDraft(answerable=True, claims=[{"text": "a", "citations": ["c_1"]}])
    assert citation_errors(ok, {"c_1"}) == []
    invented = AnswerDraft(answerable=True, claims=[{"text": "a", "citations": ["c_404"]}])
    assert "not present" in citation_errors(invented, {"c_1"})[0]
    uncited = AnswerDraft(answerable=True, claims=[{"text": "a", "citations": []}])
    assert citation_errors(uncited, {"c_1"})
    refusal_with_claims = AnswerDraft(answerable=False, claims=[{"text": "a", "citations": ["c_1"]}])
    assert citation_errors(refusal_with_claims, {"c_1"})


def test_invented_citation_is_regenerated(fake_llm):
    bad = {
        "answerable": True,
        "claims": [{"text": "TSMC supplies NVIDIA.", "citations": ["c_404"]}],
        "refusal_reason": None,
    }
    good = {
        "answerable": True,
        "claims": [{"text": "TSMC supplies NVIDIA.", "citations": ["c_1"]}],
        "refusal_reason": None,
    }
    llm = fake_llm([bad, good])
    ctx = build_context(verbalize(ROWS), [], {"c_1": passage("c_1")})
    ans = generate_answer(llm, "claude-sonnet-5-5", "low", "Who supplies NVIDIA?", ctx)
    assert ans.citations_valid and ans.regenerations == 1
    assert ans.text == "TSMC supplies NVIDIA. [c_1]"
    assert llm.client.messages.requests[0]["output_config"]["effort"] == "low"


def test_persistent_invalid_citations_become_a_refusal(fake_llm):
    bad = {"answerable": True, "claims": [{"text": "x", "citations": ["c_404"]}], "refusal_reason": None}
    llm = fake_llm([bad, bad, bad])
    ctx = build_context(None, [passage("c_1")], {})
    ans = generate_answer(llm, "claude-sonnet-5-5", None, "q", ctx)
    assert not ans.citations_valid and not ans.draft.answerable
    assert ans.draft.claims == []


def test_empty_context_refuses_without_calling_the_model(fake_llm):
    llm = fake_llm([])
    ans = generate_answer(llm, "claude-sonnet-5-5", None, "q", build_context(None, [], {}))
    assert not ans.draft.answerable and llm.client.messages.requests == []


def test_router_low_confidence_falls_back_to_hybrid(fake_llm):
    llm = fake_llm(
        [
            {"route": "graph", "confidence": 0.55, "reason": "unsure"},
            {"route": "vector", "confidence": 0.9, "reason": "definition"},
        ]
    )
    decision, effective, _ = route(llm, "claude-haiku-4-5", "q", threshold=0.7)
    assert decision.route == Route.GRAPH and effective == Route.HYBRID
    _, effective, _ = route(llm, "claude-haiku-4-5", "What is CUDA?", threshold=0.7)
    assert effective == Route.VECTOR


def test_entity_linker_alias_acronym_scan_and_threshold():
    emb = HashEmbedder()
    ents = [
        {
            "id": "company:tsmc",
            "type": "Company",
            "name": "Taiwan Semiconductor Manufacturing Company Limited",
            "aliases": ["TSMC"],
            "mentions": 40,
        },
        {
            "id": "company:nvidia",
            "type": "Company",
            "name": "NVIDIA Corporation",
            "aliases": ["NVDA"],
            "mentions": 90,
        },
        {
            "id": "company:amd",
            "type": "Company",
            "name": "Advanced Micro Devices",
            "aliases": [],
            "mentions": 50,
        },
    ]
    linker = EntityLinker(ents, emb.embed_names([e["name"].lower() for e in ents]), emb, threshold=0.9)
    assert linker.link("Nvidia Corp").id == "company:nvidia"
    assert linker.link("tsmc").id == "company:tsmc"
    assert linker.link("AMD").method == "acronym"
    assert linker.link("Globex Industries") is None
    found = {e.id for e in linker.scan("Which companies supply both NVIDIA and TSMC?")}
    assert found == {"company:nvidia", "company:tsmc"}
