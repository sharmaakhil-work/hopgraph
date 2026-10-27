from datetime import date

from kgrag.embeddings import HashEmbedder
from kgrag.ingest.resolve import Resolver, acronyms, normalize, same_entity
from kgrag.models import Chunk
from kgrag.ontology import ChunkExtraction, EntityType


def test_normalize_strips_legal_suffixes_and_punctuation():
    assert normalize("Acme Corp.") == "acme"
    assert normalize("ACME Corporation") == "acme"
    assert normalize("The Acme Company, Inc.") == "acme"
    assert normalize("Taiwan Semiconductor Manufacturing Co., Ltd.") == "taiwan semiconductor manufacturing"
    assert normalize("NXP Semiconductors N.V.") == "nxp semiconductors"
    assert normalize("AT&T Inc.") == "at and t"


def test_acronyms():
    assert "tsmc" in acronyms("Taiwan Semiconductor Manufacturing Company Limited")
    assert "amd" in acronyms("Advanced Micro Devices, Inc.")


def test_pairwise_rules():
    emb = HashEmbedder()
    assert same_entity("Acme Corp", "ACME Corporation", EntityType.COMPANY, emb, 0.95)[0]
    assert (
        same_entity("TSMC", "Taiwan Semiconductor Manufacturing Company", EntityType.COMPANY, emb, 0.95)[1]
        == "acronym"
    )
    anchors = {"nvidia": "NVDA"}
    assert not same_entity(
        "Micron Technology", "Microchip Technology", EntityType.COMPANY, emb, 0.95, anchors
    )[0]


def _chunk(cid: str, ticker: str) -> Chunk:
    return Chunk(
        id=cid,
        doc_id=f"{ticker}-doc",
        ticker=ticker,
        form="10-K",
        section_path="10-K/Item 1",
        filing_date=date(2026, 1, 1),
        ordinal=0,
        text="",
    )


def _x(entities, relations=()) -> ChunkExtraction:
    return ChunkExtraction.model_validate(
        {
            "entities": [{"name": n, "type": t} for n, t in entities],
            "relations": [
                {
                    "type": rt,
                    "source": s,
                    "target": d,
                    "source_chunk_id": "x",
                    "confidence": c,
                    "evidence": "e",
                    "role": None,
                    "year": None,
                }
                for rt, s, d, c in relations
            ],
        }
    )


def test_resolver_collapses_variants_and_canonicalizes_edges():
    extractions = {
        "c1": _x(
            [("Acme Corp", "Company"), ("NVIDIA", "Company")], [("SUPPLIES", "Acme Corp", "NVIDIA", 0.9)]
        ),
        "c2": _x(
            [("ACME Corporation", "Company"), ("NVIDIA Corporation", "Company")],
            [("CUSTOMER_OF", "NVIDIA Corporation", "ACME Corporation", 0.95)],
        ),
        "c3": _x(
            [("Acme", "Company"), ("Micron", "Company"), ("Microchip", "Company")],
            [("COMPETES_WITH", "Micron", "Acme", 0.9)],
        ),
    }
    chunks = [_chunk("c1", "NVDA"), _chunk("c2", "NVDA"), _chunk("c3", "MU")]
    res = Resolver(
        HashEmbedder(), threshold=0.97, ticker_map={"NVDA": {"cik": 1, "title": "NVIDIA CORP"}}
    ).resolve(extractions, chunks, {"NVDA": "NVIDIA CORP"})
    by_name = {e["name"]: e for e in res.entities}
    acme = next(e for e in res.entities if e["id"] == "company:acme")
    assert {"Acme Corp", "ACME Corporation"} <= set(acme["aliases"]) | {acme["name"]}
    assert "Micron" in by_name and "Microchip" in by_name  # similar strings, different companies
    nvidia = next(e for e in res.entities if e["ticker"] == "NVDA")
    supplies = [r for r in res.relations if r["type"] == "SUPPLIES"]
    assert len(supplies) == 1  # CUSTOMER_OF folded into the same SUPPLIES edge
    assert supplies[0]["src"] == "company:acme" and supplies[0]["dst"] == nvidia["id"]
    assert sorted(supplies[0]["chunk_ids"]) == ["c1", "c2"]
    assert supplies[0]["confidence"] == 0.95
    assert set(res.mentions["c3"]) >= {"company:acme"}


def test_people_with_same_name_but_no_shared_company_stay_separate():
    extractions = {
        "c1": _x(
            [("John Smith", "Person"), ("Acme", "Company")], [("DIRECTOR_OF", "John Smith", "Acme", 0.9)]
        ),
        "c2": _x(
            [("John A. Smith", "Person"), ("Globex", "Company")],
            [("DIRECTOR_OF", "John A. Smith", "Globex", 0.9)],
        ),
        "c3": _x(
            [("John Smith Jr", "Person"), ("Acme", "Company")],
            [("EXECUTIVE_OF", "John Smith Jr", "Acme", 0.9)],
        ),
    }
    chunks = [_chunk("c1", "ACME"), _chunk("c2", "GBX"), _chunk("c3", "ACME")]
    res = Resolver(HashEmbedder(), threshold=0.5).resolve(extractions, chunks, {})
    people = [e for e in res.entities if e["type"] == "Person"]
    # "John A. Smith" at Globex shares no company with the Acme Johns
    assert any(e["name"] == "John A. Smith" and not e["aliases"] for e in people)


def test_descriptor_rule_merges_generic_and_blocks_distinguishing_suffixes():
    from kgrag.ingest.resolve import descriptor_rule

    assert descriptor_rule("cadence", "cadence design systems") == "merge"
    assert descriptor_rule("supermicro", "super micro computer") == "merge"
    assert descriptor_rule("intel", "intel foundry") == "block"
    assert descriptor_rule("lam research", "lam research international") == "block"
    assert descriptor_rule("lam", "lam research") is None
    assert descriptor_rule("micron technology", "microchip technology") is None


def test_dotted_and_suffix_acronyms():
    emb = HashEmbedder()
    assert same_entity("U.S.", "United States", EntityType.LOCATION, emb, 0.99)[1] == "acronym"
    assert same_entity("ADI", "Analog Devices, Inc.", EntityType.COMPANY, emb, 0.99)[1] == "acronym"


def test_descriptor_block_vetoes_embedding_merge_in_clustering():
    extractions = {"c1": _x([("Intel Corporation", "Company"), ("Intel Foundry", "Company")])}
    res = Resolver(HashEmbedder(), threshold=0.1).resolve(extractions, [_chunk("c1", "INTC")], {})
    assert len(res.entities) == 2
