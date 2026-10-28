import re
from enum import StrEnum

from pydantic import BaseModel, Field


class EntityType(StrEnum):
    COMPANY = "Company"
    PERSON = "Person"
    PRODUCT = "Product"
    TECHNOLOGY = "Technology"
    MARKET = "Market"
    LOCATION = "Location"
    REGULATOR = "Regulator"
    FACILITY = "Facility"


class RelType(StrEnum):
    SUPPLIES = "SUPPLIES"
    CUSTOMER_OF = "CUSTOMER_OF"
    COMPETES_WITH = "COMPETES_WITH"
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    ACQUIRED = "ACQUIRED"
    PARTNERS_WITH = "PARTNERS_WITH"
    EXECUTIVE_OF = "EXECUTIVE_OF"
    DIRECTOR_OF = "DIRECTOR_OF"
    FORMERLY_AT = "FORMERLY_AT"
    MANUFACTURES = "MANUFACTURES"
    USES_TECHNOLOGY = "USES_TECHNOLOGY"
    TARGETS_MARKET = "TARGETS_MARKET"
    LOCATED_IN = "LOCATED_IN"
    REGULATED_BY = "REGULATED_BY"


E = EntityType
ORG = {E.COMPANY}

# Allowed (source, target) type pairs per relationship. Anything else is rejected at validation.
REL_SIGNATURES: dict[RelType, set[tuple[EntityType, EntityType]]] = {
    RelType.SUPPLIES: {(E.COMPANY, E.COMPANY)},
    RelType.CUSTOMER_OF: {(E.COMPANY, E.COMPANY)},
    RelType.COMPETES_WITH: {(E.COMPANY, E.COMPANY), (E.PRODUCT, E.PRODUCT)},
    RelType.SUBSIDIARY_OF: {(E.COMPANY, E.COMPANY)},
    RelType.ACQUIRED: {(E.COMPANY, E.COMPANY)},
    RelType.PARTNERS_WITH: {(E.COMPANY, E.COMPANY)},
    RelType.EXECUTIVE_OF: {(E.PERSON, E.COMPANY)},
    RelType.DIRECTOR_OF: {(E.PERSON, E.COMPANY)},
    RelType.FORMERLY_AT: {(E.PERSON, E.COMPANY)},
    RelType.MANUFACTURES: {(E.COMPANY, E.PRODUCT), (E.FACILITY, E.PRODUCT)},
    RelType.USES_TECHNOLOGY: {(E.PRODUCT, E.TECHNOLOGY), (E.COMPANY, E.TECHNOLOGY)},
    RelType.TARGETS_MARKET: {(E.COMPANY, E.MARKET), (E.PRODUCT, E.MARKET)},
    RelType.LOCATED_IN: {(E.COMPANY, E.LOCATION), (E.FACILITY, E.LOCATION)},
    RelType.REGULATED_BY: {(E.COMPANY, E.REGULATOR), (E.PRODUCT, E.REGULATOR)},
}

REL_DESCRIPTIONS: dict[RelType, str] = {
    RelType.SUPPLIES: "source sells components, manufacturing, IP or services to target (e.g. TSMC SUPPLIES NVIDIA)",
    RelType.CUSTOMER_OF: "source buys from target; use when the text frames target as a supplier/vendor of source",
    RelType.COMPETES_WITH: "text explicitly names them as competitors",
    RelType.SUBSIDIARY_OF: "source is a subsidiary / wholly-owned entity of target",
    RelType.ACQUIRED: "source acquired or agreed to acquire target; set year if stated",
    RelType.PARTNERS_WITH: "strategic partnership, joint venture, or collaboration",
    RelType.EXECUTIVE_OF: "person is an officer of the company; set role (e.g. 'CEO', 'CFO')",
    RelType.DIRECTOR_OF: "person serves or served on the company's board of directors",
    RelType.FORMERLY_AT: "person previously worked at the company (not as a current director/officer)",
    RelType.MANUFACTURES: "company or facility makes / sells the product",
    RelType.USES_TECHNOLOGY: "product or company uses or is built on the technology (process node, architecture, standard)",
    RelType.TARGETS_MARKET: "company or product serves an end market (data center, automotive, gaming, ...)",
    RelType.LOCATED_IN: "headquarters or facility located in a country/region/city",
    RelType.REGULATED_BY: "subject to a named regulator or export-control authority",
}

ENTITY_DESCRIPTIONS: dict[EntityType, str] = {
    E.COMPANY: "a legal company or corporate group (not a division or brand)",
    E.PERSON: "a named individual",
    E.PRODUCT: "a named product or product family (H100, EPYC, Snapdragon)",
    E.TECHNOLOGY: "a named technology, process node, architecture or standard (CUDA, 3nm, HBM3E, Arm)",
    E.MARKET: "an end market or customer segment (data center, automotive, PC, smartphone)",
    E.LOCATION: "a country, region or city",
    E.REGULATOR: "a government agency or regulatory body (U.S. Department of Commerce, SEC, FTC)",
    E.FACILITY: "a named fab, plant or data center",
}


class ExtractedEntity(BaseModel):
    name: str = Field(description="Most complete name as written in the chunk")
    type: EntityType


class ExtractedRelation(BaseModel):
    type: RelType
    source: str = Field(description="Must exactly match the name of an entity in `entities`")
    target: str = Field(description="Must exactly match the name of an entity in `entities`")
    source_chunk_id: str
    confidence: float = Field(description="0.0-1.0: how explicitly the text states this relationship")
    evidence: str = Field(description="Short verbatim quote from the chunk that states the relationship")
    role: str | None = Field(default=None, description="EXECUTIVE_OF only: title such as CEO")
    year: int | None = Field(default=None, description="ACQUIRED only: year of the deal if stated")


class ChunkExtraction(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]


def _squash(text: str) -> str:
    return re.sub(r"\W+", " ", text).strip().lower()


def validate_extraction(x: ChunkExtraction, chunk_id: str, chunk_text: str) -> list[str]:
    """Semantic checks that a JSON schema can't express. Returned errors are fed back to the model."""
    errors: list[str] = []
    types = {e.name: e.type for e in x.entities}
    squashed_chunk = _squash(chunk_text)
    for i, r in enumerate(x.relations):
        where = f"relations[{i}] ({r.source} -{r.type}-> {r.target})"
        if r.source_chunk_id != chunk_id:
            errors.append(f"{where}: source_chunk_id must be '{chunk_id}'")
        if r.source not in types or r.target not in types:
            errors.append(f"{where}: source and target must exactly match names listed in entities")
            continue
        if (types[r.source], types[r.target]) not in REL_SIGNATURES[r.type]:
            allowed = ", ".join(f"{a}->{b}" for a, b in sorted(REL_SIGNATURES[r.type]))
            errors.append(f"{where}: {types[r.source]}->{types[r.target]} not allowed; allowed: {allowed}")
        if not 0.0 <= r.confidence <= 1.0:
            errors.append(f"{where}: confidence must be between 0 and 1")
        if _squash(r.evidence) not in squashed_chunk:
            errors.append(f"{where}: evidence must be a verbatim quote from the chunk")
    return errors
