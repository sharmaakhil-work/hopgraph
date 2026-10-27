from datetime import date

from pydantic import BaseModel


class Section(BaseModel):
    path: str
    text: str


class Document(BaseModel):
    doc_id: str
    ticker: str
    company_name: str
    cik: int
    form: str
    filing_date: date
    url: str
    sections: list[Section]


class Chunk(BaseModel):
    id: str
    doc_id: str
    ticker: str
    form: str
    section_path: str
    filing_date: date
    ordinal: int
    text: str


class Passage(BaseModel):
    chunk_id: str
    doc_id: str
    ticker: str
    form: str
    section_path: str
    filing_date: date | None
    text: str
    score: float
    entity_ids: list[str] = []
