import hashlib
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from kgrag.models import Chunk, Document


def chunk_id(doc_id: str, section_path: str, ordinal: int) -> str:
    return "c_" + hashlib.sha1(f"{doc_id}|{section_path}|{ordinal}".encode()).hexdigest()[:12]


def chunk_document(doc: Document, chunk_chars: int, overlap: int) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_chars, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks: list[Chunk] = []
    for section in doc.sections:
        for i, piece in enumerate(splitter.split_text(section.text)):
            chunks.append(
                Chunk(
                    id=chunk_id(doc.doc_id, section.path, i),
                    doc_id=doc.doc_id,
                    ticker=doc.ticker,
                    form=doc.form,
                    section_path=section.path,
                    filing_date=doc.filing_date,
                    ordinal=i,
                    text=piece,
                )
            )
    return chunks


def doc_hash(doc: Document) -> str:
    h = hashlib.sha256()
    for s in doc.sections:
        h.update(s.path.encode())
        h.update(s.text.encode())
    return h.hexdigest()[:16]


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    return [Chunk.model_validate(json.loads(line)) for line in path.open()]
