"""Chunk-level entity/relationship extraction with Claude.

Results are cached per (document content hash, prompt version, model): re-running ingestion on unchanged
filings makes zero API calls.
"""

import hashlib
import json
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from kgrag.ingest.chunk import doc_hash
from kgrag.ingest.prompts import EXTRACTION_SYSTEM, PROMPT_VERSION, extraction_user_message
from kgrag.llm import LLM, BudgetExceeded, StructuredOutputError, Usage, strict_schema
from kgrag.models import Chunk, Document
from kgrag.ontology import ChunkExtraction, validate_extraction

MAX_TOKENS = 4096


@dataclass
class ExtractStats:
    chunks: int = 0
    cached: int = 0
    extracted: int = 0
    failed: int = 0
    usd: float = 0.0


class ExtractionCache:
    def __init__(self, root: Path, model: str):
        self.root = root / "extract"
        self.model = model

    def doc_dir(self, doc: Document) -> Path:
        key = hashlib.sha256(f"{doc_hash(doc)}|{PROMPT_VERSION}|{self.model}".encode()).hexdigest()[:20]
        return self.root / f"{doc.doc_id}__{key}"

    def get(self, doc: Document, chunk_id: str) -> ChunkExtraction | None:
        p = self.doc_dir(doc) / f"{chunk_id}.json"
        return ChunkExtraction.model_validate_json(p.read_text()) if p.exists() else None

    def put(self, doc: Document, chunk_id: str, x: ChunkExtraction) -> None:
        d = self.doc_dir(doc)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{chunk_id}.json").write_text(x.model_dump_json())


class Extractor:
    def __init__(self, llm: LLM, model: str, cache: ExtractionCache, failures_path: Path):
        self.llm = llm
        self.model = model
        self.cache = cache
        self.failures_path = failures_path

    def _user(self, chunk: Chunk, doc: Document) -> str:
        return extraction_user_message(chunk.id, doc.ticker, doc.company_name, chunk.section_path, chunk.text)

    def _log_failure(self, chunk: Chunk, errors: list[str]) -> None:
        self.failures_path.parent.mkdir(parents=True, exist_ok=True)
        with self.failures_path.open("a") as f:
            f.write(json.dumps({"chunk_id": chunk.id, "doc_id": chunk.doc_id, "errors": errors}) + "\n")

    def extract_chunk(self, chunk: Chunk, doc: Document) -> tuple[ChunkExtraction | None, Usage | None]:
        try:
            x, usage = self.llm.structured(
                model=self.model,
                system=EXTRACTION_SYSTEM,
                user=self._user(chunk, doc),
                output=ChunkExtraction,
                purpose="extract",
                max_tokens=MAX_TOKENS,
                validator=lambda x: validate_extraction(x, chunk.id, chunk.text),
            )
        except StructuredOutputError as e:
            self._log_failure(chunk, e.errors)
            return None, None
        self.cache.put(doc, chunk.id, x)
        return x, usage

    def run(self, docs: list[Document], chunks: list[Chunk], workers: int = 4) -> ExtractStats:
        by_id = {d.doc_id: d for d in docs}
        stats = ExtractStats()
        todo: list[tuple[Chunk, Document]] = []
        for c in chunks:
            stats.chunks += 1
            doc = by_id[c.doc_id]
            if self.cache.get(doc, c.id) is not None:
                stats.cached += 1
            else:
                todo.append((c, doc))
        start_spend = self.llm.meter.spent_usd
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self.extract_chunk, c, d) for c, d in todo]
            try:
                for i, fut in enumerate(as_completed(futures), 1):
                    x, _ = fut.result()
                    if x is None:
                        stats.failed += 1
                    else:
                        stats.extracted += 1
                    if i % 25 == 0:
                        print(f"  {i}/{len(todo)} chunks, ${self.llm.meter.spent_usd - start_spend:.2f}")
            except BudgetExceeded as e:
                for f in futures:
                    f.cancel()
                print(
                    f"[budget] {e}. Completed chunks are cached; re-run with a higher --max-usd to continue."
                )
        stats.usd = self.llm.meter.spent_usd - start_spend
        return stats

    def run_batch(self, docs: list[Document], chunks: list[Chunk], poll_seconds: int = 30) -> ExtractStats:
        """Message Batches API: 50% cheaper, results within ~1 hour. Invalid results get a synchronous retry."""
        by_id = {d.doc_id: d for d in docs}
        stats = ExtractStats(chunks=len(chunks))
        todo = {c.id: c for c in chunks if self.cache.get(by_id[c.doc_id], c.id) is None}
        stats.cached = len(chunks) - len(todo)
        start_spend = self.llm.meter.spent_usd
        if not todo:
            return stats
        schema = strict_schema(ChunkExtraction)
        requests = [
            {
                "custom_id": c.id,
                "params": self.llm.request_params(
                    model=self.model,
                    system=EXTRACTION_SYSTEM,
                    messages=[{"role": "user", "content": self._user(c, by_id[c.doc_id])}],
                    schema=schema,
                    max_tokens=MAX_TOKENS,
                ),
            }
            for c in todo.values()
        ]
        client = self.llm.client
        batch = client.messages.batches.create(requests=requests)
        print(f"[batch] submitted {batch.id} with {len(requests)} requests")
        while (batch := client.messages.batches.retrieve(batch.id)).processing_status != "ended":
            print(f"  status={batch.processing_status} processing={batch.request_counts.processing}")
            time.sleep(poll_seconds)

        retry: list[Chunk] = []
        for result in client.messages.batches.results(batch.id):
            chunk = todo[result.custom_id]
            if result.result.type != "succeeded":
                retry.append(chunk)
                continue
            msg = result.result.message
            self.llm.meter.record(Usage.from_response(self.model, msg.usage, batch=True), "extract-batch")
            text = next((b.text for b in msg.content if b.type == "text"), "")
            errors = LLM._validate(
                text, ChunkExtraction, lambda x, c=chunk: validate_extraction(x, c.id, c.text)
            )
            if errors:
                retry.append(chunk)
                continue
            self.cache.put(by_id[chunk.doc_id], chunk.id, ChunkExtraction.model_validate_json(text))
            stats.extracted += 1
        for chunk in retry:
            x, _ = self.extract_chunk(chunk, by_id[chunk.doc_id])
            stats.extracted += x is not None
            stats.failed += x is None
        stats.usd = self.llm.meter.spent_usd - start_spend
        return stats


def load_extractions(
    cache: ExtractionCache, docs: Iterable[Document], chunks: list[Chunk]
) -> dict[str, ChunkExtraction]:
    by_id = {d.doc_id: d for d in docs}
    out: dict[str, ChunkExtraction] = {}
    for c in chunks:
        if c.doc_id in by_id and (x := cache.get(by_id[c.doc_id], c.id)) is not None:
            out[c.id] = x
    return out
