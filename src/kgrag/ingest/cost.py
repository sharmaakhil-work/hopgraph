"""Estimate extraction cost before spending anything. count_tokens is free."""

import random
import statistics
from collections import defaultdict
from dataclasses import dataclass

from kgrag.config import BATCH_DISCOUNT, CACHE_READ_MULT, PRICES
from kgrag.ingest.prompts import EXTRACTION_SYSTEM, extraction_user_message
from kgrag.llm import LLM
from kgrag.models import Chunk, Document

# Minimum cacheable prefix for Haiku 4.5; shorter system prompts silently don't cache.
HAIKU_MIN_CACHE_TOKENS = 4096
# Observed extraction output size; replaced by the real mean once some chunks are cached.
DEFAULT_OUTPUT_TOKENS_PER_CHUNK = 700


@dataclass
class Estimate:
    model: str
    chunks: int
    system_tokens: int
    mean_user_tokens: float
    output_tokens_per_chunk: float
    system_cached: bool
    per_doc: dict[str, float]

    @property
    def total_usd(self) -> float:
        return sum(self.per_doc.values())

    @property
    def batch_usd(self) -> float:
        return self.total_usd * BATCH_DISCOUNT


def estimate(
    llm: LLM,
    model: str,
    docs: list[Document],
    chunks: list[Chunk],
    sample_size: int = 20,
    output_tokens_per_chunk: float | None = None,
) -> Estimate:
    by_id = {d.doc_id: d for d in docs}
    rng = random.Random(0)
    sample = rng.sample(chunks, min(sample_size, len(chunks)))

    system_tokens = llm.count_tokens(model=model, system=EXTRACTION_SYSTEM, user=".")
    user_tokens = []
    for c in sample:
        d = by_id[c.doc_id]
        total = llm.count_tokens(
            model=model,
            system=EXTRACTION_SYSTEM,
            user=extraction_user_message(c.id, d.ticker, d.company_name, c.section_path, c.text),
        )
        user_tokens.append(total - system_tokens)
    mean_user = statistics.mean(user_tokens) if user_tokens else 0
    chars_per_token = statistics.mean(len(c.text) for c in sample) / max(mean_user, 1)

    out_tokens = output_tokens_per_chunk or DEFAULT_OUTPUT_TOKENS_PER_CHUNK
    inp, out = PRICES[model]
    cached = model != "claude-haiku-4-5" or system_tokens >= HAIKU_MIN_CACHE_TOKENS
    system_cost = system_tokens * inp * (CACHE_READ_MULT if cached else 1.0)

    per_doc: dict[str, float] = defaultdict(float)
    for c in chunks:
        user = len(c.text) / chars_per_token
        per_doc[c.doc_id] += (system_cost + user * inp + out_tokens * out) / 1_000_000
    return Estimate(
        model=model,
        chunks=len(chunks),
        system_tokens=system_tokens,
        mean_user_tokens=mean_user,
        output_tokens_per_chunk=out_tokens,
        system_cached=cached,
        per_doc=dict(per_doc),
    )
