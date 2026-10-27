from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# USD per million tokens: (input, output). Cache writes bill at 1.25x input, cache reads at 0.1x.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10
BATCH_DISCOUNT = 0.50

# Models that reject output_config.effort
NO_EFFORT_MODELS = {"claude-haiku-4-5"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str | None = None
    edgar_user_agent: str = ""

    extract_model: str = "claude-haiku-4-5"
    router_model: str = "claude-haiku-4-5"
    answer_model: str = "claude-sonnet-5-5"
    answer_effort: str = "low"
    judge_model: str = "claude-sonnet-5-5"

    embed_model: str = "BAAI/bge-base-en-v1.5"
    embed_dim: int = 768

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "kgrag-password"
    pg_dsn: str = "postgresql://kgrag:kgrag@localhost:5433/kgrag"

    data_dir: Path = Path("data")
    eval_dir: Path = Path("eval")

    max_usd: float = 5.0
    chunk_chars: int = 3200
    chunk_overlap: int = 400

    resolution_threshold: float = 0.89  # tuned on eval/resolution_pairs.jsonl (`kgrag tune-resolution`)
    link_threshold: float = 0.82
    router_confidence_threshold: float = 0.7
    vector_top_k: int = 8
    hnsw_ef_search: int = 80
    max_graph_facts: int = 40

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def resolution_path(self) -> Path:
        return self.data_dir / "resolution.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
