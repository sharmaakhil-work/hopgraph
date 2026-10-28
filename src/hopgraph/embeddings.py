import hashlib
from typing import Protocol

import numpy as np

# bge models expect this instruction on queries (not on passages)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dim: int

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    def embed_names(self, names: list[str]) -> np.ndarray: ...


class BgeEmbedder:
    def __init__(self, model_name: str, dim: int):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = dim

    def _encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 256,
            convert_to_numpy=True,
        ).astype(np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([BGE_QUERY_PREFIX + text])[0]

    def embed_names(self, names: list[str]) -> np.ndarray:
        return self._encode(names, batch_size=128)


class HashEmbedder:
    """Deterministic character-trigram embedder for tests; similar strings get similar vectors."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = f"  {text.lower()}  "
        for i in range(len(t) - 2):
            h = int(hashlib.md5(t[i : i + 3].encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._one(text)

    def embed_names(self, names: list[str]) -> np.ndarray:
        return self.embed_passages(names)
