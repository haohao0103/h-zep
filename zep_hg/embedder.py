"""
Local sentence-transformers embedder for the Zep/Graphiti → HugeGraph adapter.

HugeGraph 1.7.0 has no first-class vector index comparable to Neo4j's
`db.index.vector.queryNodes`, so embeddings are held in-process and vector
search is a cosine scan over the cache. This is acceptable for a PoC-scale
graph (hundreds to low-thousands of nodes); the graph traversal and fulltext
channels still hit HugeGraph for real backend coverage.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


class LocalEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384):
        from sentence_transformers import SentenceTransformer
        self.dim = dim
        self._model = SentenceTransformer(model_name)
        # key (usually uuid or text) -> normalized np.array
        self._cache: dict[str, np.ndarray] = {}
        logger.info("LocalEmbedder loaded: %s (dim=%d)", model_name, dim)

    def embed(self, text: str, key: str | None = None) -> np.ndarray:
        """Embed text; optionally cache under `key` (e.g. entity uuid)."""
        k = key or text
        if k in self._cache:
            return self._cache[k]
        vec = self._model.encode(text, normalize_embeddings=True)
        self._cache[k] = vec
        return vec

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # vectors are L2-normalized

    def _to_arr(self, vec) -> np.ndarray:
        """Convert a list[float] embedding to a normalized np.array (cached)."""
        if isinstance(vec, np.ndarray):
            return vec
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    def top_k(self, query_vec: np.ndarray,
              candidates: Iterable[tuple[str, np.ndarray]],
              k: int = 10) -> list[tuple[str, float]]:
        """Return (key, score) sorted by cosine similarity desc."""
        scored = [(key, self.cosine(query_vec, vec)) for key, vec in candidates]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
