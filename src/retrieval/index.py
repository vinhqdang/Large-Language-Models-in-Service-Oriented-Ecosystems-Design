"""Nearest-neighbor vector index over ADR embeddings.

Uses sklearn's NearestNeighbors (cosine metric) rather than a dedicated
vector-search library: the corpus is ~6,000 vectors, small enough that
exact brute-force cosine search has no practical downside, and sklearn is
already a project dependency — no need to add faiss for this scale.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


class VectorIndex:
    def __init__(self, embeddings: np.ndarray, model: NearestNeighbors):
        self._embeddings = embeddings
        self._model = model

    @classmethod
    def build(cls, embeddings: np.ndarray) -> "VectorIndex":
        model = NearestNeighbors(metric="cosine")
        model.fit(embeddings)
        return cls(embeddings, model)

    def query(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = max(0, min(k, len(self._embeddings)))
        if k == 0:
            return []
        distances, indices = self._model.kneighbors(vector.reshape(1, -1), n_neighbors=k)
        similarities = 1 - distances[0]
        return list(zip(indices[0].tolist(), similarities.tolist()))
