import numpy as np
import pytest

from src.retrieval.records import ADRRecord
from src.retrieval.retriever import Retriever


class _FakeModel:
    """Maps fixed strings to fixed vectors so retrieval order is exact and
    deterministic, with no real embedding model involved."""

    _VECTORS = {
        "use microservices": [1.0, 0.0],
        "use a monolith": [0.0, 1.0],
        "use microservices for scale": [0.9, 0.1],
    }

    def encode(self, texts, **kwargs):
        return np.array([self._VECTORS[t] for t in texts])


def _record(record_id, raw_text):
    return ADRRecord(
        record_id=record_id, repo_folder="r", repository_url=None,
        relative_path=record_id, sequence_number=1, title=raw_text,
        raw_text=raw_text, extraction_status="Verified",
    )


def test_retrieve_returns_top_k_records_by_similarity():
    records = [
        _record("a", "use microservices"),
        _record("b", "use a monolith"),
        _record("c", "use microservices for scale"),
    ]
    model = _FakeModel()
    embeddings = np.array([model._VECTORS[r.raw_text] for r in records])
    retriever = Retriever(records, embeddings, model)

    results = retriever.retrieve("use microservices", k=2)

    assert [r.record_id for r in results] == ["a", "c"]


def test_retriever_rejects_mismatched_records_and_embeddings_length():
    records = [_record("a", "use microservices")]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2 rows for 1 record

    with pytest.raises(ValueError, match="length mismatch"):
        Retriever(records, embeddings, _FakeModel())
