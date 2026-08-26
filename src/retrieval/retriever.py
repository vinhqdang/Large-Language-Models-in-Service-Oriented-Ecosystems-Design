"""CADENCE Stage 1: embed a decision context, retrieve top-k precedent ADRs."""
from src.retrieval.embeddings import embed_texts
from src.retrieval.index import VectorIndex
from src.retrieval.records import ADRRecord


class Retriever:
    def __init__(self, records: list[ADRRecord], embeddings, model):
        if len(records) != len(embeddings):
            raise ValueError(
                f"records/embeddings length mismatch: {len(records)} vs {len(embeddings)}"
            )
        self._records = records
        self._model = model
        self._index = VectorIndex.build(embeddings)

    def retrieve(self, query_text: str, k: int = 5) -> list[ADRRecord]:
        query_vector = embed_texts([query_text], self._model)[0]
        results = self._index.query(query_vector, k)
        return [self._records[idx] for idx, _similarity in results]
