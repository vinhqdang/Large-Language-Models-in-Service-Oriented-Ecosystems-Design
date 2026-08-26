import json

import numpy as np


def test_run_index_build_saves_embeddings_and_returns_retriever(tmp_path, monkeypatch):
    from scripts.build_retrieval_index import run_index_build

    records_path = tmp_path / "adr_records.jsonl"
    records_path.write_text(
        json.dumps({
            "record_id": "r/0001-x.md", "repo_folder": "r", "repository_url": None,
            "relative_path": "0001-x.md", "sequence_number": 1, "title": "X",
            "raw_text": "use microservices", "extraction_status": "Verified",
        }) + "\n",
        encoding="utf-8",
    )

    class _FakeModel:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr(
        "scripts.build_retrieval_index.load_embedding_model", lambda: _FakeModel()
    )

    embeddings_path = tmp_path / "adr_embeddings.npy"
    retriever = run_index_build(records_path=records_path, embeddings_out_path=embeddings_path)

    assert embeddings_path.exists()
    saved = np.load(embeddings_path)
    assert saved.shape == (1, 2)

    results = retriever.retrieve("anything", k=1)
    assert results[0].record_id == "r/0001-x.md"
