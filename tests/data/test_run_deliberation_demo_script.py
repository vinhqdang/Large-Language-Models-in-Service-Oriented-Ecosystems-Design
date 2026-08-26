import json

import numpy as np


def test_run_demo_wires_retriever_and_orchestrator_together(tmp_path, monkeypatch):
    from scripts.run_deliberation_demo import run_demo

    records_path = tmp_path / "adr_records.jsonl"
    records_path.write_text(
        json.dumps({
            "record_id": "r/1.md", "repo_folder": "r", "repository_url": None,
            "relative_path": "1.md", "sequence_number": 1, "title": "Use read replicas",
            "raw_text": "text", "extraction_status": "Verified",
        }) + "\n",
        encoding="utf-8",
    )
    embeddings_path = tmp_path / "adr_embeddings.npy"
    np.save(embeddings_path, np.array([[1.0, 0.0]]))

    class _FakeClient:
        def generate(self, prompt, system=None):
            return "CANDIDATE: c\nRATIONALE: r"

    class _FakeEmbeddingModel:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr("scripts.run_deliberation_demo.load_local_hf_client", lambda: _FakeClient())
    monkeypatch.setattr("scripts.run_deliberation_demo.load_embedding_model", lambda: _FakeEmbeddingModel())

    result = run_demo(
        records_path=records_path, embeddings_path=embeddings_path,
        context="Sample decision context.", max_rounds=1,
    )

    assert result.converged_candidate == "c"
    assert result.rationale == "r"
    assert len(result.transcript) == 5  # 5 quality-attribute agents, 1 propose round each
