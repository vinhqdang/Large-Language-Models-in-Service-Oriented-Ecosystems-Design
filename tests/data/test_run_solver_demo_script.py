import json

import numpy as np


def test_run_solver_demo_wires_all_three_stages_together(tmp_path, monkeypatch):
    from scripts.run_solver_demo import run_solver_demo

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
            return "CANDIDATE: We will use caching.\nRATIONALE: Improves performance within budget."

    class _FakeEmbeddingModel:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr("scripts.run_solver_demo.load_local_hf_client", lambda: _FakeClient())
    monkeypatch.setattr("scripts.run_solver_demo.load_embedding_model", lambda: _FakeEmbeddingModel())

    result = run_solver_demo(
        records_path=records_path, embeddings_path=embeddings_path,
        context="Sample decision context.", max_rounds=1, tactic_budget=2, max_repair_iterations=1,
    )

    assert result.final_candidate == "We will use caching."
