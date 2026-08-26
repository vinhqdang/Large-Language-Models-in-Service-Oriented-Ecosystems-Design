import json

import numpy as np


def test_run_cadence_demo_wires_all_four_stages_together(tmp_path, monkeypatch):
    from scripts.run_cadence_demo import run_cadence_demo

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
            if "SCORE" in prompt:
                return "\n".join(
                    f"{qa.upper()}_SCORE: 7\n{qa.upper()}_WEAKNESS: none"
                    for qa in ["performance", "security", "maintainability", "scalability", "cost_operability"]
                )
            return "CANDIDATE: We will use caching.\nRATIONALE: Improves performance within budget."

    class _FakeEmbeddingModel:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr("scripts.run_cadence_demo.load_local_hf_client", lambda: _FakeClient())
    monkeypatch.setattr("scripts.run_cadence_demo.load_embedding_model", lambda: _FakeEmbeddingModel())

    result = run_cadence_demo(
        records_path=records_path, embeddings_path=embeddings_path,
        context="Sample decision context.", max_rounds=1, tactic_budget=2, max_repair_iterations=1,
    )

    assert result.decision == "We will use caching."
    assert len(result.utility_scores) == 5
