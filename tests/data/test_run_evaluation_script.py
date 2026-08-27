import json

import numpy as np


def test_run_evaluation_script_wires_everything_together(tmp_path, monkeypatch):
    from scripts.run_evaluation import run_evaluation_script

    records_path = tmp_path / "adr_records.jsonl"
    records = [
        {"record_id": f"r{i}", "repo_folder": "r", "repository_url": None, "relative_path": f"{i}.md",
         "sequence_number": 1, "title": f"Title {i}", "raw_text": "x" * 400, "extraction_status": "Verified"}
        for i in range(3)
    ]
    records_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    embeddings_path = tmp_path / "adr_embeddings.npy"
    np.save(embeddings_path, np.array([[1.0, 0.0] for _ in records]))

    class _FakeClient:
        def generate(self, prompt, system=None):
            if "SCORE" in prompt:
                from src.deliberation.knowledge_graph import QUALITY_ATTRIBUTES
                return "\n".join(
                    f"{qa.upper()}_SCORE: 7\n{qa.upper()}_WEAKNESS: none" for qa in QUALITY_ATTRIBUTES
                )
            return "CANDIDATE: Use caching.\nRATIONALE: Improves performance."

    class _FakeEmbeddingModel:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr("scripts.run_evaluation.load_local_hf_client", lambda: _FakeClient())
    monkeypatch.setattr("scripts.run_evaluation.load_embedding_model", lambda: _FakeEmbeddingModel())

    report = run_evaluation_script(
        records_path=records_path, embeddings_path=embeddings_path,
        n_test_items=2, min_length=100, k=1, max_rounds=1, tactic_budget=2, max_repair_iterations=1,
    )

    assert report.n_items == 2
    assert len(report.system_reports) == 4
