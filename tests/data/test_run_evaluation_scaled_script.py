import json

import numpy as np


def test_run_evaluation_scaled_script_wires_everything_together(tmp_path, monkeypatch):
    from scripts.run_evaluation_scaled import run_evaluation_scaled_script

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

    monkeypatch.setattr("scripts.run_evaluation_scaled.load_local_hf_client", lambda: _FakeClient())
    monkeypatch.setattr("scripts.run_evaluation_scaled.load_embedding_model", lambda: _FakeEmbeddingModel())

    reports_by_budget = run_evaluation_scaled_script(
        records_path=records_path, embeddings_path=embeddings_path,
        n_test_items=2, min_length=100, k=1, max_rounds=1, tactic_budgets=(2, 5), max_repair_iterations=1,
    )

    assert set(reports_by_budget.keys()) == {2, 5}
    for report in reports_by_budget.values():
        assert report.n_items == 2
        assert len(report.system_reports) == 4


def test_reports_to_json_round_trips_via_dataclasses_asdict(tmp_path, monkeypatch):
    from scripts.run_evaluation_scaled import _reports_to_json, run_evaluation_scaled_script

    records_path = tmp_path / "adr_records.jsonl"
    records = [
        {"record_id": f"r{i}", "repo_folder": "r", "repository_url": None, "relative_path": f"{i}.md",
         "sequence_number": 1, "title": f"Title {i}", "raw_text": "x" * 400, "extraction_status": "Verified"}
        for i in range(2)
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

    monkeypatch.setattr("scripts.run_evaluation_scaled.load_local_hf_client", lambda: _FakeClient())
    monkeypatch.setattr("scripts.run_evaluation_scaled.load_embedding_model", lambda: _FakeEmbeddingModel())

    reports_by_budget = run_evaluation_scaled_script(
        records_path=records_path, embeddings_path=embeddings_path,
        n_test_items=1, min_length=100, k=1, max_rounds=1, tactic_budgets=(3,), max_repair_iterations=1,
    )

    serialized = _reports_to_json(reports_by_budget)
    json.dumps(serialized)  # must be plain-JSON-serializable
    assert set(serialized.keys()) == {"3"}
    assert serialized["3"]["n_items"] == 1
    assert {sr["system_name"] for sr in serialized["3"]["system_reports"]} == {
        "zero_shot", "retrieval_only", "multiagent_no_solver", "cadence_full",
    }
