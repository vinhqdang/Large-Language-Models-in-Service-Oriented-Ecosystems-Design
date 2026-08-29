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
        records_path=records_path, embeddings_path=embeddings_path, seed=1,
        n_test_items=2, min_length=100, k=1, max_rounds=1, tactic_budgets=(2, 5), max_repair_iterations=1,
    )

    assert set(reports_by_budget.keys()) == {2, 5}


def test_run_evaluation_scaled_script_requires_seed_and_produces_a_disjoint_sample(monkeypatch):
    """Regression: scripts/run_evaluation.py's pilot run relies on
    sample_test_set's own default (seed=42) over the real corpus, and an
    earlier version of run_evaluation_scaled_script had no seed parameter
    at all -- both scripts silently scored the exact same 3 held-out
    items, undermining any claim that a pattern held across independent
    samples. `seed` is now a required parameter threaded through to
    sample_test_set, specifically so a caller cannot forget to set it
    (an optional default was tried first and rejected on review: two
    independently hardcoded default literals in two different files is
    the same silent-default risk that caused the original bug).

    Uses the real committed corpus (not a tmp_path fixture like this
    file's other tests) because the bug and its fix are specifically
    about *this* corpus's real Verified/min_length-eligible record count
    and the concrete seed=42-vs-other divergence over it -- a 2-3-record
    synthetic fixture can't demonstrate that."""
    from pathlib import Path

    import numpy as np

    from scripts.run_evaluation_scaled import run_evaluation_scaled_script

    project_root = Path(__file__).resolve().parent.parent.parent
    records_path = project_root / "data" / "processed" / "adr_records.jsonl"
    embeddings_path = project_root / "data" / "processed" / "adr_embeddings.npy"
    embedding_dim = np.load(embeddings_path).shape[1]

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
            # Must match the real committed embeddings' dimensionality --
            # Retriever's NearestNeighbors index is fit on the real
            # adr_embeddings.npy, so a query vector of the wrong
            # dimensionality raises a sklearn ValueError before this
            # test's own assertions ever run.
            return np.zeros((len(texts), embedding_dim))

    monkeypatch.setattr("scripts.run_evaluation_scaled.load_local_hf_client", lambda: _FakeClient())
    monkeypatch.setattr("scripts.run_evaluation_scaled.load_embedding_model", lambda: _FakeEmbeddingModel())

    from src.evaluation.held_out_set import load_verified_records, sample_test_set
    verified = load_verified_records(records_path)
    # Mirrors the two real scripts' actual effective seeds: the pilot
    # script's implicit default (42) versus the scaled script's __main__
    # value (43) -- this is the concrete collision the bug report was
    # about, not an arbitrary pair of seeds.
    pilot_default_ids = {r.record_id for r in sample_test_set(verified, n=3, min_length=300)}
    scaled_main_ids = {r.record_id for r in sample_test_set(verified, n=3, min_length=300, seed=43)}
    assert pilot_default_ids != scaled_main_ids

    reports_by_budget = run_evaluation_scaled_script(
        records_path=records_path, embeddings_path=embeddings_path, seed=43,
        n_test_items=3, min_length=300, k=1, max_rounds=1, tactic_budgets=(5,), max_repair_iterations=1,
    )

    assert reports_by_budget[5].n_items == 3
    assert len(reports_by_budget[5].system_reports) == 5


def test_run_evaluation_scaled_script_passes_through_on_budget_complete(tmp_path, monkeypatch):
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

    completed = []

    reports_by_budget = run_evaluation_scaled_script(
        records_path=records_path, embeddings_path=embeddings_path, seed=1,
        n_test_items=2, min_length=100, k=1, max_rounds=1, tactic_budgets=(2, 5), max_repair_iterations=1,
        on_budget_complete=lambda budget, report: completed.append(budget),
    )

    assert completed == [2, 5]
    assert set(reports_by_budget.keys()) == {2, 5}


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
        records_path=records_path, embeddings_path=embeddings_path, seed=1,
        n_test_items=1, min_length=100, k=1, max_rounds=1, tactic_budgets=(3,), max_repair_iterations=1,
    )

    serialized = _reports_to_json(reports_by_budget)
    json.dumps(serialized)  # must be plain-JSON-serializable
    assert set(serialized.keys()) == {"3"}
    assert serialized["3"]["n_items"] == 1
    assert {sr["system_name"] for sr in serialized["3"]["system_reports"]} == {
        "zero_shot", "retrieval_only", "multiagent_no_solver", "cadence_no_critique", "cadence_full",
    }
