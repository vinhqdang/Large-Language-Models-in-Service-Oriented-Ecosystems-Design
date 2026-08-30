import json


def test_adr_to_json_serializes_full_final_adr():
    from scripts.run_worked_example import _adr_to_json
    from src.critique.finalize import FinalADR, UtilityScore
    from src.deliberation.agent import AgentPosition

    adr = FinalADR(
        decision="Use caching.",
        rationale="Improves performance.",
        utility_scores=[
            UtilityScore(
                quality_attribute="performance", structural_component=1.0,
                qualitative_component=8.0, combined_score=9.0, weakness=None,
            ),
        ],
        overall_score=9.0,
        residual_weaknesses=[],
        is_feasible=True,
        selected_tactics=["Caching"],
        covered_quality_attributes=["performance"],
        uncovered_quality_attributes=[],
        repair_iterations=0,
        solver_caveat=None,
        precedent_titles=["Use read replicas"],
        deliberation_transcript=[
            AgentPosition(quality_attribute="performance", round_number=1, stance="propose", content="Use caching."),
        ],
    )

    result = _adr_to_json("Sample decision context.", adr)

    json.dumps(result)  # must be plain-JSON-serializable
    assert result["decision_context"] == "Sample decision context."
    assert result["precedent_titles"] == ["Use read replicas"]
    assert result["deliberation_transcript"] == [
        {"quality_attribute": "performance", "round_number": 1, "stance": "propose", "content": "Use caching."}
    ]
    assert result["selected_tactics"] == ["Caching"]
    assert result["is_feasible"] is True
    assert result["decision"] == "Use caching."
    assert result["utility_scores"][0]["combined_score"] == 9.0
    assert result["residual_weaknesses"] == []


def test_run_worked_example_wires_demo_and_writes_results(tmp_path, monkeypatch):
    import numpy as np

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

    from scripts.run_cadence_demo import run_cadence_demo
    from scripts.run_worked_example import _adr_to_json

    adr = run_cadence_demo(
        records_path=records_path, embeddings_path=embeddings_path,
        context="Sample decision context.", max_rounds=1, tactic_budget=2, max_repair_iterations=1,
    )
    result = _adr_to_json("Sample decision context.", adr)
    out_path = tmp_path / "worked_example.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["decision"] == "We will use caching."
    assert loaded["decision_context"] == "Sample decision context."
