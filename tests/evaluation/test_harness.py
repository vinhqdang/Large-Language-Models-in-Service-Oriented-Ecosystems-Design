import numpy as np

from src.deliberation.knowledge_graph import TACTICS, build_knowledge_graph, QUALITY_ATTRIBUTES
from src.retrieval.records import ADRRecord
from src.retrieval.retriever import Retriever
from src.evaluation.harness import EvaluationReport, run_evaluation


class _FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0] for _ in texts])


class _AllPurposeClient:
    def generate(self, prompt, system=None):
        if "SCORE" in prompt:
            return "\n".join(
                f"{qa.upper()}_SCORE: 7\n{qa.upper()}_WEAKNESS: none" for qa in QUALITY_ATTRIBUTES
            )
        return "CANDIDATE: Use caching.\nRATIONALE: Improves performance."


def _record(rid, text):
    return ADRRecord(record_id=rid, repo_folder="r", repository_url=None, relative_path=rid,
                       sequence_number=1, title=f"Title {rid}", raw_text=text, extraction_status="Verified")


def test_run_evaluation_produces_a_report_with_all_four_systems():
    test_records = [_record("t1", "Use caching for performance and low latency."), _record("t2", "Use replicas.")]
    all_records = test_records + [_record("other", "Some other precedent.")]
    embeddings = np.array([[1.0, 0.0] for _ in all_records])
    retriever = Retriever(all_records, embeddings, _FakeEmbeddingModel())
    graph = build_knowledge_graph(TACTICS)
    client = _AllPurposeClient()

    report = run_evaluation(
        test_records=test_records, retriever=retriever, client=client, graph=graph,
        tactics=TACTICS, quality_attributes=QUALITY_ATTRIBUTES,
        k=1, max_rounds=1, tactic_budget=2, max_repair_iterations=1,
    )

    assert isinstance(report, EvaluationReport)
    assert report.n_items == 2
    assert {r.system_name for r in report.system_reports} == {
        "zero_shot", "retrieval_only", "multiagent_no_solver", "cadence_full",
    }
    cadence_report = next(r for r in report.system_reports if r.system_name == "cadence_full")
    assert cadence_report.feasibility_rate is not None
    assert cadence_report.average_repair_iterations is not None
    zero_shot_report = next(r for r in report.system_reports if r.system_name == "zero_shot")
    assert zero_shot_report.feasibility_rate is None
