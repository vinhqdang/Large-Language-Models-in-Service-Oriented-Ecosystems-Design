import numpy as np

import src.evaluation.harness as harness_module
from src.deliberation.knowledge_graph import TACTICS, build_knowledge_graph, QUALITY_ATTRIBUTES
from src.retrieval.records import ADRRecord
from src.retrieval.retriever import Retriever
from src.evaluation.harness import EvaluationReport, run_evaluation, run_multi_budget_evaluation


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


def test_run_multi_budget_evaluation_runs_baselines_once_across_budgets(monkeypatch):
    test_records = [_record("t1", "Use caching for performance and low latency.")]
    all_records = test_records + [_record("other", "Some other precedent.")]
    embeddings = np.array([[1.0, 0.0] for _ in all_records])
    retriever = Retriever(all_records, embeddings, _FakeEmbeddingModel())
    graph = build_knowledge_graph(TACTICS)
    client = _AllPurposeClient()

    zero_shot_calls = []
    original_zero_shot = harness_module.run_zero_shot

    def _counting_zero_shot(*args, **kwargs):
        zero_shot_calls.append(1)
        return original_zero_shot(*args, **kwargs)

    monkeypatch.setattr(harness_module, "run_zero_shot", _counting_zero_shot)

    reports = run_multi_budget_evaluation(
        test_records=test_records, retriever=retriever, client=client, graph=graph,
        tactics=TACTICS, quality_attributes=QUALITY_ATTRIBUTES,
        tactic_budgets=(2, 5), k=1, max_rounds=1, max_repair_iterations=1,
    )

    assert len(zero_shot_calls) == 1  # baseline ran once, not once per requested budget
    assert set(reports.keys()) == {2, 5}
    for budget, report in reports.items():
        assert report.n_items == 1
        assert {r.system_name for r in report.system_reports} == {
            "zero_shot", "retrieval_only", "multiagent_no_solver", "cadence_no_critique", "cadence_full",
        }
        cadence_report = next(r for r in report.system_reports if r.system_name == "cadence_full")
        assert cadence_report.feasibility_rate is not None

    per_budget_systems = {"cadence_no_critique", "cadence_full"}
    baselines_at_budget_2 = {
        r.system_name: r for r in reports[2].system_reports if r.system_name not in per_budget_systems
    }
    baselines_at_budget_5 = {
        r.system_name: r for r in reports[5].system_reports if r.system_name not in per_budget_systems
    }
    assert baselines_at_budget_2 == baselines_at_budget_5


def test_run_multi_budget_evaluation_invokes_on_budget_complete_callback_incrementally():
    test_records = [_record("t1", "Use caching for performance and low latency.")]
    all_records = test_records + [_record("other", "Some other precedent.")]
    embeddings = np.array([[1.0, 0.0] for _ in all_records])
    retriever = Retriever(all_records, embeddings, _FakeEmbeddingModel())
    graph = build_knowledge_graph(TACTICS)
    client = _AllPurposeClient()

    completed: list[tuple[int, int]] = []

    def _on_budget_complete(budget, report):
        completed.append((budget, report.n_items))

    reports = run_multi_budget_evaluation(
        test_records=test_records, retriever=retriever, client=client, graph=graph,
        tactics=TACTICS, quality_attributes=QUALITY_ATTRIBUTES,
        tactic_budgets=(2, 5), k=1, max_rounds=1, max_repair_iterations=1,
        on_budget_complete=_on_budget_complete,
    )

    # Called once per budget, in order, immediately after that budget's report is ready --
    # so a caller can persist partial results without waiting for every budget to finish.
    assert completed == [(2, 1), (5, 1)]
    assert set(reports.keys()) == {2, 5}


def test_run_multi_budget_evaluation_includes_cadence_no_critique_ablation():
    test_records = [_record("t1", "Use caching for performance and low latency.")]
    all_records = test_records + [_record("other", "Some other precedent.")]
    embeddings = np.array([[1.0, 0.0] for _ in all_records])
    retriever = Retriever(all_records, embeddings, _FakeEmbeddingModel())
    graph = build_knowledge_graph(TACTICS)
    client = _AllPurposeClient()

    reports = run_multi_budget_evaluation(
        test_records=test_records, retriever=retriever, client=client, graph=graph,
        tactics=TACTICS, quality_attributes=QUALITY_ATTRIBUTES,
        tactic_budgets=(2, 5), k=1, max_rounds=1, max_repair_iterations=1,
    )

    for budget, report in reports.items():
        assert {r.system_name for r in report.system_reports} == {
            "zero_shot", "retrieval_only", "multiagent_no_solver", "cadence_no_critique", "cadence_full",
        }
        no_critique_report = next(r for r in report.system_reports if r.system_name == "cadence_no_critique")
        # cadence_no_critique includes Stage 3 (solver+repair), so -- like
        # cadence_full, and unlike the 3 budget-independent baselines --
        # it must be re-run per budget, not shared across budgets.
        assert no_critique_report.feasibility_rate is not None
        assert no_critique_report.average_repair_iterations is not None


def test_run_multi_budget_evaluation_reruns_no_critique_per_budget_not_shared(monkeypatch):
    test_records = [_record("t1", "Use caching for performance and low latency.")]
    all_records = test_records + [_record("other", "Some other precedent.")]
    embeddings = np.array([[1.0, 0.0] for _ in all_records])
    retriever = Retriever(all_records, embeddings, _FakeEmbeddingModel())
    graph = build_knowledge_graph(TACTICS)
    client = _AllPurposeClient()

    no_critique_calls = []
    original = harness_module.run_cadence_no_critique

    def _counting_no_critique(*args, **kwargs):
        no_critique_calls.append(kwargs.get("tactic_budget"))
        return original(*args, **kwargs)

    monkeypatch.setattr(harness_module, "run_cadence_no_critique", _counting_no_critique)

    run_multi_budget_evaluation(
        test_records=test_records, retriever=retriever, client=client, graph=graph,
        tactics=TACTICS, quality_attributes=QUALITY_ATTRIBUTES,
        tactic_budgets=(2, 5), k=1, max_rounds=1, max_repair_iterations=1,
    )

    assert no_critique_calls == [2, 5]  # once per budget, unlike the shared baselines


def test_run_multi_budget_evaluation_works_without_a_callback():
    test_records = [_record("t1", "Use caching for performance and low latency.")]
    all_records = test_records + [_record("other", "Some other precedent.")]
    embeddings = np.array([[1.0, 0.0] for _ in all_records])
    retriever = Retriever(all_records, embeddings, _FakeEmbeddingModel())
    graph = build_knowledge_graph(TACTICS)
    client = _AllPurposeClient()

    reports = run_multi_budget_evaluation(
        test_records=test_records, retriever=retriever, client=client, graph=graph,
        tactics=TACTICS, quality_attributes=QUALITY_ATTRIBUTES,
        tactic_budgets=(2,), k=1, max_rounds=1, max_repair_iterations=1,
    )

    assert set(reports.keys()) == {2}
