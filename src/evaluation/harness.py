"""Evaluation harness (spec §5): run every system over the held-out set,
score against ground truth, produce a comparison report.
"""
from dataclasses import dataclass

from src.evaluation.metrics import MetricScores, average_scores, compute_corpus_metrics, load_bertscorer
from src.evaluation.systems import (
    run_cadence_full, run_multiagent_no_solver, run_retrieval_only, run_zero_shot,
)
from src.retrieval.records import ADRRecord
from src.retrieval.retriever import Retriever


@dataclass(frozen=True)
class SystemReport:
    system_name: str
    average_scores: MetricScores
    feasibility_rate: float | None
    average_repair_iterations: float | None


@dataclass(frozen=True)
class EvaluationReport:
    system_reports: list[SystemReport]
    n_items: int


def _build_system_report(system_name: str, outputs: list, references: list[str], scorer) -> SystemReport:
    generated = [o.generated_text for o in outputs]
    scores = compute_corpus_metrics(generated, references, scorer=scorer)
    feasibilities = [o.is_feasible for o in outputs if o.is_feasible is not None]
    repairs = [o.repair_iterations for o in outputs if o.repair_iterations is not None]
    return SystemReport(
        system_name=system_name,
        average_scores=average_scores(scores),
        feasibility_rate=(sum(feasibilities) / len(feasibilities)) if feasibilities else None,
        average_repair_iterations=(sum(repairs) / len(repairs)) if repairs else None,
    )


def _run_baseline_systems(
    test_records: list[ADRRecord], retriever: Retriever, client, graph,
    quality_attributes: tuple[str, ...], k: int, max_rounds: int,
) -> dict[str, list]:
    outputs_by_system: dict[str, list] = {"zero_shot": [], "retrieval_only": [], "multiagent_no_solver": []}
    for record in test_records:
        context = record.title
        outputs_by_system["zero_shot"].append(run_zero_shot(context, client))
        outputs_by_system["retrieval_only"].append(
            run_retrieval_only(context, retriever, record.record_id, client, k=k)
        )
        outputs_by_system["multiagent_no_solver"].append(
            run_multiagent_no_solver(
                context, retriever, record.record_id, client, graph, quality_attributes,
                k=k, max_rounds=max_rounds,
            )
        )
    return outputs_by_system


def _run_cadence_system(
    test_records: list[ADRRecord], retriever: Retriever, client, graph, tactics,
    quality_attributes: tuple[str, ...], k: int, max_rounds: int, tactic_budget: int,
    max_repair_iterations: int,
) -> list:
    outputs = []
    for record in test_records:
        outputs.append(
            run_cadence_full(
                record.title, retriever, record.record_id, client, graph, tactics, quality_attributes,
                k=k, max_rounds=max_rounds, tactic_budget=tactic_budget,
                max_repair_iterations=max_repair_iterations,
            )
        )
    return outputs


def run_evaluation(
    test_records: list[ADRRecord],
    retriever: Retriever,
    client,
    graph,
    tactics,
    quality_attributes: tuple[str, ...],
    k: int = 3,
    max_rounds: int = 2,
    tactic_budget: int = 4,
    max_repair_iterations: int = 2,
) -> EvaluationReport:
    references = [r.raw_text for r in test_records]
    scorer = load_bertscorer()  # load once, reuse across every system report below
    baseline_outputs = _run_baseline_systems(test_records, retriever, client, graph, quality_attributes, k, max_rounds)
    cadence_outputs = _run_cadence_system(
        test_records, retriever, client, graph, tactics, quality_attributes,
        k, max_rounds, tactic_budget, max_repair_iterations,
    )
    outputs_by_system = {**baseline_outputs, "cadence_full": cadence_outputs}

    system_reports = [
        _build_system_report(system_name, outputs, references, scorer)
        for system_name, outputs in outputs_by_system.items()
    ]
    return EvaluationReport(system_reports=system_reports, n_items=len(test_records))


def run_multi_budget_evaluation(
    test_records: list[ADRRecord],
    retriever: Retriever,
    client,
    graph,
    tactics,
    quality_attributes: tuple[str, ...],
    tactic_budgets: tuple[int, ...],
    k: int = 3,
    max_rounds: int = 2,
    max_repair_iterations: int = 2,
) -> dict[int, EvaluationReport]:
    """Compare several `tactic_budget` conditions for `cadence_full` (e.g. an
    achievable budget vs. a deliberately tight one) without paying for the
    budget-independent baselines (`zero_shot`/`retrieval_only`/
    `multiagent_no_solver`) more than once -- they don't take a
    `tactic_budget` argument, so re-running them per budget would just be
    redundant LLM calls, not a fairer comparison.
    """
    references = [r.raw_text for r in test_records]
    scorer = load_bertscorer()  # load once, reuse across every system report below
    baseline_outputs = _run_baseline_systems(test_records, retriever, client, graph, quality_attributes, k, max_rounds)
    baseline_reports = [
        _build_system_report(system_name, outputs, references, scorer)
        for system_name, outputs in baseline_outputs.items()
    ]

    reports_by_budget: dict[int, EvaluationReport] = {}
    for budget in tactic_budgets:
        cadence_outputs = _run_cadence_system(
            test_records, retriever, client, graph, tactics, quality_attributes,
            k, max_rounds, budget, max_repair_iterations,
        )
        cadence_report = _build_system_report("cadence_full", cadence_outputs, references, scorer)
        reports_by_budget[budget] = EvaluationReport(
            system_reports=baseline_reports + [cadence_report], n_items=len(test_records)
        )

    return reports_by_budget
