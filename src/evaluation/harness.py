"""Evaluation harness (spec §5): run every system over the held-out set,
score against ground truth, produce a comparison report.
"""
from dataclasses import dataclass

from src.evaluation.metrics import MetricScores, average_scores, compute_corpus_metrics
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
    outputs_by_system: dict[str, list] = {
        "zero_shot": [], "retrieval_only": [], "multiagent_no_solver": [], "cadence_full": [],
    }

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
        outputs_by_system["cadence_full"].append(
            run_cadence_full(
                context, retriever, record.record_id, client, graph, tactics, quality_attributes,
                k=k, max_rounds=max_rounds, tactic_budget=tactic_budget,
                max_repair_iterations=max_repair_iterations,
            )
        )

    system_reports = []
    for system_name, outputs in outputs_by_system.items():
        generated = [o.generated_text for o in outputs]
        scores = compute_corpus_metrics(generated, references)
        feasibilities = [o.is_feasible for o in outputs if o.is_feasible is not None]
        repairs = [o.repair_iterations for o in outputs if o.repair_iterations is not None]
        system_reports.append(
            SystemReport(
                system_name=system_name,
                average_scores=average_scores(scores),
                feasibility_rate=(sum(feasibilities) / len(feasibilities)) if feasibilities else None,
                average_repair_iterations=(sum(repairs) / len(repairs)) if repairs else None,
            )
        )

    return EvaluationReport(system_reports=system_reports, n_items=len(test_records))
