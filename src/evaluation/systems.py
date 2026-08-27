"""Baseline and full-CADENCE system runners for evaluation (spec §5)."""
import warnings
from dataclasses import dataclass

from src.deliberation.agent import QualityAttributeAgent
from src.deliberation.orchestrator import DeliberationOrchestrator
from src.retrieval.records import ADRRecord
from src.retrieval.retriever import Retriever
from src.solver.repair import run_repair_loop
from src.critique.finalize import finalize_decision


@dataclass(frozen=True)
class SystemOutput:
    system_name: str
    generated_text: str
    is_feasible: bool | None
    repair_iterations: int | None


def retrieve_excluding_self(
    retriever: Retriever, query: str, exclude_record_id: str, k: int
) -> list[ADRRecord]:
    results = retriever.retrieve(query, k=k + 1)
    filtered = [r for r in results if r.record_id != exclude_record_id]
    trimmed = filtered[:k]
    if len(trimmed) < k:
        # Not an error -- a small corpus/Retriever can legitimately have
        # fewer than k eligible precedents -- but every system that calls
        # this relies on getting the *same* generation budget for a fair
        # comparison (see the evaluation-harness plan's Self-Review Notes),
        # so a silent shortfall should be visible, not swallowed.
        warnings.warn(
            f"retrieve_excluding_self: requested k={k} but only {len(trimmed)} "
            f"precedents available after excluding {exclude_record_id!r}",
            stacklevel=2,
        )
    return trimmed


def run_zero_shot(context: str, client) -> SystemOutput:
    prompt = (
        f"Decision context:\n{context}\n\n"
        "Write an architectural decision record addressing this context."
    )
    text = client.generate(prompt)
    return SystemOutput("zero_shot", text, is_feasible=None, repair_iterations=None)


def run_retrieval_only(
    context: str, retriever: Retriever, exclude_record_id: str, client, k: int = 3
) -> SystemOutput:
    precedents = retrieve_excluding_self(retriever, context, exclude_record_id, k)
    precedent_lines = "\n".join(f"- {p.title}" for p in precedents) or "(no precedents retrieved)"
    prompt = (
        f"Decision context:\n{context}\n\n"
        f"Precedent decisions from similar past projects:\n{precedent_lines}\n\n"
        "Write an architectural decision record addressing this context, informed by the precedents above."
    )
    text = client.generate(prompt)
    return SystemOutput("retrieval_only", text, is_feasible=None, repair_iterations=None)


def run_multiagent_no_solver(
    context: str,
    retriever: Retriever,
    exclude_record_id: str,
    client,
    graph,
    quality_attributes: tuple[str, ...],
    k: int = 3,
    max_rounds: int = 2,
) -> SystemOutput:
    precedents = retrieve_excluding_self(retriever, context, exclude_record_id, k)
    agents = [QualityAttributeAgent(qa, client, graph) for qa in quality_attributes]
    orchestrator = DeliberationOrchestrator(agents, client, max_rounds=max_rounds)
    result = orchestrator.deliberate(context, precedents)
    text = f"{result.converged_candidate}\n\n{result.rationale}"
    return SystemOutput("multiagent_no_solver", text, is_feasible=None, repair_iterations=None)


def run_cadence_full(
    context: str,
    retriever: Retriever,
    exclude_record_id: str,
    client,
    graph,
    tactics,
    quality_attributes: tuple[str, ...],
    k: int = 3,
    max_rounds: int = 2,
    tactic_budget: int = 4,
    max_repair_iterations: int = 2,
) -> SystemOutput:
    """All 4 stages: retrieval, deliberation, solver verification/repair,
    and self-critique finalization -- the full CADENCE pipeline."""
    precedents = retrieve_excluding_self(retriever, context, exclude_record_id, k)
    agents = [QualityAttributeAgent(qa, client, graph) for qa in quality_attributes]
    orchestrator = DeliberationOrchestrator(agents, client, max_rounds=max_rounds)
    deliberation = orchestrator.deliberate(context, precedents)

    verified = run_repair_loop(
        candidate=deliberation.converged_candidate,
        rationale=deliberation.rationale,
        required_quality_attributes=quality_attributes,
        tactic_budget=tactic_budget,
        quality_attribute_weights={},
        tactics=tactics,
        repair_client=client,
        max_repair_iterations=max_repair_iterations,
    )

    final_adr = finalize_decision(
        verified_decision=verified,
        deliberation_result=deliberation,
        precedents=precedents,
        quality_attributes=quality_attributes,
        tactics=tactics,
        critique_client=client,
    )
    text = f"{final_adr.decision}\n\n{final_adr.rationale}"
    return SystemOutput(
        "cadence_full", text, is_feasible=final_adr.is_feasible, repair_iterations=final_adr.repair_iterations
    )


def run_cadence_no_critique(
    context: str,
    retriever: Retriever,
    exclude_record_id: str,
    client,
    graph,
    tactics,
    quality_attributes: tuple[str, ...],
    k: int = 3,
    max_rounds: int = 2,
    tactic_budget: int = 4,
    max_repair_iterations: int = 2,
) -> SystemOutput:
    """Stage 1-3 only (retrieval + deliberation + solver verification/repair),
    skipping Stage 4's self-critique finalization -- the ablation isolating
    what the critique stage adds beyond solver-verified feasibility alone.
    """
    precedents = retrieve_excluding_self(retriever, context, exclude_record_id, k)
    agents = [QualityAttributeAgent(qa, client, graph) for qa in quality_attributes]
    orchestrator = DeliberationOrchestrator(agents, client, max_rounds=max_rounds)
    deliberation = orchestrator.deliberate(context, precedents)

    verified = run_repair_loop(
        candidate=deliberation.converged_candidate,
        rationale=deliberation.rationale,
        required_quality_attributes=quality_attributes,
        tactic_budget=tactic_budget,
        quality_attribute_weights={},
        tactics=tactics,
        repair_client=client,
        max_repair_iterations=max_repair_iterations,
    )
    text = f"{verified.final_candidate}\n\n{verified.rationale}"
    return SystemOutput(
        "cadence_no_critique", text, is_feasible=verified.is_feasible, repair_iterations=verified.repair_iterations
    )
