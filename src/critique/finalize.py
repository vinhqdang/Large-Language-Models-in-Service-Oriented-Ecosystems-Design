"""Assemble the CADENCE pipeline's terminal output: a FinalADR combining
Stage 4's utility scoring with every prior stage's provenance.
"""
from dataclasses import dataclass

from src.critique.llm_critique import run_qualitative_critique
from src.critique.structural_utility import compute_structural_utility
from src.deliberation.agent import AgentPosition
from src.deliberation.knowledge_graph import Tactic
from src.deliberation.orchestrator import DeliberationResult
from src.retrieval.records import ADRRecord
from src.solver.repair import VerifiedDecision


@dataclass(frozen=True)
class UtilityScore:
    quality_attribute: str
    structural_component: float
    qualitative_component: float
    combined_score: float
    weakness: str | None


@dataclass(frozen=True)
class FinalADR:
    decision: str
    rationale: str
    utility_scores: list[UtilityScore]
    overall_score: float
    residual_weaknesses: list[str]
    is_feasible: bool
    selected_tactics: list[str]
    covered_quality_attributes: list[str]
    uncovered_quality_attributes: list[str]
    repair_iterations: int
    solver_caveat: str | None
    precedent_titles: list[str]
    deliberation_transcript: list[AgentPosition]


def finalize_decision(
    verified_decision: VerifiedDecision,
    deliberation_result: DeliberationResult,
    precedents: list[ADRRecord],
    quality_attributes: tuple[str, ...],
    tactics: list[Tactic],
    critique_client,
) -> FinalADR:
    qualitative_scores = run_qualitative_critique(
        verified_decision.final_candidate, verified_decision.rationale, quality_attributes, critique_client
    )
    qualitative_by_qa = {s.quality_attribute: s for s in qualitative_scores}

    utility_scores = []
    for qa in quality_attributes:
        structural = compute_structural_utility(
            qa, verified_decision.selected_tactics, verified_decision.covered_quality_attributes, tactics
        )
        qualitative = qualitative_by_qa[qa]
        combined = 0.5 * structural * 10 + 0.5 * qualitative.score
        utility_scores.append(
            UtilityScore(qa, structural, qualitative.score, combined, qualitative.weakness)
        )

    overall_score = sum(s.combined_score for s in utility_scores) / len(utility_scores)
    residual_weaknesses = [s.weakness for s in utility_scores if s.weakness is not None]

    return FinalADR(
        decision=verified_decision.final_candidate,
        rationale=verified_decision.rationale,
        utility_scores=utility_scores,
        overall_score=overall_score,
        residual_weaknesses=residual_weaknesses,
        is_feasible=verified_decision.is_feasible,
        selected_tactics=verified_decision.selected_tactics,
        covered_quality_attributes=verified_decision.covered_quality_attributes,
        uncovered_quality_attributes=verified_decision.uncovered_quality_attributes,
        repair_iterations=verified_decision.repair_iterations,
        solver_caveat=verified_decision.caveat,
        precedent_titles=[p.title for p in precedents],
        deliberation_transcript=deliberation_result.transcript,
    )
