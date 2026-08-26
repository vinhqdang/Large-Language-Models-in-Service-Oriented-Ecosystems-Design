import pytest

from src.deliberation.agent import AgentPosition
from src.deliberation.knowledge_graph import Tactic
from src.deliberation.orchestrator import DeliberationResult
from src.retrieval.records import ADRRecord
from src.solver.repair import VerifiedDecision
from src.critique.finalize import FinalADR, finalize_decision
from src.critique.llm_critique import CritiqueParseError


def _catalog():
    return [
        Tactic("Caching", "performance", "d", {}),
        Tactic("Authentication", "security", "d", {}),
    ]


def _verified_decision(**overrides):
    defaults = dict(
        is_feasible=True, final_candidate="Use caching and authentication.",
        rationale="Balances performance and security.",
        selected_tactics=["Caching", "Authentication"],
        covered_quality_attributes=["performance", "security"],
        uncovered_quality_attributes=[], repair_iterations=0, caveat=None,
    )
    defaults.update(overrides)
    return VerifiedDecision(**defaults)


def _deliberation_result():
    positions = [AgentPosition("performance", 1, "propose", "Use caching.")]
    return DeliberationResult(
        converged_candidate="Use caching and authentication.",
        rationale="Balances performance and security.",
        transcript=positions, rounds_run=1,
    )


def _precedent():
    return ADRRecord(
        record_id="r/1.md", repo_folder="r", repository_url=None,
        relative_path="1.md", sequence_number=1, title="Use OAuth2",
        raw_text="...", extraction_status="Verified",
    )


class _FakeCritiqueClient:
    def generate(self, prompt, system=None):
        return (
            "PERFORMANCE_SCORE: 8\nPERFORMANCE_WEAKNESS: none\n"
            "SECURITY_SCORE: 6\nSECURITY_WEAKNESS: Single-factor auth only.\n"
        )


def test_finalize_decision_assembles_full_provenance():
    result = finalize_decision(
        verified_decision=_verified_decision(),
        deliberation_result=_deliberation_result(),
        precedents=[_precedent()],
        quality_attributes=("performance", "security"),
        tactics=_catalog(),
        critique_client=_FakeCritiqueClient(),
    )

    assert isinstance(result, FinalADR)
    assert result.decision == "Use caching and authentication."
    assert result.is_feasible
    assert result.selected_tactics == ["Caching", "Authentication"]
    assert result.precedent_titles == ["Use OAuth2"]
    assert result.deliberation_transcript == _deliberation_result().transcript
    assert result.repair_iterations == 0
    assert result.solver_caveat is None


def test_utility_scores_blend_structural_and_qualitative_components():
    result = finalize_decision(
        verified_decision=_verified_decision(),
        deliberation_result=_deliberation_result(),
        precedents=[_precedent()],
        quality_attributes=("performance", "security"),
        tactics=_catalog(),
        critique_client=_FakeCritiqueClient(),
    )

    performance = next(s for s in result.utility_scores if s.quality_attribute == "performance")
    assert performance.structural_component == 1.0  # covered, no incoming trade-off
    assert performance.qualitative_component == 8.0
    assert performance.combined_score == 0.5 * 1.0 * 10 + 0.5 * 8.0  # == 9.0
    assert performance.weakness is None


def test_residual_weaknesses_collects_non_none_weaknesses_across_attributes():
    result = finalize_decision(
        verified_decision=_verified_decision(),
        deliberation_result=_deliberation_result(),
        precedents=[_precedent()],
        quality_attributes=("performance", "security"),
        tactics=_catalog(),
        critique_client=_FakeCritiqueClient(),
    )

    assert result.residual_weaknesses == ["Single-factor auth only."]


def test_overall_score_is_mean_of_combined_scores():
    result = finalize_decision(
        verified_decision=_verified_decision(),
        deliberation_result=_deliberation_result(),
        precedents=[_precedent()],
        quality_attributes=("performance", "security"),
        tactics=_catalog(),
        critique_client=_FakeCritiqueClient(),
    )

    scores = [s.combined_score for s in result.utility_scores]
    assert result.overall_score == sum(scores) / len(scores)


def test_carries_forward_infeasible_solver_result_with_caveat():
    verified = _verified_decision(
        is_feasible=False, uncovered_quality_attributes=["security"],
        repair_iterations=2, caveat="Could not cover security within budget.",
    )

    result = finalize_decision(
        verified_decision=verified,
        deliberation_result=_deliberation_result(),
        precedents=[_precedent()],
        quality_attributes=("performance", "security"),
        tactics=_catalog(),
        critique_client=_FakeCritiqueClient(),
    )

    assert not result.is_feasible
    assert result.uncovered_quality_attributes == ["security"]
    assert result.repair_iterations == 2
    assert result.solver_caveat == "Could not cover security within budget."


class _FlakyCritiqueClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, prompt, system=None):
        self.calls += 1
        return self.responses.pop(0)


def test_finalize_decision_retries_critique_and_recovers_on_a_later_attempt():
    client = _FlakyCritiqueClient([
        "not a recognizable format at all",
        "PERFORMANCE_SCORE: 8\nPERFORMANCE_WEAKNESS: none\n"
        "SECURITY_SCORE: 6\nSECURITY_WEAKNESS: Single-factor auth only.\n",
    ])

    result = finalize_decision(
        verified_decision=_verified_decision(),
        deliberation_result=_deliberation_result(),
        precedents=[_precedent()],
        quality_attributes=("performance", "security"),
        tactics=_catalog(),
        critique_client=client,
    )

    assert client.calls == 2
    assert result.overall_score > 0


def test_finalize_decision_raises_after_exhausting_critique_retries():
    client = _FlakyCritiqueClient(["bad"] * 5)

    with pytest.raises(CritiqueParseError):
        finalize_decision(
            verified_decision=_verified_decision(),
            deliberation_result=_deliberation_result(),
            precedents=[_precedent()],
            quality_attributes=("performance", "security"),
            tactics=_catalog(),
            critique_client=client,
        )
    assert client.calls == 3  # MAX_CRITIQUE_ATTEMPTS
