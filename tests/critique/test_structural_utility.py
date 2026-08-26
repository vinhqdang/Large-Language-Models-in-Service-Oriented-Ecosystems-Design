from src.deliberation.knowledge_graph import Tactic
from src.critique.structural_utility import compute_structural_utility


def _catalog():
    return [
        Tactic("Caching", "performance", "d", {"maintainability": "n"}),
        Tactic("Authentication", "security", "d", {}),
    ]


def test_covered_attribute_with_no_incoming_trade_offs_scores_full():
    score = compute_structural_utility(
        "security", selected_tactics=["Authentication"],
        covered_quality_attributes=["security"], tactics=_catalog(),
    )
    assert score == 1.0


def test_uncovered_attribute_scores_zero():
    score = compute_structural_utility(
        "scalability", selected_tactics=["Caching"],
        covered_quality_attributes=["performance"], tactics=_catalog(),
    )
    assert score == 0.0


def test_covered_attribute_penalized_by_incoming_trade_off():
    # "maintainability" is covered by nothing here, but exists to show the
    # penalty applies even when the OTHER attribute is the one covered.
    score = compute_structural_utility(
        "maintainability", selected_tactics=["Caching"],
        covered_quality_attributes=["performance"], tactics=_catalog(),
    )
    assert score == 0.0  # already 0 (uncovered), penalty can't go negative


def test_covered_attribute_with_trade_off_from_a_different_selected_tactic():
    tactics = [
        Tactic("Caching", "performance", "d", {"maintainability": "n"}),
        Tactic("Dependency injection", "maintainability", "d", {}),
    ]
    score = compute_structural_utility(
        "maintainability", selected_tactics=["Caching", "Dependency injection"],
        covered_quality_attributes=["performance", "maintainability"], tactics=tactics,
    )
    assert score == 0.8  # covered (1.0) minus one incoming trade-off (0.2)


def test_score_never_goes_below_zero_with_multiple_trade_offs():
    tactics = [
        Tactic("A", "x", "d", {"y": "n"}),
        Tactic("B", "x", "d", {"y": "n"}),
        Tactic("C", "x", "d", {"y": "n"}),
        Tactic("D", "x", "d", {"y": "n"}),
        Tactic("E", "x", "d", {"y": "n"}),
        Tactic("Y1", "y", "d", {}),
    ]
    score = compute_structural_utility(
        "y", selected_tactics=["A", "B", "C", "D", "E", "Y1"],
        covered_quality_attributes=["x", "y"], tactics=tactics,
    )
    assert score == 0.0
