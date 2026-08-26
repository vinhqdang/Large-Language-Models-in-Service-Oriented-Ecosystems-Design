from src.deliberation.knowledge_graph import Tactic
from src.solver.feasibility import check_feasibility


def _catalog():
    # 2 tactics per QA, each supporting exactly one QA, no trade-offs by
    # default (kept simple; trade-off-specific tests add their own).
    return [
        Tactic("Caching", "performance", "d", {}),
        Tactic("Connection pooling", "performance", "d", {}),
        Tactic("Authentication", "security", "d", {}),
        Tactic("Encryption in transit", "security", "d", {}),
        Tactic("Automated regression test suite", "maintainability", "d", {}),
        Tactic("Dependency injection", "maintainability", "d", {}),
    ]


def _text_mentioning(*names):
    return "We will use " + " and ".join(names) + "."


def test_feasible_when_budget_covers_all_required_attributes():
    text = _text_mentioning("Caching", "Authentication")

    result = check_feasibility(
        candidate_text=text,
        required_quality_attributes=("performance", "security"),
        tactic_budget=2,
        quality_attribute_weights={},
        tactics=_catalog(),
    )

    assert result.is_feasible
    assert set(result.covered_quality_attributes) == {"performance", "security"}
    assert result.uncovered_quality_attributes == []
    assert result.unsat_core_quality_attributes == []


def test_infeasible_when_budget_too_tight_for_required_coverage():
    text = _text_mentioning("Caching", "Authentication", "Automated regression test suite")

    result = check_feasibility(
        candidate_text=text,
        required_quality_attributes=("performance", "security", "maintainability"),
        tactic_budget=2,
        quality_attribute_weights={},
        tactics=_catalog(),
    )

    assert not result.is_feasible
    assert set(result.unsat_core_quality_attributes) == {"performance", "security", "maintainability"}
    # best-effort selection still returned, within budget, covering 2 of 3
    assert len(result.selected_tactics) <= 2
    assert len(result.covered_quality_attributes) == 2
    assert len(result.uncovered_quality_attributes) == 1


def test_required_attribute_with_no_mentioned_supporting_tactic_is_uncovered():
    text = _text_mentioning("Caching")  # no security tactic mentioned at all

    result = check_feasibility(
        candidate_text=text,
        required_quality_attributes=("performance", "security"),
        tactic_budget=5,
        quality_attribute_weights={},
        tactics=_catalog(),
    )

    assert not result.is_feasible
    assert "security" in result.unsat_core_quality_attributes
    assert "security" in result.uncovered_quality_attributes
    assert "performance" in result.covered_quality_attributes


def test_optimizer_avoids_higher_weighted_trade_off_when_alternative_exists():
    tactics = [
        Tactic("Caching", "performance", "d", {"maintainability": "invalidation complexity"}),
        Tactic("Connection pooling", "performance", "d", {}),  # no trade-off: cheaper choice
    ]
    text = _text_mentioning("Caching", "Connection pooling")

    result = check_feasibility(
        candidate_text=text,
        required_quality_attributes=("performance",),
        tactic_budget=1,
        quality_attribute_weights={"maintainability": 5.0},
        tactics=tactics,
    )

    assert result.is_feasible
    assert result.selected_tactics == ["Connection pooling"]
    assert result.trade_off_cost == 0.0


def test_coverage_dominates_regardless_of_summed_trade_off_weight():
    """Regression: coverage must strictly dominate trade-off avoidance
    regardless of the magnitude of caller-supplied weights — a weighted-sum
    objective (rather than lexicographic) could let enough accumulated
    trade-off weight outvote covering a required attribute at all."""
    tactic = Tactic(
        "OnlySupport", "x", "d",
        {"p1": "n", "p2": "n", "p3": "n", "p4": "n"},
    )
    text = _text_mentioning("OnlySupport")

    result = check_feasibility(
        candidate_text=text,
        required_quality_attributes=("x",),
        tactic_budget=5,
        quality_attribute_weights={"p1": 300.0, "p2": 300.0, "p3": 300.0, "p4": 300.0},
        tactics=[tactic],
    )

    assert result.is_feasible
    assert result.selected_tactics == ["OnlySupport"]
    assert result.covered_quality_attributes == ["x"]


def test_multiple_uncoverable_required_attributes_are_all_reported_uncovered():
    """Regression: uncovered_quality_attributes (not the unsat core, which
    may only name a subset) must be the complete, authoritative signal of
    which required QAs the best-effort selection failed to cover."""
    catalog = _catalog()  # has performance/security/maintainability tactics only

    result = check_feasibility(
        candidate_text="We will use caching.",  # only performance is coverable
        required_quality_attributes=("performance", "security", "maintainability", "scalability"),
        tactic_budget=5,
        quality_attribute_weights={},
        tactics=catalog,
    )

    assert not result.is_feasible
    assert result.covered_quality_attributes == ["performance"]
    assert set(result.uncovered_quality_attributes) == {"security", "maintainability", "scalability"}


def test_budget_deficit_greater_than_one_still_reports_correct_uncovered_count():
    text = _text_mentioning("Caching", "Authentication", "Automated regression test suite")

    result = check_feasibility(
        candidate_text=text,
        required_quality_attributes=("performance", "security", "maintainability"),
        tactic_budget=1,
        quality_attribute_weights={},
        tactics=_catalog(),
    )

    assert not result.is_feasible
    assert len(result.selected_tactics) == 1
    assert len(result.covered_quality_attributes) == 1
    assert len(result.uncovered_quality_attributes) == 2


def test_no_tactics_mentioned_and_no_required_attributes_is_trivially_feasible():
    result = check_feasibility(
        candidate_text="A generic decision mentioning nothing specific.",
        required_quality_attributes=(),
        tactic_budget=3,
        quality_attribute_weights={},
        tactics=_catalog(),
    )

    assert result.is_feasible
    assert result.selected_tactics == []
