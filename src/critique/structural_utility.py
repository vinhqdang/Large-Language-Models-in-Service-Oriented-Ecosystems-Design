"""Deterministic structural component of each quality attribute's utility
function (CADENCE Stage 4) — reuses facts Stage 3 already established
(selected tactics, coverage) rather than recomputing anything.
"""
from src.deliberation.knowledge_graph import Tactic

TRADE_OFF_PENALTY = 0.2


def compute_structural_utility(
    quality_attribute: str,
    selected_tactics: list[str],
    covered_quality_attributes: list[str],
    tactics: list[Tactic],
) -> float:
    coverage_score = 1.0 if quality_attribute in covered_quality_attributes else 0.0

    by_name = {t.name: t for t in tactics}
    penalty = sum(
        TRADE_OFF_PENALTY
        for name in selected_tactics
        if name in by_name and quality_attribute in by_name[name].trade_offs
    )

    return max(0.0, coverage_score - penalty)
