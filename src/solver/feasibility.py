"""Z3-based feasibility verification for a deliberated candidate decision.

Two-phase check, verified against the installed z3-solver version before
writing this (see the plan's header): phase 1 uses Solver +
assert_and_track to get a real unsat core when infeasible; phase 2 uses a
separate Optimize instance for weighted best-effort selection, since
relying on Optimize's own (less certain) unsat-core support was avoided in
favor of this simpler, empirically-verified split.

Phase 2 optimizes QA coverage and trade-off avoidance as two SEPARATE
objectives under `priority="lex"` (coverage strictly dominates, verified
empirically), not as one weighted sum — an earlier version used a single
large COVERAGE_WEIGHT constant to *bias* coverage above trade-off cost,
but since quality_attribute_weights are caller-supplied with no bound, a
tactic with enough accumulated trade-off weight could out-vote coverage
under that scheme (confirmed by code review with a concrete repro: 4
trade-offs at weight 300 each outweighed a coverage weight of 1000).
Lexicographic priority makes coverage dominate regardless of scale.
"""
from collections import defaultdict
from dataclasses import dataclass

import z3

from src.deliberation.knowledge_graph import Tactic
from src.solver.tactic_extraction import extract_mentioned_tactics


@dataclass(frozen=True)
class FeasibilityResult:
    is_feasible: bool
    selected_tactics: list[str]
    covered_quality_attributes: list[str]
    uncovered_quality_attributes: list[str]
    # A z3 unsat core is *a* valid explanation of infeasibility, not
    # necessarily *the* complete or minimal one — it can omit required
    # quality attributes that are just as blocking as the ones it names
    # (e.g. among several symmetric zero-tactic-coverage QAs, the core may
    # name only one). Treat this as supplementary diagnostic context, not
    # as the authoritative list of what needs fixing — that's
    # uncovered_quality_attributes, which (given the lexicographic
    # coverage-first objective above) is the complete, sound signal of
    # every required QA the optimizer could not cover within budget.
    unsat_core_quality_attributes: list[str]
    trade_off_cost: float


def check_feasibility(
    candidate_text: str,
    required_quality_attributes: tuple[str, ...],
    tactic_budget: int,
    quality_attribute_weights: dict[str, float],
    tactics: list[Tactic],
) -> FeasibilityResult:
    mentioned = extract_mentioned_tactics(candidate_text, tactics)
    tactic_vars = {t.name: z3.Bool(t.name) for t in mentioned}
    all_vars = list(tactic_vars.values())

    supports: dict[str, list[z3.BoolRef]] = defaultdict(list)
    for t in mentioned:
        supports[t.category].append(tactic_vars[t.name])

    budget_constraint = z3.PbLe([(v, 1) for v in all_vars], tactic_budget) if all_vars else z3.BoolVal(True)

    # --- Phase 1: strict feasibility + unsat core ---
    solver = z3.Solver()
    solver.assert_and_track(budget_constraint, "tactic_budget")
    for qa in required_quality_attributes:
        qa_vars = supports.get(qa, [])
        constraint = z3.Or(qa_vars) if qa_vars else z3.BoolVal(False)
        solver.assert_and_track(constraint, f"cover::{qa}")

    is_feasible = solver.check() == z3.sat
    unsat_core_qas: list[str] = []
    if not is_feasible:
        core_names = {str(c) for c in solver.unsat_core()}
        unsat_core_qas = sorted(name.split("::", 1)[1] for name in core_names if name.startswith("cover::"))

    # --- Phase 2: best-effort optimized selection (always returns something) ---
    # Two separate objectives under lexicographic priority: coverage first
    # (maximize how many required QAs get a selected supporting tactic),
    # trade-off avoidance second (among coverage-optimal selections,
    # minimize weighted trade-off cost). See module docstring for why this
    # replaced a single weighted-sum objective.
    opt = z3.Optimize()
    opt.set(priority="lex")
    opt.add(budget_constraint)
    for qa in required_quality_attributes:
        qa_vars = supports.get(qa, [])
        if qa_vars:
            opt.add_soft(z3.Or(qa_vars), weight=1.0, id="coverage")
    for t in mentioned:
        for other_qa in t.trade_offs:
            weight = quality_attribute_weights.get(other_qa, 1.0)
            opt.add_soft(z3.Not(tactic_vars[t.name]), weight=weight, id="tradeoffs")

    opt.check()
    model = opt.model()
    selected_names = [
        t.name for t in mentioned
        if z3.is_true(model.eval(tactic_vars[t.name], model_completion=True))
    ]
    selected_categories = {t.category for t in mentioned if t.name in selected_names}

    covered = sorted(qa for qa in required_quality_attributes if qa in selected_categories)
    uncovered = sorted(qa for qa in required_quality_attributes if qa not in selected_categories)
    trade_off_cost = sum(
        quality_attribute_weights.get(other_qa, 1.0)
        for t in mentioned if t.name in selected_names
        for other_qa in t.trade_offs
    )

    return FeasibilityResult(
        is_feasible=is_feasible,
        selected_tactics=sorted(selected_names),
        covered_quality_attributes=covered,
        uncovered_quality_attributes=uncovered,
        unsat_core_quality_attributes=unsat_core_qas,
        trade_off_cost=trade_off_cost,
    )
