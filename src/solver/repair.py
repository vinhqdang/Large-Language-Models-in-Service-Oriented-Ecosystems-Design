"""Bounded constraint-solver repair loop (CADENCE Stage 3)."""
from dataclasses import dataclass

from src.deliberation.knowledge_graph import Tactic
from src.solver.feasibility import FeasibilityResult, check_feasibility
from src.solver.synthesis_format import parse_candidate_rationale


@dataclass(frozen=True)
class VerifiedDecision:
    is_feasible: bool
    final_candidate: str
    rationale: str
    selected_tactics: list[str]
    covered_quality_attributes: list[str]
    uncovered_quality_attributes: list[str]
    repair_iterations: int
    caveat: str | None


def _build_repair_prompt(
    candidate: str,
    result: FeasibilityResult,
    tactic_budget: int,
    required_quality_attributes: tuple[str, ...],
    tactics: list[Tactic],
) -> str:
    # uncovered_quality_attributes (not unsat_core_quality_attributes) is
    # the authoritative "what needs fixing" list — see feasibility.py's
    # FeasibilityResult docstring for why the unsat core can be an
    # incomplete explanation. The core is included only as supplementary
    # "at least these conflict" context.
    #
    # Naming concrete tactic options per uncovered attribute matters: the
    # tactic-extraction heuristic (src/solver/tactic_extraction.py) can
    # only detect tactics the response actually mentions by name (or close
    # paraphrase). Without this, an LLM asked to "cover security" has no
    # reason to use recognizable phrasing, and a repair attempt can fail
    # to move the needle even when it's a reasonable revision in plain
    # English — this list is what lets a small model reliably produce
    # extractable output, mirroring how Stage 2's agents are always given
    # their own quality attribute's tactic vocabulary.
    options_lines = "\n".join(
        f"- {qa}: " + ", ".join(t.name for t in tactics if t.category == qa)
        for qa in result.uncovered_quality_attributes
    )
    return (
        f"The following architectural decision candidate cannot be verified as feasible:\n"
        f"{candidate}\n\n"
        f"Constraint: the decision may commit to at most {tactic_budget} distinct architectural "
        "tactics (an operability budget).\n"
        f"Required quality attributes that must each be addressed by at least one tactic: "
        f"{', '.join(required_quality_attributes)}.\n"
        f"The best attempt so far leaves these quality attributes unaddressed: "
        f"{', '.join(result.uncovered_quality_attributes)} "
        f"(at least {', '.join(result.unsat_core_quality_attributes)} are known to conflict "
        "with the budget).\n\n"
        "Tactics you could name explicitly to address each unaddressed attribute:\n"
        f"{options_lines}\n\n"
        "Revise the decision to fit within the tactic budget while covering as many of the "
        "required quality attributes as possible — naming specific tactics from the lists "
        "above by name where you use them — consolidating around fewer, higher-impact "
        "tactics where needed. Respond in exactly this format:\n"
        "CANDIDATE: <one or two sentence decision>\n"
        "RATIONALE: <one paragraph rationale>"
    )


def _is_better(candidate: FeasibilityResult, current_best: FeasibilityResult) -> bool:
    if len(candidate.uncovered_quality_attributes) != len(current_best.uncovered_quality_attributes):
        return len(candidate.uncovered_quality_attributes) < len(current_best.uncovered_quality_attributes)
    return candidate.trade_off_cost < current_best.trade_off_cost


def run_repair_loop(
    candidate: str,
    rationale: str,
    required_quality_attributes: tuple[str, ...],
    tactic_budget: int,
    quality_attribute_weights: dict[str, float],
    tactics: list[Tactic],
    repair_client,
    max_repair_iterations: int = 2,
) -> VerifiedDecision:
    best_result: FeasibilityResult | None = None
    best_candidate, best_rationale = candidate, rationale

    for iteration in range(max_repair_iterations + 1):
        result = check_feasibility(
            candidate, required_quality_attributes, tactic_budget, quality_attribute_weights, tactics
        )

        if best_result is None or _is_better(result, best_result):
            best_result, best_candidate, best_rationale = result, candidate, rationale

        if result.is_feasible:
            return VerifiedDecision(
                is_feasible=True,
                final_candidate=candidate,
                rationale=rationale,
                selected_tactics=result.selected_tactics,
                covered_quality_attributes=result.covered_quality_attributes,
                uncovered_quality_attributes=[],
                repair_iterations=iteration,
                caveat=None,
            )

        if iteration == max_repair_iterations:
            break

        repair_prompt = _build_repair_prompt(candidate, result, tactic_budget, required_quality_attributes, tactics)
        try:
            response = repair_client.generate(repair_prompt)
            candidate, rationale = parse_candidate_rationale(response)
        except Exception:
            # A transient LLM error or an unparseable repair response must
            # not crash the whole run and discard best_result tracked so
            # far — treat it as a failed repair attempt: keep the current
            # candidate/rationale unchanged and use the next iteration (if
            # any remain) to try again.
            continue

    caveat = (
        f"Could not find a decision covering all required quality attributes within a "
        f"budget of {tactic_budget} tactics after {max_repair_iterations} repair attempt(s). "
        f"Best effort covers {best_result.covered_quality_attributes} but leaves "
        f"{best_result.uncovered_quality_attributes} unaddressed."
    )
    return VerifiedDecision(
        is_feasible=False,
        final_candidate=best_candidate,
        rationale=best_rationale,
        selected_tactics=best_result.selected_tactics,
        covered_quality_attributes=best_result.covered_quality_attributes,
        uncovered_quality_attributes=best_result.uncovered_quality_attributes,
        repair_iterations=max_repair_iterations,
        caveat=caveat,
    )
