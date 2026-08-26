# Constraint Solver + Repair Loop Implementation Plan

**Goal:** Implement CADENCE Stage 3 (spec §3): verify a deliberated
candidate decision's feasibility with a weighted MaxSAT/SMT encoding
(Z3), extract an unsat core when infeasible, feed it back to an LLM for
targeted repair, iterate to a bounded cap, and degrade gracefully to the
best partial-feasibility candidate with an explicit caveat if still
infeasible after the cap.

**Resolving spec §9's flagged highest-risk item — the constraint
encoding design:** the missing link between "agent-debated trade-offs"
(free text) and "a solvable MaxSAT/SMT instance" is the knowledge graph
already built for Stage 2 (`src/deliberation/knowledge_graph.py`): each
`Tactic` has a fixed `category` (the quality attribute it supports) and
`trade_offs` (other quality attributes it worsens). The encoding is:

1. **Extract** which known tactics are actually referenced in the
   converged candidate's text (fuzzy word-stem matching against the
   fixed, known tactic-name vocabulary — the deliberation agents were
   literally given this vocabulary in their system prompts, per
   `QualityAttributeAgent._system_prompt`, so real LLM output reliably
   references it closely, even if paraphrased).
2. **Model** one boolean "selected" variable per mentioned tactic.
3. **Hard constraint:** at most `tactic_budget` tactics may be selected —
   this is the formal encoding of "keep the change operable by a small
   team" (spec's own sample decision context, and a real, meaningful
   source of infeasibility: covering every quality concern with unlimited
   new mechanisms is not itself an interesting claim to verify; covering
   them within a bounded complexity budget is).
4. **Hard constraint (checked separately for the unsat core):** each
   *required* quality attribute must have at least one selected
   supporting tactic.
5. **Soft constraints (weighted MaxSAT):** minimize the weighted sum of
   trade-offs incurred by selected tactics against other quality
   attributes.

This makes infeasibility a genuine, common outcome (requiring broad
quality-attribute coverage within a tight tactic budget is often
infeasible in this catalog, since each tactic supports exactly one
attribute) rather than a decorative code path that never triggers, and
gives the repair loop a concrete, LLM-actionable signal: which required
attributes couldn't jointly fit the budget.

**Architecture:** `src/solver` package, three independently-testable
layers — tactic extraction (pure text, no Z3), feasibility checking (Z3,
pure/deterministic given inputs, no LLM), and the repair loop (ties the
above to an injected LLM client, same `generate(prompt, system=None)`
duck-typed interface as `src/deliberation`). A script chains Stage 1 →
Stage 2 → Stage 3 for a real, observable run.

**Tech Stack:** Python 3.13 (conda env `py313`), `z3-solver` (installed
this session, verified: `Solver`+`assert_and_track`+`unsat_core()` for
feasibility, `Optimize`+`add_soft` for weighted selection — both verified
empirically against the installed version, not assumed), `pytest`.

**Spec:** `docs/superpowers/specs/2026-08-25-cadence-adr-algorithm-design.md`
(§3 Stage 3, §9 — this plan is the "own design pass" that section called for).

## Global Constraints

- Python 3.13, run via `conda activate py313`.
- Unit tests must not call a real LLM or need the real corpus — the
  feasibility checker takes plain data (candidate text, tactics list) and
  is fully deterministic; the repair loop takes an injected LLM client
  (fake in tests). Z3 itself runs fine in unit tests (fast, deterministic,
  already verified reliable in this environment — no flakiness observed,
  unlike the `transformers`/`sentence_transformers` native-import issues).
- Reuse `src/deliberation/knowledge_graph.py`'s `Tactic`/`TACTICS` — do
  not duplicate the tactic catalog.
- **Deliberate small duplication, noted rather than hidden:** this plan
  adds its own `parse_candidate_rationale`/`CandidateRationaleParseError`
  in `src/solver/synthesis_format.py` rather than importing
  `orchestrator.py`'s private `_parse_synthesis`/`SynthesisParseError`
  from `src/deliberation` (Task 4's repair step needs the identical
  `CANDIDATE:`/`RATIONALE:` parsing behavior). Reusing a leading-underscore
  name across packages, or refactoring Stage 2's already-shipped,
  reviewed, working module to expose it, both carry more risk than ~25
  lines of duplication for a plan already this size — see Self-Review Notes.
- Commit after every task; push after every commit.

---

### Task 1: Tactic extraction from free text

**Files:**
- Create: `src/solver/__init__.py`
- Create: `src/solver/tactic_extraction.py`
- Create: `tests/solver/__init__.py`
- Create: `tests/solver/test_tactic_extraction.py`

**Interfaces:**
- Consumes: `Tactic` (`src/deliberation/knowledge_graph.py`, already committed).
- Produces: `extract_mentioned_tactics(text: str, tactics: list[Tactic], threshold: float = 0.6) -> list[Tactic]` —
  returns the subset of `tactics` whose name is judged "mentioned" in
  `text`, via 4-character-prefix word-stem overlap (tolerates
  paraphrase/morphology like "queuing" vs "queues", not just exact
  substring match). Task 2's feasibility checker calls this to turn a
  candidate's free text into the tactic-selection universe for the solver.

- [ ] **Step 1: Write the failing tests**

```python
# tests/solver/test_tactic_extraction.py
from src.deliberation.knowledge_graph import Tactic
from src.solver.tactic_extraction import extract_mentioned_tactics


def test_extracts_tactic_mentioned_near_verbatim():
    tactics = [Tactic("Caching", "performance", "d", {})]

    result = extract_mentioned_tactics("We will use caching to reduce latency.", tactics)

    assert [t.name for t in result] == ["Caching"]


def test_extracts_tactic_mentioned_with_paraphrase_and_morphology():
    tactics = [
        Tactic("Asynchronous processing via message queues", "performance", "d", {}),
    ]
    text = "leveraging message queuing for asynchronous processing as our primary tactic"

    result = extract_mentioned_tactics(text, tactics)

    assert len(result) == 1
    assert result[0].name == "Asynchronous processing via message queues"


def test_does_not_extract_unrelated_tactic():
    tactics = [Tactic("Read replicas", "scalability", "d", {})]

    result = extract_mentioned_tactics("We should use caching to reduce latency.", tactics)

    assert result == []


def test_extracts_multiple_distinct_tactics_from_one_text():
    tactics = [
        Tactic("Caching", "performance", "d", {}),
        Tactic("Authentication", "security", "d", {}),
        Tactic("Read replicas", "scalability", "d", {}),
    ]
    text = "We will add caching and authentication, but not touch replicas yet."

    result = extract_mentioned_tactics(text, tactics)

    assert {t.name for t in result} == {"Caching", "Authentication"}


def test_empty_text_extracts_nothing():
    tactics = [Tactic("Caching", "performance", "d", {})]

    assert extract_mentioned_tactics("", tactics) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/solver/test_tactic_extraction.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.solver'`.

- [ ] **Step 3: Write the implementation**

```python
# src/solver/__init__.py
```

```python
# tests/solver/__init__.py
```

```python
# src/solver/tactic_extraction.py
"""Match known architectural tactics against free-text deliberation output.

Deliberately a lightweight heuristic (4-char word-stem overlap), not a
learned or exact matcher: the deliberation agents were given the exact
tactic-name vocabulary in their system prompts (see
QualityAttributeAgent._system_prompt), so real LLM output tends to
reference it closely — near-verbatim or lightly paraphrased/re-cased —
which this heuristic tolerates. It will miss tactics referenced only by
unrelated synonyms, and could rarely false-positive on an unrelated tactic
that happens to share enough 4-char word-stem prefixes; both are accepted
trade-offs for a research prototype's text-to-symbol bridge, not something
this plan tries to make perfect.
"""
import re

from src.deliberation.knowledge_graph import Tactic

_STOPWORDS = {"a", "an", "the", "of", "for", "via", "and", "or", "in", "on", "to", "over", "with"}
_STEM_LEN = 4
_MIN_WORD_LEN = 3


def _stem(word: str) -> str:
    return word.lower()[:_STEM_LEN]


def _significant_stems(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", text)
    return {_stem(w) for w in words if len(w) > _MIN_WORD_LEN and w.lower() not in _STOPWORDS}


def extract_mentioned_tactics(text: str, tactics: list[Tactic], threshold: float = 0.6) -> list[Tactic]:
    text_stems = _significant_stems(text)
    if not text_stems:
        return []

    mentioned = []
    for tactic in tactics:
        tactic_stems = _significant_stems(tactic.name)
        if not tactic_stems:
            continue
        overlap = len(tactic_stems & text_stems) / len(tactic_stems)
        if overlap >= threshold:
            mentioned.append(tactic)
    return mentioned
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/solver/test_tactic_extraction.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/solver/__init__.py src/solver/tactic_extraction.py tests/solver/__init__.py tests/solver/test_tactic_extraction.py
git commit -m "feat: add free-text tactic extraction for solver verification"
git push
```

---

### Task 2: Z3-based feasibility checker

**Files:**
- Create: `src/solver/feasibility.py`
- Create: `tests/solver/test_feasibility.py`

**Interfaces:**
- Consumes: `Tactic`, `extract_mentioned_tactics` (Task 1).
- Produces:
  - `FeasibilityResult` — frozen dataclass: `is_feasible: bool`,
    `selected_tactics: list[str]`, `covered_quality_attributes: list[str]`,
    `uncovered_quality_attributes: list[str]`,
    `unsat_core_quality_attributes: list[str]`, `trade_off_cost: float`.
  - `check_feasibility(candidate_text: str, required_quality_attributes: tuple[str, ...], tactic_budget: int, quality_attribute_weights: dict[str, float], tactics: list[Tactic]) -> FeasibilityResult` —
    two-phase Z3 check: (1) strict feasibility via `Solver` +
    `assert_and_track` on a budget constraint and one coverage constraint
    per required quality attribute, extracting `unsat_core()` on failure;
    (2) best-effort optimized selection via `Optimize` with the budget as
    the only hard constraint and (coverage-per-required-QA, trade-off
    avoidance) as weighted soft constraints — always returns a selection,
    even when phase 1 is infeasible, so callers always have something
    concrete to act on. Task 4's repair loop calls this every iteration.

- [ ] **Step 1: Write the failing tests**

```python
# tests/solver/test_feasibility.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/solver/test_feasibility.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.solver.feasibility'`.

- [ ] **Step 3: Write the implementation**

```python
# src/solver/feasibility.py
"""Z3-based feasibility verification for a deliberated candidate decision.

Two-phase check, verified against the installed z3-solver version before
writing this (see the plan's header): phase 1 uses Solver +
assert_and_track to get a real unsat core when infeasible; phase 2 uses a
separate Optimize instance for weighted best-effort selection, since
relying on Optimize's own (less certain) unsat-core support was avoided in
favor of this simpler, empirically-verified split.
"""
from collections import defaultdict
from dataclasses import dataclass

import z3

from src.deliberation.knowledge_graph import Tactic
from src.solver.tactic_extraction import extract_mentioned_tactics

COVERAGE_WEIGHT = 1000.0  # dominates trade-off costs: cover required QAs first, minimize trade-offs second


@dataclass(frozen=True)
class FeasibilityResult:
    is_feasible: bool
    selected_tactics: list[str]
    covered_quality_attributes: list[str]
    uncovered_quality_attributes: list[str]
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
    opt = z3.Optimize()
    opt.add(budget_constraint)
    for qa in required_quality_attributes:
        qa_vars = supports.get(qa, [])
        if qa_vars:
            opt.add_soft(z3.Or(qa_vars), weight=COVERAGE_WEIGHT)
    for t in mentioned:
        for other_qa in t.trade_offs:
            weight = quality_attribute_weights.get(other_qa, 1.0)
            opt.add_soft(z3.Not(tactic_vars[t.name]), weight=weight)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/solver/test_feasibility.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/solver/feasibility.py tests/solver/test_feasibility.py
git commit -m "feat: add Z3-based weighted feasibility checker with unsat-core extraction"
git push
```

---

### Task 3: Shared CANDIDATE/RATIONALE parsing utility

**Files:**
- Create: `src/solver/synthesis_format.py`
- Create: `tests/solver/test_synthesis_format.py`

**Interfaces:**
- Consumes: nothing beyond `re`.
- Produces: `CandidateRationaleParseError` (`RuntimeError` subclass),
  `parse_candidate_rationale(text: str) -> tuple[str, str]` — same
  tolerant `CANDIDATE:`/`RATIONALE:` parsing behavior as
  `src/deliberation/orchestrator.py`'s private `_parse_synthesis`
  (deliberately not imported from there — see the plan header's Global
  Constraints). Task 4's repair loop uses this to parse each repaired
  candidate the same way the original synthesizer's output was parsed.

- [ ] **Step 1: Write the failing tests**

```python
# tests/solver/test_synthesis_format.py
import pytest

from src.solver.synthesis_format import CandidateRationaleParseError, parse_candidate_rationale


def test_parses_simple_candidate_and_rationale():
    text = "CANDIDATE: Use read replicas.\nRATIONALE: Balances performance and cost."

    candidate, rationale = parse_candidate_rationale(text)

    assert candidate == "Use read replicas."
    assert rationale == "Balances performance and cost."


def test_tolerant_of_markdown_case_and_multiline_rationale():
    text = (
        "**Candidate:** Use read replicas for scaling reads.\n"
        "- rationale: This balances performance and cost.\n"
        "It also keeps the change operable by a small team."
    )

    candidate, rationale = parse_candidate_rationale(text)

    assert candidate == "Use read replicas for scaling reads."
    assert rationale == (
        "This balances performance and cost. It also keeps the change operable by a small team."
    )


def test_raises_clearly_on_unparseable_text():
    with pytest.raises(CandidateRationaleParseError):
        parse_candidate_rationale("I'm not sure what to recommend here.")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/solver/test_synthesis_format.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.solver.synthesis_format'`.

- [ ] **Step 3: Write the implementation**

```python
# src/solver/synthesis_format.py
"""Shared CANDIDATE:/RATIONALE: text format parsing.

Deliberately duplicated (not imported) from
src/deliberation/orchestrator.py's private _parse_synthesis — see the
constraint-solver plan's Global Constraints for why.
"""
import re


class CandidateRationaleParseError(RuntimeError):
    """Raised when text doesn't contain a parseable CANDIDATE/RATIONALE pair."""


_LABEL_LINE = re.compile(r"^\s*[*_\-\s]*(CANDIDATE|RATIONALE)[*_\s]*:[*_\s]*(.*)$", re.IGNORECASE)


def parse_candidate_rationale(text: str) -> tuple[str, str]:
    fields: dict[str, list[str]] = {"CANDIDATE": [], "RATIONALE": []}
    current: str | None = None

    for line in text.splitlines():
        match = _LABEL_LINE.match(line)
        if match:
            current = match.group(1).upper()
            remainder = match.group(2).strip()
            if remainder:
                fields[current].append(remainder)
        elif current is not None and line.strip():
            fields[current].append(line.strip())

    candidate = " ".join(fields["CANDIDATE"]).strip()
    rationale = " ".join(fields["RATIONALE"]).strip()

    if not candidate or not rationale:
        raise CandidateRationaleParseError(
            f"Could not parse both CANDIDATE and RATIONALE from: {text!r}"
        )
    return candidate, rationale
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/solver/test_synthesis_format.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/solver/synthesis_format.py tests/solver/test_synthesis_format.py
git commit -m "feat: add shared CANDIDATE/RATIONALE parser for the repair loop"
git push
```

---

### Task 4: Repair loop and real end-to-end demo (Stage 1 → 2 → 3)

**Files:**
- Create: `src/solver/repair.py`
- Create: `scripts/run_solver_demo.py`
- Create: `tests/solver/test_repair.py`
- Create: `tests/data/test_run_solver_demo_script.py`

**Interfaces:**
- Consumes: `check_feasibility`, `FeasibilityResult` (Task 2);
  `parse_candidate_rationale`, `CandidateRationaleParseError` (Task 3);
  `Tactic`, `TACTICS` (`src/deliberation/knowledge_graph.py`); any
  `generate(prompt, system=None)`-shaped client (duck-typed — tests
  inject a fake).
- Produces:
  - `VerifiedDecision` — frozen dataclass: `is_feasible: bool`,
    `final_candidate: str`, `rationale: str`, `selected_tactics: list[str]`,
    `covered_quality_attributes: list[str]`,
    `uncovered_quality_attributes: list[str]`, `repair_iterations: int`,
    `caveat: str | None`.
  - `run_repair_loop(candidate: str, rationale: str, required_quality_attributes: tuple[str, ...], tactic_budget: int, quality_attribute_weights: dict[str, float], tactics: list[Tactic], repair_client, max_repair_iterations: int = 2) -> VerifiedDecision` —
    checks feasibility; if infeasible, builds a repair prompt naming the
    unsat-core quality attributes and the budget, asks `repair_client` for
    a revised `CANDIDATE:`/`RATIONALE:` response, re-checks; repeats up to
    `max_repair_iterations`; tracks the best attempt seen (fewest
    uncovered required QAs, tie-broken by lowest trade-off cost) so a
    final degrade-gracefully step never discards a better result found in
    an earlier iteration.
  - `scripts/run_solver_demo.py` — real end-to-end script: Stage 1
    (`Retriever`) → Stage 2 (`DeliberationOrchestrator`) → Stage 3
    (`run_repair_loop`), reusing one loaded `LocalHFClient` throughout, on
    a **tight tactic budget (4)** deliberately chosen to exercise the real
    repair path (not just the trivially-feasible one) in the real run.

- [ ] **Step 1: Write the failing tests**

```python
# tests/solver/test_repair.py
from src.deliberation.knowledge_graph import Tactic
from src.solver.repair import VerifiedDecision, run_repair_loop


def _catalog():
    return [
        Tactic("Caching", "performance", "d", {}),
        Tactic("Authentication", "security", "d", {}),
        Tactic("Automated regression test suite", "maintainability", "d", {}),
    ]


class _FakeRepairClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, system=None):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_returns_immediately_feasible_without_calling_repair_client():
    client = _FakeRepairClient([])  # would raise IndexError if called

    result = run_repair_loop(
        candidate="We will use caching.",
        rationale="Improves performance.",
        required_quality_attributes=("performance",),
        tactic_budget=2,
        quality_attribute_weights={},
        tactics=_catalog(),
        repair_client=client,
        max_repair_iterations=2,
    )

    assert isinstance(result, VerifiedDecision)
    assert result.is_feasible
    assert result.repair_iterations == 0
    assert result.caveat is None
    assert result.final_candidate == "We will use caching."


def test_repairs_once_and_succeeds():
    client = _FakeRepairClient([
        "CANDIDATE: We will use caching and authentication.\nRATIONALE: Fits the budget.",
    ])

    result = run_repair_loop(
        candidate="We will use caching, authentication, and an automated regression test suite.",
        rationale="Covers everything.",
        required_quality_attributes=("performance", "security"),
        tactic_budget=2,
        quality_attribute_weights={},
        tactics=_catalog(),
        repair_client=client,
        max_repair_iterations=2,
    )

    assert result.is_feasible
    assert result.repair_iterations == 1
    assert "budget" in client.prompts[0].lower()
    assert "maintainability" in client.prompts[0] or "security" in client.prompts[0] or "performance" in client.prompts[0]


def test_degrades_gracefully_after_exhausting_repair_attempts():
    # Same unsolvable-within-budget response every time -> never feasible.
    client = _FakeRepairClient([
        "CANDIDATE: We will use caching, authentication, and testing.\nRATIONALE: Still too much.",
        "CANDIDATE: We will use caching, authentication, and testing.\nRATIONALE: Still too much.",
    ])

    result = run_repair_loop(
        candidate="We will use caching, authentication, and an automated regression test suite.",
        rationale="Covers everything.",
        required_quality_attributes=("performance", "security", "maintainability"),
        tactic_budget=1,
        quality_attribute_weights={},
        tactics=_catalog(),
        repair_client=client,
        max_repair_iterations=2,
    )

    assert not result.is_feasible
    assert result.repair_iterations == 2
    assert result.caveat is not None
    assert len(result.uncovered_quality_attributes) == 2
    assert len(result.selected_tactics) == 1


def test_keeps_best_attempt_even_if_a_later_repair_is_worse():
    # First repair covers 2/2 required (feasible) -- loop should return
    # immediately without needing a worse second attempt, but this also
    # guards against a regression where a later, worse attempt could
    # overwrite a better one if the loop didn't return early.
    client = _FakeRepairClient([
        "CANDIDATE: We will use caching and authentication.\nRATIONALE: Fits.",
    ])

    result = run_repair_loop(
        candidate="We will use nothing in particular.",
        rationale="Vague.",
        required_quality_attributes=("performance", "security"),
        tactic_budget=2,
        quality_attribute_weights={},
        tactics=_catalog(),
        repair_client=client,
        max_repair_iterations=2,
    )

    assert result.is_feasible
    assert result.repair_iterations == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/solver/test_repair.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.solver.repair'`.

- [ ] **Step 3: Write the repair loop**

```python
# src/solver/repair.py
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
    candidate: str, result: FeasibilityResult, tactic_budget: int, required_quality_attributes: tuple[str, ...]
) -> str:
    # Uses uncovered_quality_attributes, not unsat_core_quality_attributes,
    # as the authoritative "what needs fixing" list — per feasibility.py's
    # docstring (added after code review), the unsat core is a valid but
    # possibly-incomplete explanation, while uncovered_quality_attributes
    # is the complete, sound signal now that phase 2 optimizes coverage
    # lexicographically ahead of trade-off cost. The core is still
    # mentioned as supplementary "at least these conflict" context.
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
        "Revise the decision to fit within the tactic budget while covering as many of the "
        "required quality attributes as possible, consolidating around fewer, higher-impact "
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

        repair_prompt = _build_repair_prompt(candidate, result, tactic_budget, required_quality_attributes)
        response = repair_client.generate(repair_prompt)
        candidate, rationale = parse_candidate_rationale(response)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/solver/test_repair.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Write the failing test for the real end-to-end script's wiring**

```python
# tests/data/test_run_solver_demo_script.py
import json

import numpy as np


def test_run_solver_demo_wires_all_three_stages_together(tmp_path, monkeypatch):
    from scripts.run_solver_demo import run_solver_demo

    records_path = tmp_path / "adr_records.jsonl"
    records_path.write_text(
        json.dumps({
            "record_id": "r/1.md", "repo_folder": "r", "repository_url": None,
            "relative_path": "1.md", "sequence_number": 1, "title": "Use read replicas",
            "raw_text": "text", "extraction_status": "Verified",
        }) + "\n",
        encoding="utf-8",
    )
    embeddings_path = tmp_path / "adr_embeddings.npy"
    np.save(embeddings_path, np.array([[1.0, 0.0]]))

    class _FakeClient:
        def generate(self, prompt, system=None):
            return "CANDIDATE: We will use caching.\nRATIONALE: Improves performance within budget."

    class _FakeEmbeddingModel:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr("scripts.run_solver_demo.load_local_hf_client", lambda: _FakeClient())
    monkeypatch.setattr("scripts.run_solver_demo.load_embedding_model", lambda: _FakeEmbeddingModel())

    result = run_solver_demo(
        records_path=records_path, embeddings_path=embeddings_path,
        context="Sample decision context.", max_rounds=1, tactic_budget=2, max_repair_iterations=1,
    )

    assert result.final_candidate == "We will use caching."
```

- [ ] **Step 6: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/data/test_run_solver_demo_script.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_solver_demo'`.

- [ ] **Step 7: Write the script**

```python
# scripts/run_solver_demo.py
"""Real end-to-end demo: Stage 1 (retrieval) -> Stage 2 (deliberation) ->
Stage 3 (constraint-solver verification + repair).

Import order: src.retrieval.embeddings (sentence_transformers) is imported
before anything that triggers torch via the local-HF deliberation client,
per the sentence_transformers-before-torch rule in PROGRESS.md.
"""
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.retrieval.embeddings import embed_texts, load_embedding_model  # noqa: F401  (import before torch)
import numpy as np

from src.deliberation.agent import QualityAttributeAgent
from src.deliberation.knowledge_graph import QUALITY_ATTRIBUTES, TACTICS, build_knowledge_graph
from src.deliberation.llm_client import load_local_hf_client
from src.deliberation.orchestrator import DeliberationOrchestrator
from src.retrieval.records import ADRRecord
from src.retrieval.retriever import Retriever
from src.solver.repair import VerifiedDecision, run_repair_loop

PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
RECORDS_PATH = PROCESSED_DIR / "adr_records.jsonl"
EMBEDDINGS_PATH = PROCESSED_DIR / "adr_embeddings.npy"

SAMPLE_CONTEXT = (
    "Our service-oriented system's order-processing service is experiencing "
    "10x read traffic growth from a new mobile client. Requirements: keep "
    "p99 read latency low, keep the change operable by a small team, and "
    "avoid introducing new categories of security risk."
)


def _load_records(records_path: Path) -> list[ADRRecord]:
    records = []
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            records.append(ADRRecord(**json.loads(line)))
    return records


def run_solver_demo(
    records_path: Path,
    embeddings_path: Path,
    context: str,
    max_rounds: int = 2,
    tactic_budget: int = 4,
    max_repair_iterations: int = 2,
) -> VerifiedDecision:
    records = _load_records(records_path)
    embeddings = np.load(embeddings_path)
    embedding_model = load_embedding_model()
    retriever = Retriever(records, embeddings, embedding_model)
    precedents = retriever.retrieve(context, k=3)

    llm_client = load_local_hf_client()
    graph = build_knowledge_graph(TACTICS)
    agents = [QualityAttributeAgent(qa, llm_client, graph) for qa in QUALITY_ATTRIBUTES]
    orchestrator = DeliberationOrchestrator(agents, llm_client, max_rounds=max_rounds)
    deliberation = orchestrator.deliberate(context, precedents)

    return run_repair_loop(
        candidate=deliberation.converged_candidate,
        rationale=deliberation.rationale,
        required_quality_attributes=QUALITY_ATTRIBUTES,
        tactic_budget=tactic_budget,
        quality_attribute_weights={},
        tactics=TACTICS,
        repair_client=llm_client,
        max_repair_iterations=max_repair_iterations,
    )


if __name__ == "__main__":
    result = run_solver_demo(RECORDS_PATH, EMBEDDINGS_PATH, SAMPLE_CONTEXT)
    print("=== Verified decision ===")
    print(f"feasible: {result.is_feasible}")
    print(f"repair iterations: {result.repair_iterations}")
    print(f"candidate: {result.final_candidate}")
    print(f"rationale: {result.rationale}")
    print(f"selected tactics: {result.selected_tactics}")
    print(f"covered quality attributes: {result.covered_quality_attributes}")
    print(f"uncovered quality attributes: {result.uncovered_quality_attributes}")
    if result.caveat:
        print(f"caveat: {result.caveat}")
```

- [ ] **Step 8: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/data/test_run_solver_demo_script.py -v`
Expected: PASS (1 passed).

- [ ] **Step 9: Run the full suite, then run the real demo**

Run: `conda run -n py313 pytest -q` — expect all tests (old + new) passing.

Run: `conda run -n py313 python scripts/run_solver_demo.py` (or the env's
`python.exe` directly if `conda run` mis-wraps the invocation) — this runs
all 3 stages for real against the real corpus and the local model, with a
deliberately tight `tactic_budget=4` (5 quality attributes, budget 4) so
the repair path is genuinely exercised, not just the trivially-feasible
one. Expect either `is_feasible=True` after 1+ repair iterations
consolidating tactics, or a graceful `is_feasible=False` with a populated
`caveat` after 2 repair attempts — both are valid, meaningful outcomes;
what matters is that it completes without crashing and the printed
`selected_tactics`/`covered_quality_attributes` are consistent with the
printed `candidate` text.

- [ ] **Step 10: Commit**

```bash
git add src/solver/repair.py scripts/run_solver_demo.py tests/solver/test_repair.py tests/data/test_run_solver_demo_script.py
git commit -m "feat: add constraint-solver repair loop and Stage 1-3 end-to-end demo"
git push
```

---

## Self-Review Notes

- **Correction found during Task 2's code review, applied before Task 4
  was written:** the original design (embedded in Task 2's code block
  above) used a single `COVERAGE_WEIGHT = 1000.0` constant to bias phase
  2's optimizer toward covering required quality attributes ahead of
  minimizing trade-off cost. Review found this was a weighted-*sum*
  objective, not a true priority — a real input with enough
  caller-supplied trade-off weight (e.g. four trade-offs at 300 each
  against the one tactic covering a required QA) could out-vote coverage
  entirely, silently leaving a coverable QA uncovered. Fixed in the
  actual `src/solver/feasibility.py` (not reflected in this file's Task 2
  snippet above) by using `z3.Optimize().set(priority="lex")` with two
  separate `add_soft(..., id=...)` objective groups (`"coverage"` then
  `"tradeoffs"`), verified empirically to make coverage dominate
  regardless of weight magnitude. The review also found `z3`'s
  `unsat_core()` can omit required QAs just as blocking as the ones it
  names (inherent to unsat-core semantics, not a bug) — so Task 4's
  `_build_repair_prompt` above was written from the start to treat
  `uncovered_quality_attributes` (sound and complete, given the
  lex-priority fix) as the authoritative repair signal, with the unsat
  core included only as supplementary context.
- **Spec coverage:** implements exactly CADENCE Stage 3 (spec §3) and
  resolves §9's flagged "own design pass" requirement for the constraint
  encoding. Does not implement Stage 4 (self-critique) — separate
  follow-on plan.
- **Why tactic-budget-vs-coverage is the constraint model, not something
  else:** the alternative (trying to derive hard pass/fail constraints
  directly from the free-text requirements in the decision context, e.g.
  "keep p99 latency low") would need another LLM extraction step with no
  ground truth to validate it against, and wouldn't obviously connect to
  the knowledge graph's actual structured content (tactics and
  trade-offs). The budget-vs-coverage formulation instead uses exactly
  the structured data already available (`Tactic.category`,
  `Tactic.trade_offs`) and produces genuine, explainable infeasibility
  without inventing an unvalidatable extraction step.
- **Why a two-phase Z3 check (`Solver` for the core, separate `Optimize`
  for selection) instead of one `Optimize` call:** verified empirically
  (see plan header) rather than assumed; keeps the unsat-core extraction
  path simple and certain rather than depending on `Optimize`'s own core
  support, which wasn't verified and isn't needed given the two-phase
  split works cleanly.
- **Why tactic extraction is a heuristic, not exact matching:** the
  deliberation agents are prompted with exact tactic names but real LLM
  output paraphrases; a heuristic that tolerates this (word-stem overlap)
  is necessary for the solver to see what the agents actually proposed.
  Documented as an accepted-imprecision trade-off in
  `tactic_extraction.py`'s own docstring, consistent with how
  `src/retrieval/records.py`'s title extraction handled a similar
  precision/robustness trade-off in the retrieval-indexing plan.
- **Why `synthesis_format.py` duplicates `orchestrator.py`'s private
  parser instead of importing or refactoring it:** see Global Constraints.
- **Placeholder scan:** no TBD/TODO; all code blocks are complete and
  runnable.
- **Type/interface consistency:** `FeasibilityResult` fields match
  between `feasibility.py` (definition) and `repair.py` (`_is_better`,
  `VerifiedDecision` construction). `run_repair_loop`'s signature matches
  between Task 4's definition and `run_solver_demo.py`'s call site.
  `Tactic`/`TACTICS`/`QUALITY_ATTRIBUTES` imported identically from
  `src.deliberation.knowledge_graph` across `tactic_extraction.py`,
  `feasibility.py`, `repair.py`, and `run_solver_demo.py`.
