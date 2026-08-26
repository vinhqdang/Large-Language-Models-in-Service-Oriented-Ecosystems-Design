# Self-Critique / Finalization Implementation Plan

**Goal:** Implement CADENCE Stage 4 (spec §3): score the converged,
solver-verified decision (Stage 3's `VerifiedDecision`) against explicit
per-attribute utility functions, flag residual weaknesses, and emit a
final ADR with full provenance (precedents, agent positions, constraints
checked, repair iterations) — the complete CADENCE pipeline's terminal
output.

**Utility function design:** spec §3 calls for "explicit per-attribute
utility functions," not just an LLM asked to rate a number — so each
quality attribute's utility score is a **blend of two independently
computable components**, not a single opaque LLM call:

1. **Structural component** (deterministic, reproducible, no LLM
   variance) — derived directly from Stage 3's already-computed
   `VerifiedDecision`/tactic data: 1.0 if the attribute is covered by a
   selected tactic, else 0.0, minus a penalty for each *other* selected
   tactic that trades off against it (both facts already exist in
   `src/deliberation/knowledge_graph.py`'s `Tactic.trade_offs` — Stage 4
   doesn't recompute anything Stage 3 already established, it reuses it).
2. **Qualitative component** (LLM self-critique) — a single LLM call
   (separate from the Stage 2 deliberation agents and the Stage 3
   synthesizer/repair client — a neutral critic, per spec's "separate...
   pass") reads the final decision + rationale and scores each attribute
   0–10 with a named weakness or "none."

The final `UtilityScore` per attribute is an explicit, documented formula
combining both (not "ask the LLM for a score" alone) — this is the
"explicit... utility function" spec §3 asks for, while still capturing
the qualitative judgment only an LLM pass can provide.

**Architecture:** `src/critique` package, three layers — structural
utility (pure, no LLM), LLM-based qualitative critique (LLM client DI,
same duck-typed interface as `src/deliberation`/`src/solver`), and
finalization (combines both plus every prior stage's output into one
`FinalADR`). A script chains all four CADENCE stages for a real run.

**Tech Stack:** Python 3.13 (conda env `py313`), no new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-25-cadence-adr-algorithm-design.md` (§3 Stage 4).

## Global Constraints

- Python 3.13, run via `conda activate py313`.
- Unit tests must not call a real LLM — the LLM-critique layer takes an
  injected client (fake in tests), same pattern as every prior stage.
- **Apply the tolerant-structured-output-parsing lesson from Stage 3 up
  front, not after a crash:** the LLM critique response format's parser
  must tolerate case variation, markdown decoration, and a short run of
  extra descriptor words before a label's colon — build it this way from
  the start (see `PROGRESS.md`'s environment notes on why).
- Reuse `QUALITY_ATTRIBUTES`, `Tactic` from `src/deliberation/knowledge_graph.py`;
  `VerifiedDecision` from `src/solver/repair.py`; `DeliberationResult`,
  `AgentPosition` from `src/deliberation/orchestrator.py`/`agent.py`;
  `ADRRecord` from `src/retrieval/records.py`. Do not duplicate these.
- Commit after every task; push after every commit.

---

### Task 1: Structural utility component

**Files:**
- Create: `src/critique/__init__.py`
- Create: `src/critique/structural_utility.py`
- Create: `tests/critique/__init__.py`
- Create: `tests/critique/test_structural_utility.py`

**Interfaces:**
- Consumes: `Tactic` (`src/deliberation/knowledge_graph.py`).
- Produces: `compute_structural_utility(quality_attribute: str, selected_tactics: list[str], covered_quality_attributes: list[str], tactics: list[Tactic]) -> float` —
  returns a value in `[0.0, 1.0]`: `1.0` if `quality_attribute` is in
  `covered_quality_attributes`, else `0.0`; then subtracts `0.2` per
  selected tactic (by name, looked up in `tactics`) whose `trade_offs`
  includes `quality_attribute`, floored at `0.0`. Task 3 calls this once
  per quality attribute using `VerifiedDecision.selected_tactics`/
  `.covered_quality_attributes` (Stage 3's already-computed output).

- [ ] **Step 1: Write the failing tests**

```python
# tests/critique/test_structural_utility.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/critique/test_structural_utility.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.critique'`.

- [ ] **Step 3: Write the implementation**

```python
# src/critique/__init__.py
```

```python
# tests/critique/__init__.py
```

```python
# src/critique/structural_utility.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/critique/test_structural_utility.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/critique/__init__.py src/critique/structural_utility.py tests/critique/__init__.py tests/critique/test_structural_utility.py
git commit -m "feat: add deterministic structural utility component for self-critique"
git push
```

---

### Task 2: LLM qualitative critique

**Files:**
- Create: `src/critique/llm_critique.py`
- Create: `tests/critique/test_llm_critique.py`

**Interfaces:**
- Consumes: `QUALITY_ATTRIBUTES` (`src/deliberation/knowledge_graph.py`);
  any `generate(prompt, system=None)`-shaped client (duck-typed).
- Produces:
  - `QualitativeScore` — frozen dataclass: `quality_attribute: str`,
    `score: float` (0–10), `weakness: str | None` (`None` means "none
    flagged").
  - `CritiqueParseError` (`RuntimeError` subclass).
  - `run_qualitative_critique(decision: str, rationale: str, quality_attributes: tuple[str, ...], client) -> list[QualitativeScore]` —
    builds one prompt asking for `<QA>_SCORE: <0-10>` /
    `<QA>_WEAKNESS: <text or none>` per attribute (one LLM call for all
    attributes, not N calls), parses the response with a tolerant,
    case-insensitive, markdown/extra-word-tolerant line matcher (per the
    Global Constraints lesson) into one `QualitativeScore` per attribute,
    raising `CritiqueParseError` (naming which attributes it couldn't
    parse) if any attribute's score is missing. Task 3 calls this once
    per `FinalADR` build.

- [ ] **Step 1: Write the failing tests**

```python
# tests/critique/test_llm_critique.py
import pytest

from src.critique.llm_critique import CritiqueParseError, QualitativeScore, run_qualitative_critique


class _FakeClient:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate(self, prompt, system=None):
        self.prompts.append(prompt)
        return self.response


def test_parses_score_and_weakness_per_attribute():
    response = (
        "PERFORMANCE_SCORE: 8\n"
        "PERFORMANCE_WEAKNESS: none\n"
        "SECURITY_SCORE: 6\n"
        "SECURITY_WEAKNESS: Relies on a single authentication factor.\n"
    )
    client = _FakeClient(response)

    result = run_qualitative_critique(
        decision="Use caching and authentication.",
        rationale="Balances performance and security.",
        quality_attributes=("performance", "security"),
        client=client,
    )

    assert result == [
        QualitativeScore("performance", 8.0, None),
        QualitativeScore("security", 6.0, "Relies on a single authentication factor."),
    ]
    assert "Use caching and authentication." in client.prompts[0]


def test_tolerant_of_markdown_case_and_extra_words_before_colon():
    response = (
        "**Performance Score:** 7\n"
        "**Performance Weakness Notes:** none\n"
    )
    client = _FakeClient(response)

    result = run_qualitative_critique(
        decision="d", rationale="r", quality_attributes=("performance",), client=client,
    )

    assert result == [QualitativeScore("performance", 7.0, None)]


def test_raises_naming_missing_attributes():
    response = "PERFORMANCE_SCORE: 8\nPERFORMANCE_WEAKNESS: none\n"  # security missing
    client = _FakeClient(response)

    with pytest.raises(CritiqueParseError, match="security"):
        run_qualitative_critique(
            decision="d", rationale="r",
            quality_attributes=("performance", "security"), client=client,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/critique/test_llm_critique.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.critique.llm_critique'`.

- [ ] **Step 3: Write the implementation**

```python
# src/critique/llm_critique.py
"""LLM-based qualitative critique (CADENCE Stage 4) — the "separate LLM
pass" spec §3 asks for, distinct from Stage 2's deliberation agents and
Stage 3's synthesizer/repair client.
"""
import re
from dataclasses import dataclass


class CritiqueParseError(RuntimeError):
    """Raised when the critique response is missing a score for one or
    more requested quality attributes."""


@dataclass(frozen=True)
class QualitativeScore:
    quality_attribute: str
    score: float
    weakness: str | None


def _label_pattern(quality_attribute: str, field: str) -> re.Pattern:
    # Tolerant of markdown, case, and a short run of extra descriptor
    # words before the colon -- see PROGRESS.md's environment notes on
    # why this must be built in from the start, not added after a crash.
    qa_word = quality_attribute.replace("_", "[ _]?")
    return re.compile(
        rf"^\s*[*_\-\s]*{qa_word}[ _]?{field}[A-Za-z\s]{{0,30}}?[*_\s]*:[*_\s]*(.*)$",
        re.IGNORECASE,
    )


def run_qualitative_critique(
    decision: str, rationale: str, quality_attributes: tuple[str, ...], client
) -> list[QualitativeScore]:
    fields = "\n".join(
        f"{qa.upper()}_SCORE: <0-10>\n{qa.upper()}_WEAKNESS: <specific weakness, or 'none'>"
        for qa in quality_attributes
    )
    prompt = (
        f"Decision: {decision}\n"
        f"Rationale: {rationale}\n\n"
        "Critique this architectural decision from each quality attribute "
        "perspective below. Score each 0 (fails this attribute) to 10 "
        "(exemplary), and name one specific residual weakness if any exist "
        "(or 'none'). Respond in exactly this format, one block per "
        f"attribute:\n{fields}"
    )
    response = client.generate(prompt)

    scores = []
    missing = []
    for qa in quality_attributes:
        score_match = _label_pattern(qa, "SCORE").search(response)
        weakness_match = _label_pattern(qa, "WEAKNESS").search(response)
        if not score_match:
            missing.append(qa)
            continue
        score = float(score_match.group(1).strip())
        weakness_text = weakness_match.group(1).strip() if weakness_match else ""
        weakness = None if not weakness_text or weakness_text.lower() == "none" else weakness_text
        scores.append(QualitativeScore(qa, score, weakness))

    if missing:
        raise CritiqueParseError(
            f"Could not parse a score for: {', '.join(missing)}. Response: {response!r}"
        )
    return scores
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/critique/test_llm_critique.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/critique/llm_critique.py tests/critique/test_llm_critique.py
git commit -m "feat: add LLM qualitative critique pass with tolerant structured-output parsing"
git push
```

---

### Task 3: Finalization — combine into a FinalADR with full provenance

**Files:**
- Create: `src/critique/finalize.py`
- Create: `tests/critique/test_finalize.py`

**Interfaces:**
- Consumes: `compute_structural_utility` (Task 1); `run_qualitative_critique`,
  `QualitativeScore` (Task 2); `VerifiedDecision` (`src/solver/repair.py`);
  `DeliberationResult`, `AgentPosition` (`src/deliberation/orchestrator.py`,
  `.agent`); `ADRRecord` (`src/retrieval/records.py`); `Tactic`
  (`src/deliberation/knowledge_graph.py`).
- Produces:
  - `UtilityScore` — frozen dataclass: `quality_attribute: str`,
    `structural_component: float`, `qualitative_component: float`,
    `combined_score: float` (documented formula:
    `0.5 * structural_component * 10 + 0.5 * qualitative_component` — both
    scaled to a 0–10 range before blending), `weakness: str | None`.
  - `FinalADR` — frozen dataclass: `decision: str`, `rationale: str`,
    `utility_scores: list[UtilityScore]`, `overall_score: float` (mean of
    `combined_score`), `residual_weaknesses: list[str]`, `is_feasible: bool`,
    `selected_tactics: list[str]`, `covered_quality_attributes: list[str]`,
    `uncovered_quality_attributes: list[str]`, `repair_iterations: int`,
    `solver_caveat: str | None`, `precedent_titles: list[str]`,
    `deliberation_transcript: list[AgentPosition]`.
  - `finalize_decision(verified_decision, deliberation_result, precedents, quality_attributes, tactics, critique_client) -> FinalADR` —
    computes structural utility per attribute, runs one qualitative
    critique call, blends them into `UtilityScore`s, and assembles the
    complete `FinalADR` with every prior stage's provenance. **This is the
    CADENCE pipeline's terminal function** — a real end-to-end script
    (Task 4) calls Stage 1 → 2 → 3 → this.

- [ ] **Step 1: Write the failing tests**

```python
# tests/critique/test_finalize.py
from src.deliberation.agent import AgentPosition
from src.deliberation.knowledge_graph import Tactic
from src.deliberation.orchestrator import DeliberationResult
from src.retrieval.records import ADRRecord
from src.solver.repair import VerifiedDecision
from src.critique.finalize import FinalADR, finalize_decision


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/critique/test_finalize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.critique.finalize'`.

- [ ] **Step 3: Write the implementation**

```python
# src/critique/finalize.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/critique/test_finalize.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/critique/finalize.py tests/critique/test_finalize.py
git commit -m "feat: assemble FinalADR combining Stage 4 utility scoring with full pipeline provenance"
git push
```

---

### Task 4: Real end-to-end demo (all four CADENCE stages)

**Files:**
- Create: `scripts/run_cadence_demo.py`
- Create: `tests/data/test_run_cadence_demo_script.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3 plus every prior stage's real entry
  point (`Retriever`, `DeliberationOrchestrator`, `run_repair_loop`).
- Produces: `scripts/run_cadence_demo.py` — the complete, real, four-stage
  CADENCE pipeline run: Stage 1 retrieval → Stage 2 deliberation → Stage 3
  solver+repair → Stage 4 self-critique/finalization, printing the full
  `FinalADR`. This is the reference implementation the evaluation-harness
  plan will drive at scale.

- [ ] **Step 1: Write the failing test for the script's wiring**

```python
# tests/data/test_run_cadence_demo_script.py
import json

import numpy as np


def test_run_cadence_demo_wires_all_four_stages_together(tmp_path, monkeypatch):
    from scripts.run_cadence_demo import run_cadence_demo

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
            if "SCORE" in prompt:
                return "\n".join(
                    f"{qa.upper()}_SCORE: 7\n{qa.upper()}_WEAKNESS: none"
                    for qa in ["performance", "security", "maintainability", "scalability", "cost_operability"]
                )
            return "CANDIDATE: We will use caching.\nRATIONALE: Improves performance within budget."

    class _FakeEmbeddingModel:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr("scripts.run_cadence_demo.load_local_hf_client", lambda: _FakeClient())
    monkeypatch.setattr("scripts.run_cadence_demo.load_embedding_model", lambda: _FakeEmbeddingModel())

    result = run_cadence_demo(
        records_path=records_path, embeddings_path=embeddings_path,
        context="Sample decision context.", max_rounds=1, tactic_budget=2, max_repair_iterations=1,
    )

    assert result.decision == "We will use caching."
    assert len(result.utility_scores) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/data/test_run_cadence_demo_script.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_cadence_demo'`.

- [ ] **Step 3: Write the script**

```python
# scripts/run_cadence_demo.py
"""The complete, real, four-stage CADENCE pipeline: retrieval ->
deliberation -> solver+repair -> self-critique/finalization.

Import order: src.retrieval.embeddings (sentence_transformers) is imported
before anything that triggers torch via the local-HF client, per the
sentence_transformers-before-torch rule in PROGRESS.md.
"""
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.retrieval.embeddings import embed_texts, load_embedding_model  # noqa: F401  (import before torch)
import numpy as np

from src.critique.finalize import FinalADR, finalize_decision
from src.deliberation.agent import QualityAttributeAgent
from src.deliberation.knowledge_graph import QUALITY_ATTRIBUTES, TACTICS, build_knowledge_graph
from src.deliberation.llm_client import load_local_hf_client
from src.deliberation.orchestrator import DeliberationOrchestrator
from src.retrieval.records import ADRRecord
from src.retrieval.retriever import Retriever
from src.solver.repair import run_repair_loop

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


def run_cadence_demo(
    records_path: Path,
    embeddings_path: Path,
    context: str,
    max_rounds: int = 2,
    tactic_budget: int = 4,
    max_repair_iterations: int = 2,
) -> FinalADR:
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

    verified = run_repair_loop(
        candidate=deliberation.converged_candidate,
        rationale=deliberation.rationale,
        required_quality_attributes=QUALITY_ATTRIBUTES,
        tactic_budget=tactic_budget,
        quality_attribute_weights={},
        tactics=TACTICS,
        repair_client=llm_client,
        max_repair_iterations=max_repair_iterations,
    )

    return finalize_decision(
        verified_decision=verified,
        deliberation_result=deliberation,
        precedents=precedents,
        quality_attributes=QUALITY_ATTRIBUTES,
        tactics=TACTICS,
        critique_client=llm_client,
    )


if __name__ == "__main__":
    adr = run_cadence_demo(RECORDS_PATH, EMBEDDINGS_PATH, SAMPLE_CONTEXT)
    print("=== Final ADR ===")
    print(f"Decision: {adr.decision}")
    print(f"Rationale: {adr.rationale}")
    print(f"Feasible: {adr.is_feasible} (repair iterations: {adr.repair_iterations})")
    if adr.solver_caveat:
        print(f"Solver caveat: {adr.solver_caveat}")
    print(f"Overall utility score: {adr.overall_score:.2f}/10")
    print("Per-attribute utility:")
    for score in adr.utility_scores:
        print(
            f"  {score.quality_attribute}: {score.combined_score:.2f}/10 "
            f"(structural={score.structural_component:.1f}, qualitative={score.qualitative_component:.1f})"
        )
        if score.weakness:
            print(f"    weakness: {score.weakness}")
    print(f"Precedents used: {adr.precedent_titles}")
    print(f"Deliberation transcript: {len(adr.deliberation_transcript)} positions across the debate")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/data/test_run_cadence_demo_script.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the full suite, then run the real demo**

Run: `conda run -n py313 pytest -q` — expect all tests (old + new) passing.

Run: `conda run -n py313 python scripts/run_cadence_demo.py` (or the env's
`python.exe` directly if `conda run` mis-wraps the invocation) — the
complete real pipeline, all four stages, against the real corpus and
local model. Expect a printed `FinalADR` with plausible per-attribute
scores and at least one flagged weakness (a small local model is unlikely
to rate everything a perfect 10) that completes without crashing.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_cadence_demo.py tests/data/test_run_cadence_demo_script.py
git commit -m "feat: add complete four-stage CADENCE pipeline demo"
git push
```

---

## Self-Review Notes

- **Spec coverage:** implements exactly CADENCE Stage 4 (spec §3) and
  completes the full four-stage pipeline. The next plan (evaluation
  harness) is the first one that runs this at scale over held-out corpus
  examples rather than one sample context.
- **Why the utility function blends structural + qualitative components
  instead of being purely LLM-judged:** spec §3 explicitly says
  "explicit... utility functions," not "an LLM score" — a formula that's
  half deterministic/reproducible (from facts Stage 3 already established)
  and half LLM-judged is a genuine "function," auditable and consistent
  across runs for its structural half, while still capturing qualitative
  judgment an LLM pass can catch that a formula alone cannot (e.g., a
  technically-covered attribute addressed by a weak or vague rationale).
- **Why one LLM call scores all attributes together, not N separate
  calls:** cheaper (one generation instead of five) and lets the critic
  see the full decision holistically rather than attribute-by-attribute
  in isolation, which matters for judging trade-off-aware quality.
- **Why the qualitative critique parser is written tolerant from the
  start:** Stage 3's real end-to-end run crashed on a synthesis format
  deviation from the exact same class of small-model output variance —
  see `PROGRESS.md`'s environment notes. This plan's parser
  (`_label_pattern`) applies that lesson proactively rather than waiting
  to find the same bug again.
- **Placeholder scan:** no TBD/TODO; all code blocks are complete and
  runnable.
- **Type/interface consistency:** `UtilityScore`/`FinalADR` fields match
  between `finalize.py`'s definition and `run_cadence_demo.py`'s print
  statements. `finalize_decision`'s signature matches between Task 3's
  definition and Task 4's call site. `QualitativeScore` fields match
  between `llm_critique.py` (definition) and `finalize.py` (consumption
  via `qualitative_by_qa[qa].score`/`.weakness`).
