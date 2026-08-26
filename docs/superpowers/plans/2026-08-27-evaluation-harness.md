# Evaluation Harness Implementation Plan

**Goal:** Implement spec §5's evaluation plan: run CADENCE and three
baselines over a held-out sample of real, human-authored ADRs, score each
against the ground truth with standard generation metrics (BERTScore,
BLEU, ROUGE-1, METEOR) plus the two metrics only a solver-verified
pipeline can report (constraint-satisfaction rate, repair-loop
convergence), and produce a comparison report. This is what generates the
manuscript's results section — not a formality plan.

**Held-out set and query construction:** the corpus's own upstream package
is literally titled *"ADR Generation from Titles: A Comprehensive
Experimental Study"* (see `data/README.md`'s upstream provenance) — i.e.
the precedent this field already uses for this exact task is: **decision
context = the ADR's title, ground truth = the ADR's full body text.**
This plan adopts the same construction rather than inventing a new one:
sample N records from `data/processed/adr_records.jsonl` filtered to
`extraction_status == "Verified"` (the corpus's own high-confidence
subset) with substantial body length, use `title` as the query context for
every system, and `raw_text` as the reference for metric scoring.

**Baselines (spec §5), mapped to already-built pipeline entry points —
no baseline needs new algorithmic code, only new wiring:**

| Baseline | Stages used | Entry points |
|---|---|---|
| (a) Dhar-et-al.-style zero-shot | none | one `llm_client.generate(prompt)` call |
| (b) Context-Matters-style retrieval-only | Stage 1 | `Retriever.retrieve` + one `generate()` call with precedents as context |
| (c) MAAD-style multi-agent, no solver (ablation) | Stages 1+2 | `Retriever.retrieve` + `DeliberationOrchestrator.deliberate` |
| (d) Full CADENCE | Stages 1–4 | `Retriever.retrieve` + `DeliberationOrchestrator.deliberate` + `run_repair_loop` + `finalize_decision` |
| (e) human-authored ground truth | — | the ADR's own `raw_text` (the reference, not a system) |

Row (c) is simultaneously a baseline *and* the "no solver" ablation spec
§5 asks for — no separate ablation code is needed for that comparison.
"No retrieval" / "no self-critique" ablations are like (a)/(c)
respectively in everything but self-critique's inclusion; this plan
implements the four systems in the table, which cover every ablation
comparison spec §5 lists except a solo "no self-critique but has solver"
variant — noted as a cheap follow-on (skip Stage 4 only) if that specific
comparison is wanted later, not built here to keep this plan's scope to
what's actually requested.

**Leakage prevention:** the retrieval index (`data/processed/adr_embeddings.npy`)
was built over the *entire* corpus, including whatever gets sampled as a
test item — every retrieval call in this plan excludes the query's own
`record_id` from returned precedents so a system can never "retrieve
itself."

**Architecture:** `src/evaluation` package — held-out set construction
(pure), metrics (wraps 4 real metric libraries, batched for efficiency),
system runners (one function per row in the table above, LLM client DI
like every prior stage), and a harness that ties them together into a
comparison report. A script runs a small real evaluation for verification.

**Tech Stack:** Python 3.13 (conda env `py313`), `bert-score`,
`rouge-score`, `nltk` (METEOR — needs `nltk.download("wordnet")` etc. at
runtime, done idempotently in the metrics module), `sacrebleu` (BLEU) —
all installed and smoke-tested this session, no flakiness observed.

**Spec:** `docs/superpowers/specs/2026-08-25-cadence-adr-algorithm-design.md` (§5).

## Global Constraints

- Python 3.13, run via `conda activate py313`.
- Unit tests inject fake LLM clients (established DI pattern) for system
  runners; metrics tests may call the real metric libraries directly on
  tiny fixed strings (deterministic, fast enough — BERTScore's one-time
  model load costs a few seconds, acceptable, same trade-off already made
  for Z3 in the solver plan rather than mocking a fast, reliable
  dependency).
- **Every structured-LLM-output consumer in this plan must use the
  established tolerant-parser-plus-bounded-retry pattern from the start**
  — see `PROGRESS.md`'s environment notes on the five real format
  deviations found in Stages 3–4. The zero-shot and retrieval-only
  baselines here call `generate()` for free text (no structured format to
  parse), so this only applies where this plan reuses Stage 2/3/4 code,
  which already has it.
- The real verification run (Task 4) uses small numbers (N=3 test items,
  `max_rounds=1`, `k=2`) to keep wall-clock time reasonable while still
  proving the harness end-to-end — scaling N/rounds up for an actual
  paper-quality evaluation run is a deliberate follow-on decision, not
  done automatically by this plan.
- Commit after every task; push after every commit.

---

### Task 1: Held-out test set construction

**Files:**
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/held_out_set.py`
- Create: `tests/evaluation/__init__.py`
- Create: `tests/evaluation/test_held_out_set.py`

**Interfaces:**
- Consumes: `ADRRecord` (`src/retrieval/records.py`).
- Produces:
  - `load_verified_records(records_path: Path) -> list[ADRRecord]` — reads
    the JSONL file, filters to `extraction_status == "Verified"`.
  - `sample_test_set(records: list[ADRRecord], n: int, min_length: int = 300, seed: int = 42) -> list[ADRRecord]` —
    filters to `len(raw_text) >= min_length` (excludes near-empty ADRs
    that can't meaningfully be compared against), then
    `random.Random(seed).sample(...)` — fixed seed for reproducibility, a
    real requirement for a research evaluation, not a style preference.
    Raises `ValueError` if fewer than `n` eligible records exist. Task 4's
    script calls both in sequence against the real corpus.

- [ ] **Step 1: Write the failing tests**

```python
# tests/evaluation/test_held_out_set.py
import pytest

from src.retrieval.records import ADRRecord
from src.evaluation.held_out_set import load_verified_records, sample_test_set


def _record(record_id, status, raw_text):
    return ADRRecord(
        record_id=record_id, repo_folder="r", repository_url=None,
        relative_path=record_id, sequence_number=1, title=f"Title {record_id}",
        raw_text=raw_text, extraction_status=status,
    )


def test_load_verified_records_filters_by_status(tmp_path):
    import dataclasses
    import json

    records = [
        _record("a", "Verified", "x" * 400),
        _record("b", "Doubt (name sequence)", "x" * 400),
        _record("c", "Verified", "x" * 400),
    ]
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(json.dumps(dataclasses.asdict(r)) for r in records) + "\n", encoding="utf-8"
    )

    result = load_verified_records(path)

    assert {r.record_id for r in result} == {"a", "c"}


def test_sample_test_set_excludes_short_records():
    records = [_record("short", "Verified", "x" * 50), _record("long", "Verified", "x" * 400)]

    result = sample_test_set(records, n=1, min_length=300)

    assert [r.record_id for r in result] == ["long"]


def test_sample_test_set_is_reproducible_with_same_seed():
    records = [_record(str(i), "Verified", "x" * 400) for i in range(20)]

    first = sample_test_set(records, n=5, seed=7)
    second = sample_test_set(records, n=5, seed=7)

    assert [r.record_id for r in first] == [r.record_id for r in second]


def test_sample_test_set_raises_when_not_enough_eligible_records():
    records = [_record("a", "Verified", "x" * 400)]

    with pytest.raises(ValueError):
        sample_test_set(records, n=5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/evaluation/test_held_out_set.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation'`.

- [ ] **Step 3: Write the implementation**

```python
# src/evaluation/__init__.py
```

```python
# tests/evaluation/__init__.py
```

```python
# src/evaluation/held_out_set.py
"""Held-out evaluation set construction.

Query construction follows the corpus's own upstream precedent for this
exact task ("ADR Generation from Titles" -- see data/README.md): title as
decision context, full body as ground truth. Filters to the corpus's own
high-confidence Verified subset for a clean reference.
"""
import json
import random
from pathlib import Path

from src.retrieval.records import ADRRecord


def load_verified_records(records_path: Path) -> list[ADRRecord]:
    records = []
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            record = ADRRecord(**json.loads(line))
            if record.extraction_status == "Verified":
                records.append(record)
    return records


def sample_test_set(
    records: list[ADRRecord], n: int, min_length: int = 300, seed: int = 42
) -> list[ADRRecord]:
    eligible = [r for r in records if len(r.raw_text) >= min_length]
    if len(eligible) < n:
        raise ValueError(f"Only {len(eligible)} eligible records, need {n}")
    return random.Random(seed).sample(eligible, n)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/evaluation/test_held_out_set.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Sanity-check against the real corpus (manual, not a pytest)**

Run as a script file:
```python
from src.evaluation.held_out_set import load_verified_records, sample_test_set
records = load_verified_records("data/processed/adr_records.jsonl")
print(f"{len(records)} Verified records")
sample = sample_test_set(records, n=5)
for r in sample:
    print(f"  [{r.repo_folder}] {r.title!r} ({len(r.raw_text)} chars)")
```
Expected: prints ~750 Verified records (matches the corpus schema
inspection in `PROGRESS.md`) and 5 sampled titles with substantial body
lengths.

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/__init__.py src/evaluation/held_out_set.py tests/evaluation/__init__.py tests/evaluation/test_held_out_set.py
git commit -m "feat: add held-out evaluation set construction"
git push
```

---

### Task 2: Generation metrics

**Files:**
- Create: `src/evaluation/metrics.py`
- Create: `tests/evaluation/test_metrics.py`

**Interfaces:**
- Consumes: `bert_score`, `rouge_score`, `nltk`, `sacrebleu`.
- Produces:
  - `MetricScores` — frozen dataclass: `bertscore_f1: float`, `bleu: float`,
    `rouge1_f: float`, `meteor: float`.
  - `ensure_nltk_data() -> None` — idempotently downloads `wordnet`,
    `punkt_tab`, `omw-1.4` (quiet, safe to call every run).
  - `compute_corpus_metrics(generated_texts: list[str], reference_texts: list[str]) -> list[MetricScores]` —
    one `MetricScores` per pair, same order as input. BERTScore is
    computed once for the whole batch (`bert_score.score(generated_texts, reference_texts, lang="en")`) —
    much faster than per-pair scoring, which would reload the model
    conceptually-per-call-overhead N times. Task 4's harness calls this
    once per system (list of N generated outputs vs. N references), not
    once per item.
  - `average_scores(scores: list[MetricScores]) -> MetricScores` — mean of
    each field, for the summary table.

- [ ] **Step 1: Write the failing tests**

```python
# tests/evaluation/test_metrics.py
from src.evaluation.metrics import MetricScores, average_scores, compute_corpus_metrics, ensure_nltk_data

ensure_nltk_data()  # module-level: METEOR needs this before any test runs


def test_compute_corpus_metrics_returns_one_score_per_pair():
    generated = ["The cat sat on the rug.", "We will use caching."]
    reference = ["The cat sat on the mat.", "We will use caching for performance."]

    scores = compute_corpus_metrics(generated, reference)

    assert len(scores) == 2
    for s in scores:
        assert isinstance(s, MetricScores)
        assert 0.0 <= s.bertscore_f1 <= 1.0
        assert 0.0 <= s.bleu <= 100.0
        assert 0.0 <= s.rouge1_f <= 1.0
        assert 0.0 <= s.meteor <= 1.0


def test_identical_text_scores_near_perfect():
    scores = compute_corpus_metrics(["Use read replicas for scaling."], ["Use read replicas for scaling."])

    assert scores[0].bertscore_f1 > 0.99
    assert scores[0].rouge1_f == 1.0
    assert scores[0].meteor > 0.9


def test_completely_unrelated_text_scores_low():
    scores = compute_corpus_metrics(["The weather is sunny today."], ["We will use caching for performance."])

    assert scores[0].rouge1_f < 0.3
    assert scores[0].bleu < 20.0


def test_average_scores_computes_mean_per_field():
    scores = [
        MetricScores(bertscore_f1=0.8, bleu=20.0, rouge1_f=0.5, meteor=0.4),
        MetricScores(bertscore_f1=0.6, bleu=10.0, rouge1_f=0.3, meteor=0.2),
    ]

    avg = average_scores(scores)

    assert avg.bertscore_f1 == 0.7
    assert avg.bleu == 15.0
    assert avg.rouge1_f == 0.4
    assert avg.meteor == 0.30000000000000004 or abs(avg.meteor - 0.3) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/evaluation/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation.metrics'`.

- [ ] **Step 3: Write the implementation**

```python
# src/evaluation/metrics.py
"""Standard generation metrics (spec §5) for comparing a system's output
against a held-out ADR's real body text.
"""
from dataclasses import dataclass

import nltk
import sacrebleu
from rouge_score import rouge_scorer


@dataclass(frozen=True)
class MetricScores:
    bertscore_f1: float
    bleu: float
    rouge1_f: float
    meteor: float


def ensure_nltk_data() -> None:
    for resource in ("wordnet", "punkt_tab", "omw-1.4"):
        nltk.download(resource, quiet=True)


def compute_corpus_metrics(generated_texts: list[str], reference_texts: list[str]) -> list[MetricScores]:
    import bert_score
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score

    ensure_nltk_data()

    _, _, bertscore_f1s = bert_score.score(generated_texts, reference_texts, lang="en", verbose=False)

    rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    scores = []
    for i, (generated, reference) in enumerate(zip(generated_texts, reference_texts)):
        bleu = sacrebleu.sentence_bleu(generated, [reference]).score
        rouge1_f = rouge.score(reference, generated)["rouge1"].fmeasure
        meteor = meteor_score([word_tokenize(reference)], word_tokenize(generated))
        scores.append(
            MetricScores(
                bertscore_f1=bertscore_f1s[i].item(),
                bleu=bleu,
                rouge1_f=rouge1_f,
                meteor=meteor,
            )
        )
    return scores


def average_scores(scores: list[MetricScores]) -> MetricScores:
    n = len(scores)
    return MetricScores(
        bertscore_f1=sum(s.bertscore_f1 for s in scores) / n,
        bleu=sum(s.bleu for s in scores) / n,
        rouge1_f=sum(s.rouge1_f for s in scores) / n,
        meteor=sum(s.meteor for s in scores) / n,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/evaluation/test_metrics.py -v`
Expected: PASS (4 passed). Note: first run downloads BERTScore's default
model (~1.4 GB for the default `roberta-large` — consider passing a
smaller `model_type` if this is too slow/large for routine test runs;
decide based on the real download time observed in Step 4, and document
the choice in Self-Review Notes rather than guessing up front).

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/evaluation/test_metrics.py
git commit -m "feat: add batched generation metrics (BERTScore, BLEU, ROUGE-1, METEOR)"
git push
```

---

### Task 3: System runners (baselines + full CADENCE)

**Files:**
- Create: `src/evaluation/systems.py`
- Create: `tests/evaluation/test_systems.py`

**Interfaces:**
- Consumes: `Retriever` (`src/retrieval/retriever.py`);
  `QualityAttributeAgent`, `DeliberationOrchestrator`,
  `build_knowledge_graph` (`src/deliberation`); `run_repair_loop`
  (`src/solver/repair.py`); `finalize_decision` (`src/critique/finalize.py`);
  any `generate(prompt, system=None)`-shaped client (duck-typed).
- Produces:
  - `SystemOutput` — frozen dataclass: `system_name: str`,
    `generated_text: str`, `is_feasible: bool | None`,
    `repair_iterations: int | None` (both `None` for systems without a
    solver stage).
  - `retrieve_excluding_self(retriever: Retriever, query: str, exclude_record_id: str, k: int) -> list[ADRRecord]` —
    calls `retriever.retrieve(query, k=k+1)` (one extra, to make room for
    a possible self-match) then filters out `exclude_record_id` and trims
    to `k`. Every system below that uses retrieval calls this, never
    `retriever.retrieve` directly, so leakage exclusion is never
    forgotten.
  - `run_zero_shot(context: str, client) -> SystemOutput`.
  - `run_retrieval_only(context: str, retriever: Retriever, exclude_record_id: str, client, k: int = 3) -> SystemOutput`.
  - `run_multiagent_no_solver(context: str, retriever: Retriever, exclude_record_id: str, client, graph, quality_attributes: tuple[str, ...], k: int = 3, max_rounds: int = 2) -> SystemOutput`.
  - `run_cadence_full(context: str, retriever: Retriever, exclude_record_id: str, client, graph, tactics, quality_attributes: tuple[str, ...], k: int = 3, max_rounds: int = 2, tactic_budget: int = 4, max_repair_iterations: int = 2) -> SystemOutput`.
  Task 4's harness calls all four per test item.

- [ ] **Step 1: Write the failing tests**

```python
# tests/evaluation/test_systems.py
import numpy as np

from src.deliberation.knowledge_graph import TACTICS, build_knowledge_graph, QUALITY_ATTRIBUTES
from src.retrieval.records import ADRRecord
from src.retrieval.retriever import Retriever
from src.evaluation.systems import (
    SystemOutput, retrieve_excluding_self, run_zero_shot, run_retrieval_only,
    run_multiagent_no_solver, run_cadence_full,
)


class _FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0] for _ in texts])


def _make_retriever(record_ids):
    records = [
        ADRRecord(record_id=rid, repo_folder="r", repository_url=None, relative_path=rid,
                   sequence_number=1, title=f"Title {rid}", raw_text=f"text {rid}", extraction_status="Verified")
        for rid in record_ids
    ]
    embeddings = np.array([[1.0, 0.0] for _ in records])
    return Retriever(records, embeddings, _FakeEmbeddingModel())


class _FakeClient:
    def __init__(self, response="generated text"):
        self.response = response
        self.prompts = []

    def generate(self, prompt, system=None):
        self.prompts.append(prompt)
        return self.response


def test_retrieve_excluding_self_never_returns_the_excluded_record():
    retriever = _make_retriever(["self", "a", "b", "c"])

    results = retrieve_excluding_self(retriever, "query", exclude_record_id="self", k=2)

    assert "self" not in [r.record_id for r in results]
    assert len(results) == 2


def test_run_zero_shot_calls_client_once_with_only_the_context():
    client = _FakeClient("Use caching.")

    result = run_zero_shot("Handle high read traffic.", client)

    assert isinstance(result, SystemOutput)
    assert result.system_name == "zero_shot"
    assert result.generated_text == "Use caching."
    assert result.is_feasible is None
    assert "Handle high read traffic." in client.prompts[0]


def test_run_retrieval_only_includes_precedents_in_the_prompt():
    retriever = _make_retriever(["self", "a"])
    client = _FakeClient("Use read replicas.")

    result = run_retrieval_only("ctx", retriever, exclude_record_id="self", client=client, k=1)

    assert result.system_name == "retrieval_only"
    assert "Title a" in client.prompts[0]


def test_run_multiagent_no_solver_returns_deliberation_output_with_no_feasibility():
    retriever = _make_retriever(["self", "a"])
    graph = build_knowledge_graph(TACTICS)

    class _SynthesisClient(_FakeClient):
        def generate(self, prompt, system=None):
            self.prompts.append(prompt)
            return "CANDIDATE: Use caching.\nRATIONALE: Improves performance."

    client = _SynthesisClient()
    result = run_multiagent_no_solver(
        "ctx", retriever, exclude_record_id="self", client=client, graph=graph,
        quality_attributes=("performance",), k=1, max_rounds=1,
    )

    assert result.system_name == "multiagent_no_solver"
    assert "Use caching." in result.generated_text
    assert result.is_feasible is None
    assert result.repair_iterations is None


def test_run_cadence_full_returns_feasibility_and_repair_iterations():
    retriever = _make_retriever(["self", "a"])
    graph = build_knowledge_graph(TACTICS)

    class _AllPurposeClient(_FakeClient):
        def generate(self, prompt, system=None):
            self.prompts.append(prompt)
            if "SCORE" in prompt:
                return "\n".join(
                    f"{qa.upper()}_SCORE: 7\n{qa.upper()}_WEAKNESS: none" for qa in QUALITY_ATTRIBUTES
                )
            return "CANDIDATE: Use caching.\nRATIONALE: Improves performance."

    client = _AllPurposeClient()
    result = run_cadence_full(
        "ctx", retriever, exclude_record_id="self", client=client, graph=graph, tactics=TACTICS,
        quality_attributes=QUALITY_ATTRIBUTES, k=1, max_rounds=1, tactic_budget=2, max_repair_iterations=1,
    )

    assert result.system_name == "cadence_full"
    assert result.is_feasible is not None
    assert result.repair_iterations is not None
    assert "Use caching." in result.generated_text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/evaluation/test_systems.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation.systems'`.

- [ ] **Step 3: Write the implementation**

```python
# src/evaluation/systems.py
"""Baseline and full-CADENCE system runners for evaluation (spec §5)."""
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
    return filtered[:k]


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
    return SystemOutput("cadence_full", text, is_feasible=final_adr.is_feasible, repair_iterations=final_adr.repair_iterations)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/evaluation/test_systems.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/systems.py tests/evaluation/test_systems.py
git commit -m "feat: add baseline and full-CADENCE system runners for evaluation"
git push
```

---

### Task 4: Evaluation harness and real verification run

**Files:**
- Create: `src/evaluation/harness.py`
- Create: `scripts/run_evaluation.py`
- Create: `tests/evaluation/test_harness.py`
- Create: `tests/data/test_run_evaluation_script.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces:
  - `SystemReport` — frozen dataclass: `system_name: str`,
    `average_scores: MetricScores`, `feasibility_rate: float | None`
    (fraction of items with `is_feasible=True`, `None` if the system
    never reports feasibility), `average_repair_iterations: float | None`.
  - `EvaluationReport` — frozen dataclass: `system_reports: list[SystemReport]`,
    `n_items: int`.
  - `run_evaluation(test_records: list[ADRRecord], retriever: Retriever, client, graph, tactics, quality_attributes: tuple[str, ...], **system_kwargs) -> EvaluationReport` —
    for each test record, runs all four systems (each retrieval call
    excludes that record's own `record_id`), collects `SystemOutput`s;
    batches metric computation per system (all N outputs vs. all N
    references in one `compute_corpus_metrics` call, not N separate
    calls); assembles the report. **This is the evaluation harness's
    entry point** — `scripts/run_evaluation.py` calls it against the real
    corpus.
  - `scripts/run_evaluation.py` — real script: loads the held-out set (3
    items, per Global Constraints), builds the real `Retriever` from
    `data/processed/`, runs `run_evaluation` with the real local LLM
    client, prints a comparison table.

- [ ] **Step 1: Write the failing tests**

```python
# tests/evaluation/test_harness.py
import numpy as np

from src.deliberation.knowledge_graph import TACTICS, build_knowledge_graph, QUALITY_ATTRIBUTES
from src.retrieval.records import ADRRecord
from src.retrieval.retriever import Retriever
from src.evaluation.harness import EvaluationReport, run_evaluation


class _FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0] for _ in texts])


class _AllPurposeClient:
    def generate(self, prompt, system=None):
        if "SCORE" in prompt:
            return "\n".join(
                f"{qa.upper()}_SCORE: 7\n{qa.upper()}_WEAKNESS: none" for qa in QUALITY_ATTRIBUTES
            )
        return "CANDIDATE: Use caching.\nRATIONALE: Improves performance."


def _record(rid, text):
    return ADRRecord(record_id=rid, repo_folder="r", repository_url=None, relative_path=rid,
                       sequence_number=1, title=f"Title {rid}", raw_text=text, extraction_status="Verified")


def test_run_evaluation_produces_a_report_with_all_four_systems():
    test_records = [_record("t1", "Use caching for performance and low latency."), _record("t2", "Use replicas.")]
    all_records = test_records + [_record("other", "Some other precedent.")]
    embeddings = np.array([[1.0, 0.0] for _ in all_records])
    retriever = Retriever(all_records, embeddings, _FakeEmbeddingModel())
    graph = build_knowledge_graph(TACTICS)
    client = _AllPurposeClient()

    report = run_evaluation(
        test_records=test_records, retriever=retriever, client=client, graph=graph,
        tactics=TACTICS, quality_attributes=QUALITY_ATTRIBUTES,
        k=1, max_rounds=1, tactic_budget=2, max_repair_iterations=1,
    )

    assert isinstance(report, EvaluationReport)
    assert report.n_items == 2
    assert {r.system_name for r in report.system_reports} == {
        "zero_shot", "retrieval_only", "multiagent_no_solver", "cadence_full",
    }
    cadence_report = next(r for r in report.system_reports if r.system_name == "cadence_full")
    assert cadence_report.feasibility_rate is not None
    assert cadence_report.average_repair_iterations is not None
    zero_shot_report = next(r for r in report.system_reports if r.system_name == "zero_shot")
    assert zero_shot_report.feasibility_rate is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/evaluation/test_harness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation.harness'`.

- [ ] **Step 3: Write the harness**

```python
# src/evaluation/harness.py
"""Evaluation harness (spec §5): run every system over the held-out set,
score against ground truth, produce a comparison report.
"""
from dataclasses import dataclass

from src.evaluation.metrics import MetricScores, average_scores, compute_corpus_metrics
from src.evaluation.systems import (
    run_cadence_full, run_multiagent_no_solver, run_retrieval_only, run_zero_shot,
)
from src.retrieval.records import ADRRecord
from src.retrieval.retriever import Retriever


@dataclass(frozen=True)
class SystemReport:
    system_name: str
    average_scores: MetricScores
    feasibility_rate: float | None
    average_repair_iterations: float | None


@dataclass(frozen=True)
class EvaluationReport:
    system_reports: list[SystemReport]
    n_items: int


def run_evaluation(
    test_records: list[ADRRecord],
    retriever: Retriever,
    client,
    graph,
    tactics,
    quality_attributes: tuple[str, ...],
    k: int = 3,
    max_rounds: int = 2,
    tactic_budget: int = 4,
    max_repair_iterations: int = 2,
) -> EvaluationReport:
    references = [r.raw_text for r in test_records]
    outputs_by_system: dict[str, list] = {
        "zero_shot": [], "retrieval_only": [], "multiagent_no_solver": [], "cadence_full": [],
    }

    for record in test_records:
        context = record.title
        outputs_by_system["zero_shot"].append(run_zero_shot(context, client))
        outputs_by_system["retrieval_only"].append(
            run_retrieval_only(context, retriever, record.record_id, client, k=k)
        )
        outputs_by_system["multiagent_no_solver"].append(
            run_multiagent_no_solver(
                context, retriever, record.record_id, client, graph, quality_attributes,
                k=k, max_rounds=max_rounds,
            )
        )
        outputs_by_system["cadence_full"].append(
            run_cadence_full(
                context, retriever, record.record_id, client, graph, tactics, quality_attributes,
                k=k, max_rounds=max_rounds, tactic_budget=tactic_budget,
                max_repair_iterations=max_repair_iterations,
            )
        )

    system_reports = []
    for system_name, outputs in outputs_by_system.items():
        generated = [o.generated_text for o in outputs]
        scores = compute_corpus_metrics(generated, references)
        feasibilities = [o.is_feasible for o in outputs if o.is_feasible is not None]
        repairs = [o.repair_iterations for o in outputs if o.repair_iterations is not None]
        system_reports.append(
            SystemReport(
                system_name=system_name,
                average_scores=average_scores(scores),
                feasibility_rate=(sum(feasibilities) / len(feasibilities)) if feasibilities else None,
                average_repair_iterations=(sum(repairs) / len(repairs)) if repairs else None,
            )
        )

    return EvaluationReport(system_reports=system_reports, n_items=len(test_records))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/evaluation/test_harness.py -v`
Expected: PASS (1 passed). This test loads real `bert_score`/`nltk`/etc.
(via `compute_corpus_metrics`, not mocked, per Global Constraints) so
expect the same first-run model-download cost as Task 2.

- [ ] **Step 5: Write the failing test for the real script's wiring**

```python
# tests/data/test_run_evaluation_script.py
import json

import numpy as np


def test_run_evaluation_script_wires_everything_together(tmp_path, monkeypatch):
    from scripts.run_evaluation import run_evaluation_script

    records_path = tmp_path / "adr_records.jsonl"
    records = [
        {"record_id": f"r{i}", "repo_folder": "r", "repository_url": None, "relative_path": f"{i}.md",
         "sequence_number": 1, "title": f"Title {i}", "raw_text": "x" * 400, "extraction_status": "Verified"}
        for i in range(3)
    ]
    records_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    embeddings_path = tmp_path / "adr_embeddings.npy"
    np.save(embeddings_path, np.array([[1.0, 0.0] for _ in records]))

    class _FakeClient:
        def generate(self, prompt, system=None):
            if "SCORE" in prompt:
                from src.deliberation.knowledge_graph import QUALITY_ATTRIBUTES
                return "\n".join(
                    f"{qa.upper()}_SCORE: 7\n{qa.upper()}_WEAKNESS: none" for qa in QUALITY_ATTRIBUTES
                )
            return "CANDIDATE: Use caching.\nRATIONALE: Improves performance."

    class _FakeEmbeddingModel:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr("scripts.run_evaluation.load_local_hf_client", lambda: _FakeClient())
    monkeypatch.setattr("scripts.run_evaluation.load_embedding_model", lambda: _FakeEmbeddingModel())

    report = run_evaluation_script(
        records_path=records_path, embeddings_path=embeddings_path,
        n_test_items=2, min_length=100, k=1, max_rounds=1, tactic_budget=2, max_repair_iterations=1,
    )

    assert report.n_items == 2
    assert len(report.system_reports) == 4
```

- [ ] **Step 6: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/data/test_run_evaluation_script.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_evaluation'`.

- [ ] **Step 7: Write the script**

```python
# scripts/run_evaluation.py
"""Real evaluation harness run: CADENCE + 3 baselines over a held-out
sample of real ADRs, scored against ground truth (spec §5).

Import order: src.retrieval.embeddings (sentence_transformers) is imported
before anything that triggers torch via the local-HF client, per the
sentence_transformers-before-torch rule in PROGRESS.md.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.retrieval.embeddings import embed_texts, load_embedding_model  # noqa: F401  (import before torch)
import numpy as np

from src.deliberation.knowledge_graph import QUALITY_ATTRIBUTES, TACTICS, build_knowledge_graph
from src.deliberation.llm_client import load_local_hf_client
from src.evaluation.harness import EvaluationReport, run_evaluation
from src.evaluation.held_out_set import load_verified_records, sample_test_set
from src.retrieval.retriever import Retriever

PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
RECORDS_PATH = PROCESSED_DIR / "adr_records.jsonl"
EMBEDDINGS_PATH = PROCESSED_DIR / "adr_embeddings.npy"


def run_evaluation_script(
    records_path: Path,
    embeddings_path: Path,
    n_test_items: int = 3,
    min_length: int = 300,
    k: int = 2,
    max_rounds: int = 1,
    tactic_budget: int = 4,
    max_repair_iterations: int = 2,
) -> EvaluationReport:
    verified = load_verified_records(records_path)
    test_records = sample_test_set(verified, n=n_test_items, min_length=min_length)

    all_records = load_verified_records(records_path)  # rebuilt below against full record set for retrieval
    import json
    all_records = []
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            from src.retrieval.records import ADRRecord
            all_records.append(ADRRecord(**json.loads(line)))
    embeddings = np.load(embeddings_path)
    embedding_model = load_embedding_model()
    retriever = Retriever(all_records, embeddings, embedding_model)

    client = load_local_hf_client()
    graph = build_knowledge_graph(TACTICS)

    return run_evaluation(
        test_records=test_records, retriever=retriever, client=client, graph=graph,
        tactics=TACTICS, quality_attributes=QUALITY_ATTRIBUTES,
        k=k, max_rounds=max_rounds, tactic_budget=tactic_budget,
        max_repair_iterations=max_repair_iterations,
    )


if __name__ == "__main__":
    report = run_evaluation_script(RECORDS_PATH, EMBEDDINGS_PATH)
    print(f"=== Evaluation report ({report.n_items} held-out items) ===")
    for sr in report.system_reports:
        print(f"\n{sr.system_name}:")
        print(f"  BERTScore F1: {sr.average_scores.bertscore_f1:.3f}")
        print(f"  BLEU:         {sr.average_scores.bleu:.2f}")
        print(f"  ROUGE-1 F1:   {sr.average_scores.rouge1_f:.3f}")
        print(f"  METEOR:       {sr.average_scores.meteor:.3f}")
        if sr.feasibility_rate is not None:
            print(f"  Constraint-satisfaction rate: {sr.feasibility_rate:.2f}")
            print(f"  Avg repair iterations:        {sr.average_repair_iterations:.2f}")
```

- [ ] **Step 8: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/data/test_run_evaluation_script.py -v`
Expected: PASS (1 passed).

- [ ] **Step 9: Run the full suite, then run the real evaluation**

Run: `conda run -n py313 pytest -q` — expect all tests (old + new) passing.

Run: `conda run -n py313 python scripts/run_evaluation.py` (or the env's
`python.exe` directly if `conda run` mis-wraps the invocation) — the real
evaluation over 3 held-out ADRs × 4 systems (12 total generations, several
involving many real LLM calls each for the deliberation-based systems —
expect several minutes of wall-clock time). Given this session's history
of format-deviation crashes on real model output, **if this crashes with a
parse error the retry logic didn't absorb, that's real signal**: either
bump `MAX_SYNTHESIS_ATTEMPTS`/`MAX_CRITIQUE_ATTEMPTS` (in
`src/deliberation/orchestrator.py`/`src/critique/finalize.py`), or note
the new variant and decide whether it needs its own fix — don't just
retry the whole script and hope. Expect a printed comparison table; sanity
check that `cadence_full`'s `BERTScore F1` is plausible (roughly
comparable to or higher than the baselines' — if it's dramatically lower,
investigate whether `generated_text`'s decision+rationale format is
unfairly penalized by length/style mismatch against full ADR prose, which
is a known, already-documented granularity mismatch — see Self-Review
Notes — not necessarily a code bug).

- [ ] **Step 10: Commit**

```bash
git add src/evaluation/harness.py scripts/run_evaluation.py tests/evaluation/test_harness.py tests/data/test_run_evaluation_script.py
git commit -m "feat: add evaluation harness tying baselines, CADENCE, and metrics together"
git push
```

---

## Self-Review Notes

- **Spec coverage:** implements spec §5's baselines (a)-(d), all four
  standard metrics, and the two solver-specific metrics. Does not
  implement a full statistical-significance test suite (e.g. paired
  bootstrap significance between systems) or the "Human/LLM-judge
  evaluation" spec §5 also mentions — both are follow-on work once a
  larger real evaluation run (beyond this plan's N=3 verification sample)
  produces enough data to make significance testing meaningful.
- **Known granularity mismatch, accepted rather than engineered around:**
  `multiagent_no_solver`/`cadence_full`'s `generated_text` is a short
  decision + one-paragraph rationale, while a real ADR's `raw_text` can be
  much longer prose (see the retrieval-indexing plan's corpus stats:
  median 4 files/repo, individual ADR bodies vary widely in length).
  BERTScore is comparatively robust to this (semantic, not n-gram based);
  BLEU/ROUGE-1 are known to penalize length mismatch more harshly. This is
  flagged explicitly rather than silently accepted — the results section
  should discuss it, and a follow-on plan could ask each system to
  generate full Nygard/MADR-style ADR prose (Status/Context/Decision/
  Consequences) instead of just a short decision+rationale, for a fairer
  length-matched comparison, if the pilot numbers here suggest it matters.
- **Why baselines don't get their own `max_rounds`/`tactic_budget`
  tuning:** every system in this plan uses the *same* `k`/`max_rounds`/
  `tactic_budget`/`max_repair_iterations` where applicable, so differences
  in scores are attributable to the systems' architectural differences
  (spec's actual comparison question), not to giving one system more
  generation budget than another.
- **Why N=3 for the verification run, not a full evaluation:** this
  plan's job is to prove the harness is correct and produces sane,
  internally-consistent numbers — not to produce the manuscript's final
  reported results. Scaling N up (and likely `max_rounds` back to 2,
  matching the other stage demos) is a deliberate, separate decision once
  the harness itself is trusted, given the real wall-clock cost of many
  LLM calls observed throughout this session.
- **Why `compute_corpus_metrics` batches BERTScore across the whole
  system's outputs instead of per-item:** `bert_score.score` accepts
  lists and is meaningfully faster batched (one model forward pass over
  all pairs) than calling it N separate times, per its own documented
  usage pattern.
- **Placeholder scan:** no TBD/TODO; all code blocks are complete and
  runnable.
- **Type/interface consistency:** `SystemOutput` fields match between
  `systems.py` (definition, all four runners) and `harness.py`
  (consumption via `o.generated_text`/`o.is_feasible`/`o.repair_iterations`).
  `MetricScores` fields match between `metrics.py` and `harness.py`'s
  `SystemReport.average_scores`. `run_evaluation`'s signature matches
  between Task 4's definition and `run_evaluation_script`'s call site.
