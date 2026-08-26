# Progress Log

Read this first on any machine after `git pull` — it says what's done, what's
next, and what you need to regenerate locally (some things are gitignored on
purpose and won't come through the clone).

## Project

New algorithm manuscript for **IEEE Transactions on Services Computing**,
Special Issue: *Large Language Models in Service-Oriented Ecosystems Design:
Advances and Applications*. Submission deadline **31 October 2026**.

Algorithm: **CADENCE** — retrieval-augmented case-based reasoning +
knowledge-graph-grounded multi-agent deliberation (quality-attribute advocate
agents) + constraint-solver-verified feasibility with an LLM repair loop + a
separate self-critique finalization stage, for LLM-assisted architectural
decision-making in service-oriented architectures.

Full design: `docs/superpowers/specs/2026-08-25-cadence-adr-algorithm-design.md`

## Status: constraint-solver + repair-loop plan complete (CADENCE Stages 1+2+3 done)

`docs/superpowers/plans/2026-08-27-constraint-solver-repair-loop.md` — all
4 tasks done. Resolved spec §9's flagged highest-risk item (the
trade-off-to-MaxSAT encoding). Verified Z3 API shapes empirically before
coding (two-phase `Solver`+unsat-core / `Optimize`+weighted-selection
split; `priority="lex"` for true coverage-before-trade-off-cost priority,
not a magnitude-biased weighted sum). Two independent code reviews on this
plan found and fixed real bugs: (1) the original weighted-sum objective
could let large caller-supplied trade-off weights outvote covering a
required attribute entirely — fixed with lexicographic priority; (2) an
LLM error or unparseable repair response crashed the whole run and
discarded all tracked progress — now caught and treated as a failed
attempt, retried next iteration. The real Stage 1→2→3 end-to-end demo
(`scripts/run_solver_demo.py`) surfaced two more real bugs beyond what
either review caught, both fixed and regression-tested: repair prompts
never named concrete tactic options for uncovered attributes (so a small
model's revisions often weren't extractable even when reasonable in plain
English), and the CANDIDATE/RATIONALE label parser was too strict for a
real model's `"**Candidate Decision:**"` phrasing (crashed
`DeliberationOrchestrator.deliberate` itself, not just the solver). After
both fixes, the real run correctly makes measurable repair progress (1→2
of 5 quality attributes covered across repair iterations) and correctly
reports genuine infeasibility for its deliberately-tight demo parameters
(5 required attributes, budget 4 — mathematically impossible to fully
satisfy in this catalog, since each tactic supports exactly one
attribute). Everything is merged to `main`.

Before this, `docs/superpowers/plans/2026-08-26-multi-agent-deliberation.md`
(CADENCE Stage 2), `docs/superpowers/plans/2026-08-26-adr-retrieval-indexing.md`
(CADENCE Stage 1), and `docs/superpowers/plans/2026-08-25-adr-corpus-acquisition.md`
completed — see git history for those plans' details; not repeated here
since they're superseded by what follows.

**What exists in the repo now (CADENCE Stages 1+2+3, spec §3):**
- `src/retrieval/records.py` — `ADRRecord` schema + `parse_corpus`/`parse_adr_folder`: lenient parser for the real corpus's inconsistent ADR filenames/headings.
- `src/retrieval/embeddings.py` — `embed_texts`/`load_embedding_model` (sentence-transformers `all-MiniLM-L6-v2`, model injected for testability).
- `src/retrieval/index.py` — `VectorIndex` (sklearn `NearestNeighbors`, cosine).
- `src/retrieval/retriever.py` — `Retriever.retrieve(query_text, k) -> list[ADRRecord]` — **the CADENCE Stage 1 entry point**.
- `scripts/build_adr_dataset.py` — parses the real corpus into `data/processed/adr_records.jsonl` (committed, 6,173 records, ~19.5 MB).
- `scripts/build_retrieval_index.py` — embeds the processed dataset for real and saves `data/processed/adr_embeddings.npy` (committed, ~9.5 MB, `all-MiniLM-L6-v2` embeddings for all 6,173 records).
- `src/deliberation/knowledge_graph.py` — `TACTICS` (25 hand-curated architectural tactics) + `build_knowledge_graph`/`supporting_tactics_for`/`trade_offs_for_tactic` (`networkx.DiGraph`).
- `src/deliberation/llm_client.py` — `GeminiClient` (real API shape verified against installed SDK source, untested for real — no `GEMINI_API_KEY` configured), `LocalHFClient`/`load_local_hf_client` (real, verified reliable — see Environment notes on the `pipeline()` segfault it works around).
- `src/deliberation/agent.py` — `QualityAttributeAgent.propose`/`.critique` — **the CADENCE Stage 2 per-agent entry point**.
- `src/deliberation/orchestrator.py` — `DeliberationOrchestrator.deliberate(context, precedents) -> DeliberationResult` — **the CADENCE Stage 2 entry point** the next plan (constraint solver) should import directly, alongside `Retriever` from Stage 1.
- `scripts/run_deliberation_demo.py` — real end-to-end Stage 1 → Stage 2 demo script (not committing any output artifact — this one's outputs are ephemeral transcripts, not data to persist).
- `src/solver/tactic_extraction.py` — `extract_mentioned_tactics(text, tactics)`: fuzzy word-stem matching, bridges free-text deliberation output to the solver's boolean variables.
- `src/solver/feasibility.py` — `check_feasibility(...) -> FeasibilityResult`: two-phase Z3 check (strict feasibility + unsat core via `Solver`, best-effort lexicographically-optimized selection via `Optimize`). `uncovered_quality_attributes` is the sound/complete signal of what needs fixing; `unsat_core_quality_attributes` is supplementary (can be an incomplete explanation — inherent to unsat-core semantics).
- `src/solver/synthesis_format.py` — `parse_candidate_rationale`/`CandidateRationaleParseError` (deliberately duplicated from `orchestrator.py`'s private parser, not imported — see the solver plan's Global Constraints).
- `src/solver/repair.py` — `run_repair_loop(...) -> VerifiedDecision` — **the CADENCE Stage 3 entry point**: check feasibility, ask an LLM to revise within budget on failure (naming concrete tactic options per uncovered attribute), track the best attempt across iterations, degrade gracefully with an explicit caveat after exhausting `max_repair_iterations`. Never crashes on a bad/failed repair attempt — treats it as "try again next iteration."
- `scripts/run_solver_demo.py` — real end-to-end Stage 1 → Stage 2 → Stage 3 demo script.
- `data/corpus_inventory.json`, `data/README.md` — corpus provenance + `processed/` schema docs.
- 88 tests passing (`conda run -n py313 pytest -q`).

**What is NOT in the repo (gitignored, regenerate locally if ever needed —
normal work should not need to):**
- `data/raw/`, `data/extracted/` (~11 GB) — fully consumed by `scripts/build_adr_dataset.py` and deleted. Regenerate with `conda run -n py313 python scripts/fetch_adr_corpus.py` then `python scripts/build_adr_dataset.py` then `python scripts/build_retrieval_index.py` if `data/processed/` is ever lost.
- `.env` — not committed. Create fresh per machine: `GEMINI_API_KEY=...`, `OPENROUTER_API_KEY=...` (primary LLM backbone `gemini-3.5-flash-lite`; local open-weight model via CUDA is the reproducible secondary backbone; OpenRouter optional/tertiary — spec §6).

## Environment notes (apply on every machine)

- Python 3.13 via conda env `py313`, GPU available (`torch` CUDA confirmed working — NVIDIA RTX 5000 Ada Generation Laptop GPU on the dev machine).
- **Import order matters:** `import sentence_transformers` before `import torch` in any module that uses both — the reverse order segfaults (exit 139) on Windows in this env. `src/retrieval/embeddings.py` already enforces this; preserve it in any new module that imports both.
- `z3-solver` is installed and verified reliable (no flakiness observed, unlike the torch/transformers native-import issues below). Two API details worth remembering: `z3.PbLe([], n)` raises `ValueError` (guard for empty variable lists), and multi-objective `Optimize` needs `opt.set(priority="lex")` + distinct `add_soft(..., id=...)` groups to get true lexicographic priority — a single weighted-sum objective lets large weights in one group silently outvote another regardless of intent (see `src/solver/feasibility.py`'s docstring for the concrete bug this caused).
- `conda run -n py313 python <script>.py` occasionally mis-wraps arguments (silently drops flags/exits 127 with no useful output) — if that happens, retry once, or fall back to the env's python.exe directly (`<conda envs dir>\py313\python.exe <script>.py`).
- `pip install -U sentence-transformers` (and friends) is safe to do again if a future dependency bump seems to reintroduce the segfault — re-verify with a plain `python -c "import sentence_transformers; import torch"` smoke test, not a guess.
- **Occasional transient crashes during heavy native imports (torch/transformers/sentence_transformers) are a known nuisance on this machine.** A few times this session, `pytest -q` (or a script) crashed mid-import with a native traceback ending inside `sentence_transformers`/`transformers` internals, then passed cleanly on an immediate retry with no code changes. Before spending time debugging an import-time crash, retry once or twice first — but don't use this as an excuse to dismiss a *reproducible* crash (see the `pipeline()` finding below, which reproduced repeatedly and turned out to be real).
- **`transformers.pipeline("text-generation", ...)` is unreliable on this machine — do not use it.** It segfaults (native crash, exit 139) unpredictably at inconsistent points across repeated attempts: sometimes during `from transformers import pipeline` itself, sometimes during model load, sometimes during the generate call. Not GPU-specific (reproduced with `device="cpu"` too), not token-count-specific, not sampling-specific — root cause not identified after significant isolation effort (ruled out: GPU/driver state, VRAM, system RAM, CUDA vs CPU, foreground vs background process, token count, greedy vs sampled decoding). The fix: call `AutoTokenizer.from_pretrained` + `AutoModelForCausalLM.from_pretrained` + `.generate()` directly instead of the `pipeline()` wrapper — verified reliable across many repeated runs, including multiple sequential real generate calls against one loaded model (CPU and CUDA, greedy and sampled). `src/deliberation/llm_client.py`'s `load_local_hf_client` already does this. If you need any other HF `pipeline(...)` task type, verify it in isolation first rather than assuming it works.
- **Any prompt asking a small local model for structured "LABEL: value" output needs a tolerant parser and a real-output test, not just a hand-written happy-path example.** The `CANDIDATE:`/`RATIONALE:` format (used by both `src/deliberation/orchestrator.py` and `src/solver/repair.py`) crashed a real end-to-end run because the model wrote `"**Candidate Decision:**"` (an extra descriptor word) instead of `"**Candidate:**"` — a strict regex only tolerating markdown/whitespace before the colon missed it. Fixed by tolerating a short run of extra words between the label and the colon. If Stage 4 (self-critique) or anything later asks the model for its own structured format, write the parser tolerant from the start (bounded extra-words allowance, case-insensitive, markdown-tolerant) and validate it against at least one real (not hand-crafted) model response before trusting it in an unattended run.

## Real corpus schema (for any future plan reading `data/extracted/` or `data/processed/`)

The corpus is the **full "Context Matters" replication package**, not a bare
ADR dump — Zenodo DOI [10.5281/zenodo.18370195](https://doi.org/10.5281/zenodo.18370195),
derived from Buchgeher et al.'s MSR study (IEEE Access, DOI [10.1109/ACCESS.2023.3287654](https://doi.org/10.1109/ACCESS.2023.3287654)).

- `Data/ADRs/{repo}_{adr_folder}/*.md` — the retrieval corpus: 883 folders, 6,173 `.md` files. Median 4 files/repo, mean 6.76, max 129 (confirms the spec's sparsity note). Filenames are inconsistent across repos (`0001-slug.md`, `0001 - slug.md`, `adr-0001 slug.md`, `ADR-1_slug.md`, 3-digit, 0-indexed, and some files with **no leading number at all**) — `src/retrieval/records.py` handles this leniently; don't assume a strict pattern in future code either.
- `Data/dataset_index.json` (954 entries) — extraction-confidence status per folder: `Verified`:750, `Doubt (name sequence)`:96, `Repo Inaccessible`:40, `Doubt (no repo dir)`:33, `Doubt (missing file)`:30, `Doubt (file contents)`:5. Carried through as `ADRRecord.extraction_status` — filter to `Verified`-only later if a cleaner held-out evaluation split is needed.
- ADR body format is usually Nygard/MADR (`# Title`, `Date:`, `## Status`, `## Context`, `## Decision`, `## Consequences`) but not universally, so `ADRRecord` deliberately only extracts `title` + full `raw_text`, no general section-splitter — see the retrieval-indexing plan's self-review notes for why, and regex out a specific section from `raw_text` only if/when a future stage actually needs it.
- **Not yet used, worth remembering for the evaluation-harness plan:** `Experiments/` and `Results/` in the same package are the Context Matters paper's *own* generation+evaluation pipeline — `Results/{Baseline,All_N,First_K,Last_K,RAG_Based}/{Model}/Dataset/{repo}.json` (candidate ADRs) and `.../Evaluations/{repo}.json` (`rouge1_f`, `rouge2_f`, `rougeL_f`, `bleu_avg`, `meteor`, `bert_f1/precision/recall` — exactly spec §5's metric set) for 4 models × 5 strategies. **Check whether the "Context-Matters-style retrieval-only" baseline (spec §5(b)) can reuse these existing generations/scores directly instead of re-running that pipeline** — this data no longer exists on disk (it was in the now-deleted `data/extracted/`), so re-fetch the corpus first if this is pursued.

## Next step

Write and execute the **self-critique / finalization** implementation plan
(spec §3 Stage 4): a separate LLM pass scores the converged, solver-
verified `VerifiedDecision` (this session's output) against explicit
per-attribute utility functions, flags residual weaknesses, and emits the
final ADR with full provenance (precedents used from Stage 1, agent
positions from Stage 2's transcript, constraints checked and repair
iterations from Stage 3). This plan doesn't exist yet. Reuse the
established LLM-client DI pattern and the tolerant-structured-output-
parsing lesson above.

After that: evaluation harness (baselines: Dhar et al. ICSA'24-style,
Context-Matters-style retrieval-only — possibly reusable from the corpus,
see above — MAAD-style no-solver ablation; metrics: BERTScore/BLEU/ROUGE-1/
METEOR + novel constraint-satisfaction-rate + repair-convergence, both of
which Stage 3's `VerifiedDecision`/`FeasibilityResult` already expose per-
run) → manuscript (IEEEtran template already in repo root, 6 sections,
≤14 pages).

## Working conventions established so far

- Python 3.13 via conda env `py313` for everything.
- Work happens directly on `main` (no feature branches) — user's explicit choice.
- Push after every commit, no Claude identity in commits (global user convention).
- Implementation plans executed task-by-task: write failing tests, implement, verify passing, sanity-check against real data when the real data is small enough to (never in automated unit tests — those stay fast and network/data-free), get an independent code review before committing non-trivial logic, commit, push.
