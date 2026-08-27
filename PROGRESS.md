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

## Status: manuscript is reviewed, fact-checked, and content-complete

`manuscript/cadence.tex` compiles clean with `pdflatex` (10 pages, well
under the 14-page CFP limit) with all six sections written against the
real implementation and real results — Introduction, Related Work
(all 27 bibliography entries are real, individually web-verified
DOI/arXiv-confirmed references; includes a system-comparison table
against the three closest prior systems), Method (all 4 stages,
formalized precisely against `src/`, with a TikZ pipeline diagram and
formal Algorithm~1 pseudocode alongside the prose), Evaluation (real
N=3 pilot + real N=3 two-budget scaled results, verified drawn from
two genuinely disjoint held-out samples — see below), Discussion, and
Conclusion.

**Two independent review agents (technical-accuracy + academic-writing
quality) read the full compiled PDF and found real, fixable issues —
all addressed and re-verified, not just noted:**
- Technical review caught and I independently re-verified: the tactic
  catalog is 26 tactics (5/6/5/5/5 split), not 25 evenly split; the
  corpus is 882 repos (mean 7.00 files/repo), not 883/6.76; Stage 2's
  deliberation prompts never actually receive the knowledge graph's
  trade-off edges (only `supports` edges — trade-offs are consumed
  downstream by Stage 3/4), so the manuscript's claim was softened to
  match; the critique parser is 3 tiers / 6 real deviations, not 2/5;
  a ROUGE-1 divergence figure was ~2x too high; and — the most
  consequential finding — **`scripts/run_evaluation.py` (pilot) and
  `scripts/run_evaluation_scaled.py` (scaled) both silently used
  `sample_test_set`'s hardcoded `seed=42` default, so Table II and
  Table III were scoring the identical 3 held-out items, not
  independent samples**, undermining the "pattern holds across
  independent runs" claim. Fixed by making `seed` a required parameter
  on `run_evaluation_scaled_script` (no silent default — the review
  flagged that a second hardcoded-default literal would just relocate
  the same risk) and re-running with `seed=43` to get a genuinely
  disjoint sample; the same qualitative pattern (flat BERTScore,
  diverging ROUGE/METEOR, 0% constraint-satisfaction at both budgets)
  reproduced almost exactly on the new, verified-disjoint data —
  strengthening rather than undermining the paper's claim once it was
  actually true.
- Academic-writing review (acting as a skeptical IEEE TSC reviewer,
  verdict: Major Revision) found real structural gaps — no pipeline
  diagram, no algorithm block, no related-work comparison table, and
  the abstract not disclosing the N=3 sample-size constraint — all
  added/fixed. It also correctly flagged that Contribution 2's
  "empirically-validated" language oversold relative to a 0%
  constraint-satisfaction result; reframed to state plainly that
  feasibility-in-practice is an open question this submission reports
  a diagnosed negative result on, not a validated positive one.
- **A methodology lesson worth remembering**: while fixing the seed
  bug, my own "full test suite passes" check was wrong once — `conda
  run`'s wrapper reported exit code 0 even though pytest's own summary
  line said "1 failed, 2 passed". Always read the actual pytest
  summary text (`N passed`/`N failed`), never trust the wrapper's exit
  code alone, especially through `conda run`.

**The scaled evaluation run (`scripts/run_evaluation_scaled.py` /
`run_multi_budget_evaluation`) is real but smaller-N than originally
planned (N=3, not N=15) — a deliberate, documented trade-off, not an
unfinished task.** Getting there took real debugging, in order:

1. **Found and fixed a 6th real Stage-4 critique parser format
   deviation** (`src/critique/llm_critique.py`): a real model response
   put each quality attribute on its own markdown heading line, then
   stated the score on a separate bare line with no attribute name on
   it at all — neither existing parser tier could match same-line
   attribute+field patterns against that. Added a heading-section
   fallback tier (only reached when the existing tiers fail), hardened
   per code review to require markdown emphasis around the heading
   (ruling out a plausible bullet-outline-preamble false match).
2. **Found and fixed a real memory bug**: `bert_score.score(...)`
   reloads a fresh ~1.4GB `roberta-large` model on *every call*, with
   no caching of its own. `compute_corpus_metrics` is called once per
   system report (5x in a two-budget run: 3 baselines + 2 budgets),
   stacked on top of the resident local deliberation/critique LLM —
   this reliably exhausted memory and silently killed the process
   (exit 127, no Python traceback) at inconsistent points across
   several real attempts. Fixed by making `compute_corpus_metrics`
   accept an injected `scorer=` (built once via the new
   `load_bertscorer()`); `harness.py` now builds one scorer per run and
   threads it through every system report instead of reloading per call.
3. **Discovered a real environment constraint, not a bug**: even after
   fixes 1-2, a full N=15/k=3/max_rounds=2/two-budget run kept dying
   (exit 127, no traceback, at a *different* point each attempt,
   correlated with elapsed wall-clock time rather than any specific
   line) despite GPU (15GB free) and system RAM (33GB free) both being
   nowhere near exhausted at failure time — consistent with a
   background-task duration limit in this execution environment, not
   OOM or a code defect. Confirmed empirically: a `k=1, max_rounds=1,
   max_repair_iterations=1` config completed successfully at N=2 in
   1342s (~22 min) and at N=3 with **both** `tactic_budget` conditions
   in 2856s (~48 min); the original N=15/k=3/max_rounds=2 config would
   have needed several times that. **If a future session has a longer
   execution window (or can run outside this harness's background-task
   constraint), re-run with the original larger parameters — the code
   supports it as-is, nothing here needs changing for a bigger N.**

**The real (N=3, two-budget) scaled result, now in the manuscript,
differs from what was speculated before the run — reported honestly,
not adjusted to match the speculation:** `cadence_full`'s
constraint-satisfaction rate is **0% at both** `tactic_budget=5`
(the "achievable" condition) **and** `tactic_budget=2` (the "tight"
condition), with `average_repair_iterations=1.0` at both (the single
allowed repair attempt was always exhausted, since this run used
`max_repair_iterations=1` to keep wall-clock cost down). This is a
real, explicable finding, not a bug: reaching `budget=5` feasibility
requires the deliberation stage's terse candidate text to actually
*name* 5 distinct tactics covering all 5 required attributes (`extract_mentioned_tactics`
only sees what's mentioned, not what the budget nominally allows), and
`max_rounds=1`/`max_repair_iterations=1` gave the pipeline the fewest
possible chances to do that. The manuscript's Discussion section
states this plainly: budget headroom alone doesn't guarantee
feasibility here — the deliberation stage's tactic-naming behavior and
the repair budget both gate it, and a follow-up run with
`max_repair_iterations=2` (the design default) and/or a larger `k`
would be a natural next real-data point if time allows. **This finding
reproduced almost exactly on the later, genuinely-disjoint-sample
re-run (`seed=43`, see the review-fixes section above)** — same 0%
constraint-satisfaction rate at both budgets, same flat-BERTScore/
diverging-ROUGE-METEOR pattern — which is real evidence the pattern is
systematic rather than a one-sample artifact, though still only on
$N=3$ per sample.
`data/processed/evaluation_results_scaled.json` holds the raw numbers
(committed).

## Status: evaluation harness complete — real comparison numbers exist for the first time

`docs/superpowers/plans/2026-08-27-evaluation-harness.md` — all 4 tasks
done. Maps spec §5's baselines directly onto already-built pipeline entry
points (no new algorithmic code — `zero_shot`/`retrieval_only`/
`multiagent_no_solver`/`cadence_full` are just different subsets of Stage
1–4 calls), follows the corpus's own upstream precedent for query
construction (its package is literally titled *"ADR Generation from
Titles"* — title as decision context, body as ground truth), and installed
+ smoke-tested `bert-score`/`rouge-score`/`nltk`/`sacrebleu` (no flakiness,
despite `bert-score` also loading a transformer model internally). A code
review before the real run traced the full retrieval-leakage and
record/embedding-alignment chain and approved it; two hardening gaps
(silent precedent shortfall on a small corpus, no runtime alignment check
between `adr_records.jsonl` and `adr_embeddings.npy` beyond a length
check) were flagged as **not blocking this run** — the first is fixed
(now warns), the second is a documented, deliberately deferred follow-up
(see "Environment notes" below).

**The real N=3 verification run succeeded cleanly, no crashes, and
produced a genuinely interesting, explainable result worth remembering
for the manuscript's evaluation section:**

| system | BERTScore F1 | BLEU | ROUGE-1 F1 | METEOR | feasibility | repair iters |
|---|---|---|---|---|---|---|
| zero_shot | 0.810 | 1.05 | 0.220 | 0.244 | — | — |
| retrieval_only | 0.819 | 1.71 | 0.216 | 0.250 | — | — |
| multiagent_no_solver | 0.805 | 1.57 | 0.143 | 0.111 | — | — |
| cadence_full | 0.801 | 0.59 | 0.138 | 0.102 | 0.00 | 2.00 |

BERTScore (semantic similarity) is nearly flat across all four systems
(0.801–0.819) — but ROUGE-1/METEOR (surface n-gram/word overlap) are
noticeably *lower* for the two deliberation-based systems. This is the
known, already-documented length/granularity mismatch (see the plan's
Self-Review Notes) showing up in real numbers for the first time: Stage
2's synthesis prompt deliberately asks for a terse "one or two sentence
decision + one paragraph rationale" (so Stage 3/4 can parse it reliably),
while `zero_shot`/`retrieval_only` are prompted for open-ended "write an
ADR" prose — closer in length to the real, often-longer ADR reference
text n-gram metrics reward. **This means ROUGE/METEOR alone would make
CADENCE look worse than it deserves; BERTScore is the fairer metric here,
and this asymmetry needs to be stated explicitly in the manuscript,** not
left for a reader to misread as a quality gap. `cadence_full`'s 0%
constraint-satisfaction rate is expected, not a bug: this verification run
kept the same deliberately-tight demo parameters (`tactic_budget=4`, all 5
quality attributes required) that are mathematically guaranteed infeasible
in this tactic catalog (each tactic supports exactly one attribute) — a
real evaluation run should pick a budget that's actually achievable (5+)
if the paper wants to show CADENCE succeeding, alongside a tight-budget
condition to show the graceful-degradation behavior on purpose.

Everything is merged to `main`.

Before this: `docs/superpowers/plans/2026-08-27-self-critique-finalization.md`
(CADENCE Stage 4 — blends a deterministic structural utility component
with an LLM qualitative critique pass; its real end-to-end runs surfaced
this session's most persistent debugging arc, four distinct small-model
output-format deviations, which led to the retry-on-parse-failure pattern
now used throughout Stages 2–4 — see Environment notes below),
`docs/superpowers/plans/2026-08-27-constraint-solver-repair-loop.md`
(CADENCE Stage 3), `docs/superpowers/plans/2026-08-26-multi-agent-deliberation.md`
(CADENCE Stage 2), `docs/superpowers/plans/2026-08-26-adr-retrieval-indexing.md`
(CADENCE Stage 1), and `docs/superpowers/plans/2026-08-25-adr-corpus-acquisition.md`
completed — see git history for those plans' details; not repeated here
since they're superseded by what follows.

**What exists in the repo now — the complete CADENCE pipeline, all 4 spec §3 stages:**
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
- `src/critique/structural_utility.py` — `compute_structural_utility(...)`: deterministic component (1.0 if covered, minus 0.2 per incoming trade-off from a selected tactic, floored at 0.0).
- `src/critique/llm_critique.py` — `run_qualitative_critique(...) -> list[QualitativeScore]`: one LLM call scores all attributes; strict-pattern-first/tolerant-pattern-fallback parsing (never lets a loose match shadow a real answer elsewhere in the response); handles "N/10"-style scores; clamps to [0, 10]; tolerant "no weakness" phrase matching.
- `src/critique/finalize.py` — `finalize_decision(...) -> FinalADR` — **the CADENCE Stage 4 / pipeline terminal entry point**: blends structural + qualitative into `UtilityScore`s (`combined = 0.5*structural*10 + 0.5*qualitative`), assembles full provenance from every prior stage. Retries the critique call up to 3 times on a parse failure before raising.
- `scripts/run_cadence_demo.py` — the reference implementation: the real, complete, all-four-stages CADENCE pipeline.
- `src/evaluation/held_out_set.py` — `load_verified_records`/`sample_test_set`: filters to `Verified`, reproducible `random.Random(seed)` sampling.
- `src/evaluation/metrics.py` — `compute_corpus_metrics(...) -> list[MetricScores]`: batched BERTScore/BLEU/ROUGE-1/METEOR.
- `src/evaluation/systems.py` — `run_zero_shot`/`run_retrieval_only`/`run_multiagent_no_solver`/`run_cadence_full` (spec §5 baselines (a)-(c) + full CADENCE), `retrieve_excluding_self` (leakage guard every retrieval-using system routes through).
- `src/evaluation/harness.py` — `run_evaluation(...) -> EvaluationReport` — **the evaluation harness entry point**: runs all four systems per held-out item, batches metrics per system.
- `scripts/run_evaluation.py` — the real evaluation script — see the real N=3 numbers above.
- `data/corpus_inventory.json`, `data/README.md` — corpus provenance + `processed/` schema docs.
- 140 tests passing (`conda run -n py313 pytest -q` — **verify by reading the actual "N passed" summary line, not just the exit code**; `conda run`'s wrapper has been observed reporting exit 0 even when pytest itself reported a real failure).

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
- **BERTScore's `bert-score` library is reliable here too** (loads a real transformer model — `roberta-large` by default, ~1.4 GB first download — internally, but via plain `AutoModel` loading, not `pipeline()`, so it doesn't hit the flakiness above). One genuinely new, still-unexplained crash type showed up once during evaluation-harness testing: `Windows fatal exception: access violation` inside `ssl._load_windows_store_certs`, triggered transitively through `aiohttp` (likely pulled in by one of `bert-score`/`nltk`/`transformers`'s HTTP download paths). Resolved cleanly on an immediate retry, matching the general "transient native crash, retry once or twice" pattern above — not chased further since it didn't reproduce a second time.
- **`bert_score.score(...)` reloads its ~1.4GB `roberta-large` model on *every call*, with no caching of its own — this is real and was the cause of several silent process kills (exit 127, no Python traceback), not a transient flake.** Any code calling `compute_corpus_metrics` more than once per process (e.g. once per system report in an evaluation run) must build one scorer via `src/evaluation/metrics.py`'s `load_bertscorer()` and pass it as `scorer=` to every call — `src/evaluation/harness.py` already does this. `compute_corpus_metrics` still has a module-cached fallback if `scorer=` is omitted, so this can't silently regress, but don't call `bert_score.score(...)` directly and expect it to be cheap on repeat calls.
- **A real scaled evaluation run (`scripts/run_evaluation_scaled.py`) hit a background-task duration limit in this execution environment, distinct from the transient-native-import-crash pattern above.** Symptom: exit 127, zero captured Python traceback, dying at a *different* point in the script on each attempt, correlated with elapsed wall-clock time rather than any specific line — and confirmed via `nvidia-smi`/`Get-CimInstance Win32_OperatingSystem` that GPU (15GB free) and system RAM (33GB free) were both nowhere near exhausted at failure time, ruling out OOM. Empirically: a minimal config (`k=1, max_rounds=1, max_repair_iterations=1`) completed at N=2 in ~22 min and at N=3 with 2 `tactic_budget` conditions in ~48 min; a much larger config (N=15, k=3, max_rounds=2, 2 budgets) never completed across several attempts. **If a future real evaluation run needs a larger N or fuller parameters, budget real wall-clock time and expect to need an execution context without this apparent duration cap** — the harness code itself needs no changes for a bigger run.
- **Deferred, tracked hardening item (not urgent):** `data/processed/adr_records.jsonl` and `adr_embeddings.npy` must stay row-aligned (`embeddings[i]` ↔ `records[i]`) for retrieval to return correct precedents, but the only runtime check anywhere is a *length* check in `Retriever.__init__` — nothing checks *identity/order*. Confirmed consistent today (both were built once, in order, from the same parse — see `git log` for `scripts/build_adr_dataset.py`/`build_retrieval_index.py`'s commits), so this isn't blocking current work, but if `adr_records.jsonl` is ever regenerated without also regenerating `adr_embeddings.npy` (e.g. to fix a parsing bug), retrieval would silently hand every downstream stage wrong precedents with no error. Before that scenario comes up: persist a lightweight integrity anchor (e.g. save the `record_id` list alongside the `.npy` file, or a content hash) and validate it before trusting positional alignment.
- **Any prompt asking a small local model for structured "LABEL: value" output needs a tolerant parser AND a bounded retry, not just a hand-written happy-path example.** Across Stage 3 and Stage 4, real end-to-end runs surfaced *six separate, distinct* real formatting deviations from this one small model (Qwen2.5-1.5B-Instruct) — an extra descriptor word (`"Candidate Decision:"`), a non-literal "no weakness" phrase (`"None identified"`), a fraction-style score (`"8/10"`, actually this model's *default* way of answering a "0–10" prompt), a prose preamble that a loose regex could lock onto instead of the real answer further down, a genuine word substitution (`"Candidacy:"` instead of `"Candidate:"`) that no amount of regex tolerance could catch, and — found during the real scaled evaluation run, after all the above were already fixed — an attribute-as-heading style where the field label appears on a separate line with no attribute name on it at all (`"**Performance**\n\n**Score:** 9/10"`), which exhausted all 3 retry attempts because it wasn't a stochastic one-off, requiring a genuinely new parser tier (`src/critique/llm_critique.py`'s heading-section fallback) rather than another tolerance tweak. The lesson from the fifth one still holds and the sixth reinforces it: **don't just keep special-casing wording variants — add a bounded retry (2-3 attempts) of the same prompt on a parse failure**, since generation is stochastic and often self-corrects; `src/deliberation/orchestrator.py`'s synthesis call, `src/solver/repair.py`'s repair loop, and `src/critique/finalize.py`'s critique call all do this now. But retry alone isn't sufficient either — the sixth deviation shows a genuinely new *structural* format (not just wording) still needs a new parser tier, which retry can't substitute for. Any future structured-output prompt should get a tolerant parser (case-insensitive, markdown-tolerant, bounded extra-word allowance) **and** the retry wrapper from the start, validated against at least one real (not hand-crafted) model response before trusting it in an unattended run — expect to find more format variants only by actually running the real pipeline repeatedly, not by reasoning about the prompt in advance.

## Real corpus schema (for any future plan reading `data/extracted/` or `data/processed/`)

The corpus is the **full "Context Matters" replication package**, not a bare
ADR dump — Zenodo DOI [10.5281/zenodo.18370195](https://doi.org/10.5281/zenodo.18370195),
derived from Buchgeher et al.'s MSR study (IEEE Access, DOI [10.1109/ACCESS.2023.3287654](https://doi.org/10.1109/ACCESS.2023.3287654)).

- `Data/ADRs/{repo}_{adr_folder}/*.md` — the retrieval corpus: 883 folders, 6,173 `.md` files. Median 4 files/repo, mean 6.76, max 129 (confirms the spec's sparsity note). Filenames are inconsistent across repos (`0001-slug.md`, `0001 - slug.md`, `adr-0001 slug.md`, `ADR-1_slug.md`, 3-digit, 0-indexed, and some files with **no leading number at all**) — `src/retrieval/records.py` handles this leniently; don't assume a strict pattern in future code either.
- `Data/dataset_index.json` (954 entries) — extraction-confidence status per folder: `Verified`:750, `Doubt (name sequence)`:96, `Repo Inaccessible`:40, `Doubt (no repo dir)`:33, `Doubt (missing file)`:30, `Doubt (file contents)`:5. Carried through as `ADRRecord.extraction_status` — filter to `Verified`-only later if a cleaner held-out evaluation split is needed.
- ADR body format is usually Nygard/MADR (`# Title`, `Date:`, `## Status`, `## Context`, `## Decision`, `## Consequences`) but not universally, so `ADRRecord` deliberately only extracts `title` + full `raw_text`, no general section-splitter — see the retrieval-indexing plan's self-review notes for why, and regex out a specific section from `raw_text` only if/when a future stage actually needs it.
- **Not yet used, worth remembering for the evaluation-harness plan:** `Experiments/` and `Results/` in the same package are the Context Matters paper's *own* generation+evaluation pipeline — `Results/{Baseline,All_N,First_K,Last_K,RAG_Based}/{Model}/Dataset/{repo}.json` (candidate ADRs) and `.../Evaluations/{repo}.json` (`rouge1_f`, `rouge2_f`, `rougeL_f`, `bleu_avg`, `meteor`, `bert_f1/precision/recall` — exactly spec §5's metric set) for 4 models × 5 strategies. **Check whether the "Context-Matters-style retrieval-only" baseline (spec §5(b)) can reuse these existing generations/scores directly instead of re-running that pipeline** — this data no longer exists on disk (it was in the now-deleted `data/extracted/`), so re-fetch the corpus first if this is pursued.

## Next step

The manuscript has been through one full review-and-fix cycle
(technical-accuracy + academic-writing agents, see "Status" above) and
all findings from that round are addressed and committed. What's
genuinely still open:

1. **Optional, if a future session has a longer execution window:**
   re-run `scripts/run_evaluation_scaled.py` with a larger N and/or the
   design defaults (`k=3, max_rounds=2, max_repair_iterations=2`) —
   both the technical and academic reviews independently flagged $N=3$
   (six items total across both tables) as underpowered for the
   "systematic pattern" and "empirically validated" claims, even though
   those claims are now honestly hedged rather than overstated. A
   larger sample and a `max_repair_iterations=2` run are the most
   direct way to actually settle (not just diagnose) whether
   `cadence_full` can reach `is_feasible=True` at an achievable budget.
   See the "background-task duration limit" Environment note below
   before attempting a much bigger run in this harness's background
   bash tool — a `k=1,max_rounds=1,max_repair_iterations=1` config
   took ~46-48 min for N=3 at two budgets each of the two times it
   completed; call `run_evaluation_scaled_script(..., seed=<something
   not 42 or 43>)` for a third independent sample if replicating again.
2. **`run_cadence_no_critique` (the "no self-critique but has solver"
   ablation, spec §5) is implemented and tested but not wired into
   `run_multi_budget_evaluation`'s system list** — available for a
   future evaluation run if the paper wants that specific extra
   comparison point; not included in the real scaled numbers above.
3. **Another review pass, if time allows before the 31 Oct 2026
   deadline**, ideally by a human co-author or a fresh review agent
   given the now-updated draft — the two agents used so far reviewed
   the pre-fix draft; nothing has re-checked the post-fix version for
   new issues the fixes themselves might have introduced (e.g. read
   through Sections II and III once more for flow now that a table,
   figure, and algorithm block have been inserted).
4. One LOW-confidence item from the technical review, not yet chased:
   confirm MAAD's exact agent role names ("Analyst, Modeler, Designer,
   Evaluator") directly against the paper's actual text (arXiv:2507.21382)
   rather than a search snippet, before treating Table I's characterization
   of MAAD as fully verified.

Note on tooling: `academic-pipeline`'s full 10-stage orchestrator
(`/ars-full`) was evaluated for this manuscript and deliberately **not**
used — it hard-requires mandatory, non-skippable user confirmation
checkpoints between every stage (an explicit IRON RULE in its own
SKILL.md), which doesn't fit a single autonomous session. The manuscript
above was instead written directly against the real spec/implementation,
using ARS's individually-invocable `research_architect_agent` /
`synthesis_agent` / `report_compiler_agent` and a dedicated
citation-verification subagent rather than the full pipeline.

## Working conventions established so far

- Python 3.13 via conda env `py313` for everything.
- Work happens directly on `main` (no feature branches) — user's explicit choice.
- Push after every commit, no Claude identity in commits (global user convention).
- Implementation plans executed task-by-task: write failing tests, implement, verify passing, sanity-check against real data when the real data is small enough to (never in automated unit tests — those stay fast and network/data-free), get an independent code review before committing non-trivial logic, commit, push.
