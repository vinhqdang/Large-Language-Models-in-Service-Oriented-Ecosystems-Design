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

- **On a fresh machine (or a `py313` env shared with other projects), don't assume `requirements.txt` is satisfied — check it.** This project's conda env is shared across projects per the user's global convention, so a machine that has never run *this* project's code can be missing packages entirely (`z3-solver`, `bert-score`, `rouge-score`, `nltk`, `sacrebleu`, `google-genai` were all absent on a second machine encountered mid-project) or have versions below this project's floor (`transformers`, `sentence-transformers`, `accelerate` were present but too old, left over from other work). Fix with `conda run -n py313 pip install -U <pkg>>=<floor>` for just the packages actually below floor — this leaves the existing CUDA-enabled `torch` build untouched (pip only touches a dependency if the installed version fails the constraint). Verify with `conda run -n py313 pip list | grep -iE "z3|torch|transformers|bert-score|..."` before assuming, then confirm with a full `pytest -q` run and the actual summary line (see the next bullet).
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

## Status: second manuscript review-and-fix cycle complete (2026-08-29/30)

A fresh pull onto a second machine surfaced a dependency-sync gap first:
this machine's shared `py313` conda env was missing `z3-solver`,
`bert-score`, `rouge-score`, `nltk`, `sacrebleu`, `google-genai` entirely
and had below-`requirements.txt`-floor `transformers`/`sentence-transformers`/
`accelerate` (left over from other projects sharing the same env, per the
project's "one shared conda env" convention) — fixed with
`pip install -U` against the exact floors in `requirements.txt`; torch's
existing CUDA build was untouched. **140 passed** confirmed afterward via
the actual pytest summary line, not the wrapper's exit code.

With the environment fixed, ran a **second** review-and-fix cycle: two
fresh review agents (technical-accuracy, academic-writing) re-read the
already-once-fixed manuscript from scratch, assuming nothing. Both found
real, new issues the first cycle missed:

- **Technical review**: confirmed all 8 previously-fixed claims still
  hold, then found 4 *new* factual errors — Stage 1 passes only precedent
  *titles* to Stage 2's prompts, not bodies (the manuscript said "titles
  and bodies"); Stage 3's tactic extraction matches tactic *name* only,
  not "name and description"; the Abstract/Introduction/Fig.~1
  caption/Conclusion all framed the solver's **unsatisfiable core** as
  the repair loop's driver and said repair goes "back to the agents" —
  the actual driver is the uncovered-quality-attributes signal, and
  repair is a **standalone** LLM call that never re-enters Stage 2
  (Section III-D's own prose already said this correctly; only the
  front-matter/figure/conclusion had drifted from it). Also caught an
  internal inconsistency (a closing sentence claimed retry "explains four
  of the six" format deviations when the same paragraph's own preceding
  logic assigns only one), and flagged that deliberation's `k`/`max_rounds`
  parameters were never disclosed — the **pilot run used `max_rounds=1`,
  meaning zero inter-agent critique ever happened** in Table II's
  multi-agent columns, which materially changes how that table should be
  read.
- **Academic-writing review** (skeptical IEEE TSC reviewer, verdict: Major
  Revision): found that `tactic_budget=2` is **mathematically infeasible
  by construction** — same logic already used to explain the pilot's
  `budget=4` — so of three budget conditions reported across both tables,
  only `budget=5` is a genuine empirical test; the paper had been framing
  "0% at both conditions" as two confirmations of the same finding. Also
  found the Conclusion claimed the solver/critique differences are
  "load-bearing" and that the three baselines "isolate each stage's
  marginal contribution," neither of which the data supports (the
  `multiagent_no_solver` vs `cadence_full` ablation shows no measurable
  metric difference at $N=3$, and the ablation removes Stages 3 and 4
  *together*, not separately). Most seriously: **the Abstract and
  Conclusion — the two sections most readers actually read — omitted the
  0% constraint-satisfaction / non-converging-repair result that the
  paper's middle sections already reported honestly.** Also flagged a
  missing code/data-availability statement despite heavy "reproducible"
  emphasis and 4 unused pages, a dangling cross-reference, an unstated
  BLEU scale, and a citation-attribution inconsistency.

**All of the above are fixed and verified**, not just noted: Algorithm 1
now tracks the best-coverage attempt across iterations (matching what the
prose already claimed) and uses the uncovered-attribute signal, not the
unsat core; Figure 1's back-arrow now shows the repair loop as *internal*
to Stage 3 rather than looping back into Stage 2; the Results/Discussion
now explicitly separate the metric-asymmetry finding (genuinely reproduces
across two independent samples) from the constraint-satisfaction finding
(one real data point at `B=5`, not yet corroborated); the Abstract and
Conclusion now state the negative finding plainly; a "Data and Code
Availability" section with the repo's real GitHub link was added (single-
anonymous review confirmed, so this is safe). `manuscript/cadence.tex`
recompiles clean, **11 pages** (14-page CFP limit). Also fixed a stale
"883 repository folders" figure in `data/README.md` (882 actually contain
`.md` files) found during the same pass.

## Status: codebase cleanup, manuscript density expansion, expanded scaled evaluation (2026-08-30)

User feedback: an 11/14-page manuscript with dense prose and thin
experiments was rejected as unacceptable ("too much text, too little
table, charts, graphs" — see `feedback_manuscript_density` in the
project's Claude memory). Responded with three things, all committed:

1. **Codebase cleanup**: removed the root `Computer_Society_LaTeX_template.zip`
   (already fully extracted into `manuscript/` long ago) and the unused
   IEEE template reference files it left behind (`New_IEEEtran_how-to.*`,
   `bare_jrnl_new_sample4.*`, an unused `fig1.png`) — none were referenced
   by `cadence.tex`.
2. **Manuscript density**: converted prose-only content into tables/figures
   and added previously-omitted real detail (verified against source, not
   assumed) — the full 26-tactic catalog as a table, a small knowledge-graph
   diagram, the six LLM output-format deviations as a table, the corpus's
   full 954-folder extraction-confidence breakdown as a table (caught a
   real scoping bug in a first attempt to recompute this — see the
   commit), and a bar chart of the metric-asymmetry finding. 11 → 12 pages.
3. **Expanded the scaled evaluation itself** (not just formatting): added
   `on_budget_complete` incremental checkpointing to the harness first
   (tested), wired `run_cadence_no_critique` into
   `run_multi_budget_evaluation` (it existed but was never used — now
   properly isolates Stage 4's marginal contribution, closing a gap two
   review cycles had flagged), and re-ran the scaled evaluation with
   `max_repair_iterations=2` (the full design default, up from the
   earlier cost-cut 1). **Real results, now in the manuscript:**
   - Giving the repair loop its full budget did **not** change the 0%
     constraint-satisfaction finding at `B=5` — `average_repair_iterations=2.00`
     confirms both attempts were exhausted, ruling out "not enough repair
     attempts" and narrowing the diagnosis specifically to Stage 2's
     tactic-naming behavior.
   - `cadence_full` beats `cadence_no_critique` on all 4 generation-quality
     metrics at `B=5` and 3 of 4 at `B=2` — the first ablation in this
     whole project where the fuller pipeline wins consistently rather than
     being flat, a modest real positive signal for self-critique.
   - This run **supersedes** the earlier 4-system/`max_repair_iterations=1`
     run as the canonical Table VI. `data/processed/evaluation_results_scaled.json`
     now holds the new 5-system/repair=2 data; the superseded run is
     archived at `evaluation_results_scaled_repair1.json`.

## Status: figure fixes, a real worked example, and published-baseline comparison (2026-08-30, later)

Three more rounds of direct user feedback, each addressed for real, not cosmetically:

1. **"Figure 2 is too wide and overlap with text, figure 3 is not readable"**
   — Figure 2 (knowledge-graph excerpt) was rendering wider than the IEEE
   column and bleeding into the adjacent column's text; wrapped in
   `\resizebox{\columnwidth}{!}{...}`. Figure 3 (a pilot-run bar chart) was
   illegible at print size; replaced with Table IX, a compact min/max/spread
   table over the same three metrics — states the actual analytical point
   (BERTScore's spread is an order of magnitude smaller than ROUGE-1's or
   METEOR's) more directly than the chart did.
2. **"There is no example from our running, to show what are we doing"**
   — added `scripts/run_worked_example.py`, which runs the real, complete
   4-stage pipeline once on one decision context and saves the full
   structured result to `data/processed/worked_example.json`. The real run
   turned out to be a good illustration on its own: under `tactic_budget=4`,
   Stage 3 selected only 3 tactics (leaving one budget slot unused) because
   the deliberated candidate's text never named a fourth tactic covering
   security or maintainability — a concrete instance of exactly the
   mechanism the aggregate results diagnose, not a contrived example. Now
   manuscript Section III-G, with Table IV of its per-attribute utility
   scores.
3. **"We evaluated using only [our own] dataset and no baseline model from
   published papers... the comparison is weak"** — real, substantive gap:
   every baseline in the evaluation was our own re-implementation, never an
   actual published system's reported numbers. Fixed by going back to the
   "Context Matters" replication package (already used for the retrieval
   corpus) and discovering it ships its own real generations and scores
   from four frontier models (Gemini-2.5-Pro, GLM-4.6, Qwen3-235B,
   Gemma3-4B) under 5 real prompting strategies, computed with equivalent
   metric libraries to our own (HuggingFace `evaluate`'s ROUGE/METEOR/
   BERTScore; its BLEU is 0–1 scale, so ×100 for consistency with our
   sacrebleu numbers). Matched our exact 3 scaled-run held-out items into
   that package **by title** (verified unique per repo) and extracted the
   identical-item scores under its `Baseline` (matches our `zero_shot`)
   and `RAG_Based` $k=3$ (matches our `retrieval_only`) strategies — a
   true apples-to-apples comparison against real published numbers on the
   same items, not an approximation. New:
   `src/evaluation/published_baselines.py` (index-matching + aggregation,
   tested), `scripts/extract_published_baseline_comparison.py`, and the
   committed `data/processed/published_baseline_comparison.json`. Now
   manuscript Section IV-E with Table VIII, honestly caveated (these are
   much larger models, their prompt is considerably more elaborate than
   ours, $N=3$ can't support a capability claim either way — framed as an
   external sanity check, not a "our tiny model beats frontier models"
   claim, even though the raw numbers are close).

Manuscript is now **13 pages** (14-page CFP limit), **153 tests passing**.

## Status: fixed missing citations in the published-baseline comparison (2026-08-30, later still)

User caught it: Section IV-E named Gemini-2.5-Pro, GLM-4.6, Qwen3-235B,
Gemma3-4B, and the "Context Matters" package with zero `\cite`s. Researched
(fetched each arXiv abstract page directly, no assumed IDs) and added real
citations: Gemini 2.5 (arXiv:2507.06261), Qwen3 (arXiv:2505.09388), Gemma 3
(arXiv:2503.19786). **GLM-4.6 itself has no dedicated technical report** —
it's a blog-style incremental update over GLM-4.5 with no new architecture
paper — cited GLM-4.5's report (arXiv:2508.06471) instead and said so
explicitly in the manuscript text rather than pretending a GLM-4.6-specific
paper exists. Bibliography: 27 → 31 entries.

**Manuscript is now exactly 14 pages — the CFP's stated limit.** Any future
addition needs to trim something else first; there is no headroom left.
153 tests passing.

## Next step

What's genuinely still open, in priority order:

0. **Consider whether the published-baseline comparison (Table VIII) should
   be extended to more of the corpus's 5 strategies/4 models**, or whether
   the current 2-strategy/4-model slice is sufficient — this was added
   reactively to close a real gap, not originally planned, so it hasn't
   been through a review cycle yet.
1. **Larger $N$** is still the single most valuable remaining action —
   giving repair its full budget ruled out one candidate explanation but
   did not add sample size; `B=5`'s 0% result is still one data point at
   $N=3$. Use `scripts/run_evaluation_repair2_verification.py` as the
   template (it already includes `cadence_no_critique` and
   `max_repair_iterations=2`) but with a larger `n_test_items` and a fresh
   seed (not 42 or 43). See the "background-task duration limit"
   Environment note below before attempting this in a background bash
   tool — the 5-system/repair=2 config that produced the current Table VI
   took noticeably longer than the earlier 4-system/repair=1 runs (exact
   duration not captured; budget real wall-clock time generously).
2. **Minor, low-priority code-hygiene item**: `scripts/run_evaluation_scaled.py`'s
   own `__main__` defaults still don't reproduce any of the committed
   tables (that script is now superseded for the canonical Table VI by
   `run_evaluation_repair2_verification.py`, which *does* hardcode exactly
   what it ran) — not a paper-correctness issue, just worth cleaning up or
   removing the stale script eventually.
3. **A third review pass, if time allows before the 31 Oct 2026
   deadline** — ideally by a human co-author, since two independent
   automated review cycles have now each found real issues the prior
   cycle missed, and the manuscript has changed substantially since the
   last review (new tables/figures, new evaluation data); diminishing
   returns are likely but not guaranteed.
~~4. One LOW-confidence item from the first review cycle: confirm MAAD's
   exact agent role names against the paper's actual text.~~ **Done** —
   fetched arxiv.org/abs/2507.21382 directly: confirms MAAD's four
   agents are exactly Analyst, Modeler, Designer, Evaluator, each doing
   what the manuscript's Table I and Section II-A describe.

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
