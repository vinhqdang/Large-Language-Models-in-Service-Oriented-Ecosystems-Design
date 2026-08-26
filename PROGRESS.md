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

## Status: corpus acquisition plan complete

`docs/superpowers/plans/2026-08-25-adr-corpus-acquisition.md` — all 4 tasks
done, individually reviewed, final whole-branch reviewed, one fix wave
applied and re-reviewed clean. Everything is merged to `main`.

**What exists in the repo now:**
- `src/data/download.py` — checksum-verified, retrying, atomic-write streaming downloader.
- `src/data/inventory.py`, `src/data/paths.py` — archive extraction + structural inventory, with Windows long-path (`\\?\`, including UNC) handling.
- `scripts/fetch_adr_corpus.py` — fetches, extracts, and inventories the real ADR corpus.
- `data/corpus_inventory.json`, `data/README.md` — committed; describe the real corpus structure and provenance.
- `pyproject.toml` — `pythonpath = ["."]` so both `pytest` and running scripts directly work from repo root.

**What is NOT in the repo (gitignored, regenerate locally):**
- `data/raw/` — deleted automatically after extraction anyway (nothing to regenerate).
- `data/extracted/` — the real corpus, ~11 GB, 48,008 files (41,654 `.json`, 6,258 `.md`, rest minor). **Regenerate with:**
  ```
  conda env create -n py313 -f environment.yml   # first time only
  conda run -n py313 python scripts/fetch_adr_corpus.py
  ```
  Source: Zenodo DOI [10.5281/zenodo.18370195](https://doi.org/10.5281/zenodo.18370195) (~460 MB download). Expect it to take a while and possibly need 2-3 attempts if the connection drops mid-stream — the downloader retries automatically (3 attempts, exponential backoff) but a truly flaky connection may still need a manual re-run.
- `.env` — **not committed anywhere, on purpose.** You must create it fresh on each machine with your own keys:
  ```
  GEMINI_API_KEY=...
  OPENROUTER_API_KEY=...
  ```
  (Primary LLM backbone is `gemini-3.5-flash-lite`; local open-weight model via CUDA is the reproducible secondary backbone — no key needed for that, just local GPU + conda env. OpenRouter is optional/tertiary only, per the spec §6.)

## Deferred but tracked (don't forget)

- `data/extracted/` (~11 GB) should be **manually deleted** once the next plan (retrieval indexing) has parsed it into a compact `data/processed/` — this is documented in `data/README.md` too, but it depends on someone actually doing it. Not done yet as of this log entry.
- The corpus is **JSON-heavy** (6.6:1 over Markdown by file count) — the next plan must open with a real structural/schema inspection (directory names at depth 2-4, per-directory counts, a sampled JSON key-schema) before writing any parser. The current `InventoryReport` (a flat extension histogram) is deliberately not sufficient for this — that was a scoped decision, not an oversight, made during the corpus-acquisition plan's final review.

## Session 2026-08-26

- **Environment fix found and verified:** in the `py313` conda env, `import torch` before `import sentence_transformers` segfaults (exit 139) on Windows — upgrading sentence-transformers 3.3.1→6.0.0 did *not* fix it. Workaround: **always `import sentence_transformers` before `import torch`** in any module that uses both. `z3` (needed later for CADENCE Stage 3) is still not installed — unrelated, separate gap for that stage's plan.
- **Downloader hardened.** The Zenodo download kept read-timing-out after only a few MB, and the old code restarted from byte 0 on every retry, so it could never finish over a flaky connection. Fixed in `src/data/download.py`: HTTP Range-based resume across retries (falls back to a full restart if the server doesn't honor 206, including on an outright rejection like 416), wider read timeout, more attempts, and (per code review) a stale/unrelated `.part` file is never trusted and cleanup now covers every exception type, not just `requests.exceptions.RequestException`. 21 tests, reviewed twice (initial pass found 1 critical + 1 high + 1 medium issue, all fixed and re-verified clean). Corpus re-fetch then succeeded: `data/extracted/` regenerated, matches the previously committed `data/corpus_inventory.json` exactly (48,008 files).
- **Real schema inspection done** (the corpus is the "Context Matters" replication package in full, not just a bare ADR dump):
  - `Data/ADRs/{repo}_{adr_folder}/*.md` — the actual retrieval corpus. **883 on-disk folders, 6,173 `.md` files** (one folder has 0). Distribution: median 4 files/repo, mean 6.76, max 129; buckets `1-5`:545, `6-20`:289, `21-50`:41, `51+`:7 — confirms the spec's sparsity note.
  - `Data/dataset_index.json` (954 entries) gives an extraction-confidence status per folder: `Verified`:750, `Doubt (name sequence)`:96, `Repo Inaccessible`:40 (0 files), `Doubt (no repo dir)`:33, `Doubt (missing file)`:30, `Doubt (file contents)`:5. `Data/data.csv` has the same info in one-row-per-repo CSV form (922 rows). Plan: keep all on-disk records for retrieval (more data helps), but carry the status through as `extraction_status` on each parsed record so a later plan (evaluation ground truth) can filter to `Verified`-only if it wants a cleaner held-out split.
  - **Filenames are inconsistent** across repos — `0001-slug.md`, `0001 - slug.md`, `adr-0001 slug.md`, `ADR-1_slug.md`, 3-digit (`001-...`), 0-indexed (`0000-...`), and some files with **no leading number at all** (e.g. `splitting-and-bundling.md`). A parser must extract a sequence number leniently (regex on leading digits) and tolerate `None` — never assume a strict `NNNN-slug.md` pattern.
  - ADR body format is the standard Nygard/MADR style (`# Title`, `Date:`, `## Status`, `## Context`, `## Decision`, `## Consequences`) but not universally — decided to keep `ADRRecord` minimal (title + full raw text + metadata) rather than build a brittle general section-splitter; downstream stages can regex out Status/Context/Decision from `raw_text` themselves if/when they specifically need it.
  - **Bonus find, not in scope now but worth remembering:** `Experiments/` and `Results/` in the same package are the *Context Matters paper's own* generation+evaluation pipeline — `Results/{Baseline,All_N,First_K,Last_K,RAG_Based}/{Model}/Dataset/{repo}.json` (candidate ADRs, `{title, content, context}`) and `.../Evaluations/{repo}.json` (`rouge1_f`, `rouge2_f`, `rougeL_f`, `bleu_avg`, `meteor`, `bert_f1/precision/recall`, length stats — exactly the metric set spec §5 calls for) for 4 models × 5 strategies. This accounts for the bulk of the corpus's 41,654 JSON files. **When the evaluation-harness plan comes up, check whether the "Context-Matters-style retrieval-only" baseline (spec §5(b)) can reuse these existing generations/scores directly instead of re-running that pipeline.** `Experiments/RAG_Based/*.py` is also a working reference implementation of a retrieval-only ADR-generation baseline.

## Next step

`docs/superpowers/plans/2026-08-26-adr-retrieval-indexing.md` — implementation
plan for parsing `Data/ADRs/` into `ADRRecord`s, building embeddings + a
vector index, and exposing Stage 1 of the CADENCE pipeline (spec §3). Written
this session against the real schema above; execute task-by-task next.

After that, remaining plans in pipeline order: multi-agent deliberation (KG
grounding) → constraint solver + repair loop → self-critique → evaluation
harness (baselines: Dhar et al. ICSA'24-style, Context-Matters-style
retrieval-only, MAAD-style no-solver ablation; metrics: BERTScore/BLEU/
ROUGE-1/METEOR + novel constraint-satisfaction-rate + repair-convergence) →
manuscript (IEEEtran template already in repo root, 6 sections, ≤14 pages).

## Working conventions established so far

- Python 3.13 via conda env `py313` for everything.
- Work happens directly on `main` (no feature branches) — user's explicit choice.
- Push after every commit, no Claude identity in commits (global user convention).
- Implementation plans executed via subagent-driven-development: fresh implementer per task, task review, final whole-branch review, one fix wave max.
