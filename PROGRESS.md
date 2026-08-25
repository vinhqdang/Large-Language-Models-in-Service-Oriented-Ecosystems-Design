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

## Next step (not started)

Write and execute the **retrieval-indexing** implementation plan: parse the
real corpus (per the schema inspection above) into an `ADRRecord` schema,
build embeddings + a vector index, implement Stage 1 of the CADENCE pipeline
(spec §3). This plan doesn't exist yet — write it fresh once the schema
inspection is done, per the same brainstorming → spec → plan → SDD-execution
workflow used for corpus acquisition.

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
