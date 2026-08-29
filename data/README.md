# ADR Corpus

Source: "Context Matters" replication package, Zenodo DOI
[10.5281/zenodo.18370195](https://doi.org/10.5281/zenodo.18370195), derived from
Buchgeher, Schöberl, Geist, Dorninger, Haindl, Weinreich, "Using Architecture
Decision Records in Open Source Projects — An MSR Study on GitHub," IEEE Access,
vol. 11, pp. 63725-63740, 2023, DOI
[10.1109/ACCESS.2023.3287654](https://doi.org/10.1109/ACCESS.2023.3287654).
License: CC-BY-4.0.

`raw/` and `extracted/` are gitignored — regenerate with
`python scripts/fetch_adr_corpus.py`. `corpus_inventory.json` is committed and
summarizes the extracted structure (file counts by extension, top-level entries)
for downstream planning.

**Cleanup status:** `raw/` (the zip) is deleted automatically once extraction
succeeds. `extracted/` (~11 GB) has now been fully consumed by
`scripts/build_adr_dataset.py`, which parsed it into `processed/adr_records.jsonl`
(6,173 ADRs, ~19.5 MB, committed) — the only thing later plans read from.
`extracted/` has been deleted; regenerate it (and re-run
`scripts/build_adr_dataset.py`) only if you need to re-derive
`processed/` from the raw corpus again — normal work should never need to.

## `processed/adr_records.jsonl`

One JSON object per line, one line per ADR file across 882 of the 883
repository folders under `Data/ADRs/` in the "Context Matters" package (one
folder, `csc_swr_architecture_source_decisions_adrs`, is empty and
contributes no records) (see
`docs/superpowers/plans/2026-08-26-adr-retrieval-indexing.md` for the full
schema inspection this was built against). Fields: `record_id`,
`repo_folder`, `repository_url`, `relative_path`, `sequence_number`
(nullable — ~4.5% of files have no parseable leading number), `title`
(best-effort — first non-section-name markdown heading, else the filename),
`raw_text` (full file content), `extraction_status` (the corpus's own
extraction-confidence label: `Verified`, one of several `Doubt (...)`
categories, or `unknown` if a folder had no matching `dataset_index.json`
entry). Produced by `scripts/build_adr_dataset.py`; parsed by
`src/retrieval/records.py`.
