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
succeeds. `extracted/` (~11 GB) is intentionally still on disk — the next
plan (retrieval indexing) reads from it to build `data/processed/` (a compact
parsed dataset). **Once that plan has run and `data/processed/` exists,
delete `data/extracted/` manually** (`Remove-Item -Recurse -Force data\extracted`) — it is fully
regenerable from the Zenodo source above and should not be left on disk
long-term.
