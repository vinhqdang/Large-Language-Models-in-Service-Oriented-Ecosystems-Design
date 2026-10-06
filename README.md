# CADENCE

**Case-Grounded, Deliberation-Driven, and Solver-Verified Architectural Decision-Making with Large Language Models**

CADENCE is a four-stage algorithm that helps an LLM produce an architecture
decision record (ADR) that is grounded in precedent, argued from competing
quality-attribute viewpoints, and checked for feasibility by a formal solver
rather than by a second LLM opinion.

| Stage | What it does | Code |
|---|---|---|
| 1. Retrieval | Case-based reasoning over 6,173 real, mined ADRs (dense vector search) | `src/retrieval` |
| 2. Deliberation | One agent per ISO/IEC 25010 quality attribute, grounded in a 26-tactic knowledge graph, debating to one candidate | `src/deliberation` |
| 3. Solver verification | Tactic budget and quality-attribute coverage encoded for Z3 (two phases, lexicographic optimisation) with an LLM repair loop | `src/solver` |
| 4. Self-critique | Deterministic structural utility score blended with a separate LLM critique, emitted with full provenance | `src/critique` |

`src/evaluation` re-implements four baselines (zero-shot, retrieval-only,
multi-agent without solver, CADENCE without critique) and the metrics
(BERTScore, BLEU, ROUGE-1, METEOR, constraint-satisfaction rate, repair
iterations). `src/data` fetches and inventories the corpus.

## Headline findings (small sample, N = 3 per run)

- At the one tactic budget where feasibility is arithmetically possible
  (B = 5), constraint satisfaction is 0 % and repair never converges, even with
  the full repair budget. This is reported as a diagnosed negative result.
- Self-critique gives a modest, consistent gain over the same pipeline without
  it (higher in 7 of 8 system/budget/metric comparisons).
- Surface n-gram metrics penalise CADENCE's deliberately terse, solver-parseable
  output; BERTScore stays comparable across systems.
- Against four published frontier models on the identical held-out items,
  `cadence_full` with a 1.5B backbone scores below all of them.

Numbers, tables and caveats are in the manuscript and in `PROGRESS.md`.

## Repository layout

```
src/            the implementation (retrieval, deliberation, solver, critique, evaluation, data)
scripts/        entry points: corpus fetch, index build, demos, evaluation runs, worked example
tests/          pytest suite (unit tests stay fast and need no network or data)
data/           corpus inventory and processed artefacts (ADR records, embeddings, result JSON)
manuscript/     paper sources and PDFs (see below)
docs/           design spec and implementation plans
PROGRESS.md     running log: status, decisions, environment notes, next steps
```

## Manuscripts

| File | Target | Format |
|---|---|---|
| `manuscript/cadence.tex` / `.pdf` | IEEE Transactions on Services Computing (special issue); cut to the 12-page limit, then **rejected on scope grounds** (not technical merit) | `IEEEtran`, numbered citations |
| `manuscript/cadence_jss.tex` / `.pdf` | Elsevier *Journal of Systems and Software* | `elsarticle`, author-year citations |

The JSS version also needs `cadence_jss.bib` and the figure files it inputs
(`arch_figure.tex`, `impl_figure.tex`, `example_figure.tex`,
`results_figure.tex`), so keep them in the same folder. Its figures cover the
pipeline, the component architecture, the software architecture of this code
base, the worked example as an architecture, the knowledge-graph excerpt, and
the evaluation results.

Build (needs a TeX distribution with `elsarticle`, `pgfplots`, `adjustbox`,
`placeins`, `natbib`):

```bash
cd manuscript
pdflatex cadence_jss && bibtex cadence_jss && pdflatex cadence_jss && pdflatex cadence_jss
```

## Setup

Python 3.13. With conda:

```bash
conda env create -f environment.yml && conda activate py313
# or: pip install -r requirements.txt
```

Install `torch` with CUDA first if you want GPU generation (see the comment in
`environment.yml`). The local backbone is `Qwen2.5-1.5B-Instruct`; a Gemini
client is also available (`src/deliberation/llm_client.py`).

## Running

```bash
python scripts/fetch_adr_corpus.py          # download + inventory the corpus (large; only if rebuilding)
python scripts/build_adr_dataset.py         # parse it into data/processed/adr_records.jsonl
python scripts/build_retrieval_index.py     # embed records -> data/processed/adr_embeddings.npy

python scripts/run_cadence_demo.py          # full four-stage pipeline on one context
python scripts/run_worked_example.py        # the manuscript's worked example
python scripts/run_evaluation_scaled.py     # scaled evaluation (budget real wall-clock time)
python scripts/extract_published_baseline_comparison.py   # recompute published-model metrics
```

The committed `data/processed/` already contains the ADR records, embeddings and
result files, so tests and most scripts run without re-fetching the corpus.
Generation runs are slow on a 1.5B local model and may hit execution-time limits
in managed environments; see the environment notes in `PROGRESS.md`.

Tests:

```bash
pytest -q
```

## Data and licence

The ADR corpus comes from the "Context Matters" replication package (Zenodo,
DOI [10.5281/zenodo.18370195](https://doi.org/10.5281/zenodo.18370195), CC-BY-4.0),
derived from Buchgeher et al., *IEEE Access* 11, 2023. See `data/README.md`.
Code is under the licence in `LICENSE`.

## Author

Quang-Vinh Dang, British University Vietnam, Hung Yen, Vietnam
(vinh.dq4@buv.edu.vn, ORCID 0000-0002-3877-8024).
