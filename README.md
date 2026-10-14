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

## Headline findings (paired evaluation, N = 30 held-out ADRs)

- Retrieval slightly improves agreement with the recorded ADRs (BERTScore +0.004, significant).
- Multi-agent deliberation lowers agreement on all four metrics. Truncating the
  baselines to CADENCE's length closes only part of the gap, so this is not
  just a length effect.
- The solver flags 29 of 30 deliberated decisions as leaving required quality
  attributes without a committed tactic. Repair raises mean coverage from 0.57
  to 1.63 of 5 attributes but makes only 1 of 30 decisions feasible at B = 5,
  and 0 at B = 7. The bottleneck is tactic diversity, not budget.
- Stage 4 (critique) never changes the decision text. Its residual weaknesses,
  rendered as an ADR "Consequences" section, significantly increase agreement.
- The earlier N = 3 result crediting self-critique with gains (7 of 8
  comparisons) was sampling noise from unpaired runs, and is retracted.

Numbers, tables and caveats are in the manuscript; `PROGRESS.md` is the running log of decisions and history.

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

## Status

1. *IEEE Transactions on Services Computing* special issue: **rejected on scope
   grounds** (service-oriented framing judged to be motivation only).
2. Elsevier *Journal of Systems and Software*: **desk-rejected** (2026-10). The
   editor found that the paper did not engage enough with the software
   engineering literature or show how it advances the state of the art in SE.
3. The paper is now being revised for Elsevier *Information and Software
   Technology* (`manuscript/cadence_ist.tex`). The revision rebuilds the
   Related Work around the software-architecture literature, reframes the
   paper around four research questions, and replaces the N = 3 evaluation
   with a paired N = 30 evaluation.

## Manuscripts

| File | Target | Format |
|---|---|---|
| `manuscript/cadence.tex` / `.pdf` | IEEE TSC special issue (12 pages, rejected on scope) | `IEEEtran`, numbered citations |
| `manuscript/cadence_jss.tex` / `.pdf` | Elsevier *Journal of Systems and Software* (desk-rejected) | `elsarticle`, author-year citations |
| `manuscript/cadence_ist.tex` / `.pdf` | Elsevier *Information and Software Technology* (in preparation) | `elsarticle`, author-year citations, structured abstract |

The JSS version is self-contained: it includes the full 26-tactic catalog with
every trade-off edge (Appendix A) and every prompt template (Appendix B), so it
can be read without this repository. It also needs `cadence_jss.bib` and the
figure files it inputs (`arch_figure.tex`, `impl_figure.tex`,
`example_figure.tex`, `results_figure.tex`), so keep them in the same folder.
Its figures cover the pipeline, the component architecture, the software
architecture of this code base, the worked example as an architecture, the
knowledge-graph excerpt, and the evaluation results.

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
python scripts/run_evaluation_scaled.py     # earlier N = 3 evaluation (unpaired)
python scripts/run_evaluation_paired.py 30 44   # paired N = 30 evaluation (about 5 min per item on a T4; resumable)
python scripts/length_control.py            # re-score baselines truncated to CADENCE's length
python scripts/analyze_paired.py            # bootstrap CIs, Wilcoxon + Holm, solver and length statistics
python scripts/extract_published_baseline_comparison.py [--paired]   # recompute published-model metrics
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
