"""Build a genuinely apples-to-apples comparison against the "Context
Matters" replication package's own real generations for exactly our N=3
scaled-evaluation held-out items (seed=43), under its "Baseline"
(title-only, no context -- matches our zero_shot) and "RAG_Based/K_3"
(semantic retrieval, k=3 -- matches our retrieval_only) strategies,
across all four models the package itself evaluated (Gemini-2.5-Pro,
Gemma3-4b, GLM-4.6, Qwen3-235b).

This gives a genuine head-to-head against real published frontier-model
results on the identical held-out items, not just our own
re-implementations of baseline-style prompting -- addressing reviewer
feedback that the evaluation compared only against our own baselines,
never an actual published system's reported numbers.

We recompute all four metrics from the package's raw generation/
ground-truth text pairs using THIS PROJECT'S OWN `compute_corpus_metrics`
(src/evaluation/metrics.py), rather than trusting the package's own
precomputed Evaluations/*.json scores -- a third review cycle found and
verified that the package's own evaluation script computes BLEU/METEOR
with prediction and reference reversed relative to our own metrics code,
and ROUGE without a stemmer where ours uses one, making its raw numbers
not comparable to this work's own. Recomputing from the raw text with
identical metric code (same BERTScore checkpoint, same orientation, same
stemming) on both sides eliminates every cross-implementation difference
at once, rather than patching individual numbers.

Import order: src.retrieval.embeddings (sentence_transformers) is
imported before anything that triggers torch via bert-score, per the
sentence_transformers-before-torch rule in PROGRESS.md.

Requires `data/extracted/Context Matters/Results/` (the raw corpus
archive) to be present -- regenerate with `scripts/fetch_adr_corpus.py`
if it has since been deleted; this script's own output
(data/processed/published_baseline_comparison.json) is committed and
small, so re-running this script is only needed if the held-out sample
or model/strategy list changes.
"""
import dataclasses
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.retrieval.embeddings import embed_texts, load_embedding_model  # noqa: F401  (import before torch)

from src.evaluation.held_out_set import load_verified_records, sample_test_set
from src.evaluation.metrics import compute_corpus_metrics, load_bertscorer
from src.evaluation.published_baselines import find_dataset_index, load_generation_pair

RESULTS_ROOT = _PROJECT_ROOT / "data" / "extracted" / "Context Matters" / "Results"
RECORDS_PATH = _PROJECT_ROOT / "data" / "processed" / "adr_records.jsonl"
OUT_PATH = _PROJECT_ROOT / "data" / "processed" / "published_baseline_comparison.json"

MODELS = ["Gemini-2.5-Pro", "Gemma3-4b", "GLM-4.6", "Qwen3-235b"]
# (harness_strategy_name, package_experiment_dir, package_generated_adrs_subdir)
STRATEGIES = [
    ("Baseline", "Baseline", ""),
    ("RAG_Based_K3", "RAG_Based", "/K_3"),
]


def extract_comparison(records_path: Path, results_root: Path, scorer) -> dict:
    verified = load_verified_records(records_path)
    heldout = sample_test_set(verified, n=3, min_length=300, seed=43)

    per_strategy = {}
    for strategy_name, exp_dir, sub in STRATEGIES:
        per_model = {}
        for model in MODELS:
            per_item = []
            for record in heldout:
                generated_adrs_path = results_root / exp_dir / model / f"Generated_ADRs{sub}" / f"{record.repo_folder}.json"
                index = find_dataset_index(generated_adrs_path, record.title)
                pair = load_generation_pair(generated_adrs_path, index) if index is not None else None
                scores = None
                if pair is not None:
                    generation, ground_truth = pair
                    recomputed = compute_corpus_metrics([generation], [ground_truth], scorer=scorer)[0]
                    scores = dataclasses.asdict(recomputed)
                per_item.append({
                    "record_id": record.record_id,
                    "repo_folder": record.repo_folder,
                    "title": record.title,
                    "dataset_index": index,
                    "scores": scores,
                })
            per_model[model] = per_item
        per_strategy[strategy_name] = per_model

    return {
        "held_out_items": [
            {"record_id": r.record_id, "repo_folder": r.repo_folder, "title": r.title} for r in heldout
        ],
        "metric_note": (
            "All scores recomputed from the package's raw generation/ground-truth "
            "text pairs (Generated_ADRs/) using this project's own "
            "compute_corpus_metrics, NOT the package's own precomputed "
            "Evaluations/*.json scores -- those use a reversed prediction/reference "
            "orientation for BLEU/METEOR and no ROUGE stemmer, making them "
            "non-comparable to this work's own metric numbers."
        ),
        "results": per_strategy,
    }


if __name__ == "__main__":
    scorer = load_bertscorer()
    comparison = extract_comparison(RECORDS_PATH, RESULTS_ROOT, scorer)
    OUT_PATH.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")

    for strategy_name, per_model in comparison["results"].items():
        print(f"\n=== {strategy_name} ===")
        for model, per_item in per_model.items():
            found = [i for i in per_item if i["scores"] is not None]
            print(f"  {model}: {len(found)}/{len(per_item)} items matched")
            if len(found) < len(per_item):
                missing = [i["repo_folder"] for i in per_item if i["scores"] is None]
                print(f"    missing: {missing}")
