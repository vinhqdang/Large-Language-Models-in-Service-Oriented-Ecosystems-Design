"""Match our held-out items against the "Context Matters" replication
package's own published per-item generations, so the manuscript can compare
against real reported frontier-model results (not just our own
re-implementations of baseline-style prompting) on the exact same items.

The package's ``Results/{strategy}/{model}/Generated_ADRs/{repo}.json``
files hold one entry per ADR generated for that repo, each with the
model's real "generation" and the real "ground-truth" reference text --
matching our held-out item by title (stable across the package's own
reruns and ours) gives the exact generation/reference pair for that item
under that model/strategy.

We deliberately do NOT use the package's own precomputed
``Evaluations/{repo}.json`` scores: its evaluation script
(``Experiments/CommonCode/evaluate_ADRs.py``) calls
``compare_ADRs(reference=generation, hypothesis=ground_truth)`` -- i.e.
its BLEU/METEOR are computed with prediction and reference reversed
relative to this project's own ``src/evaluation/metrics.py`` (and its
ROUGE uses no stemmer, while ours does), making its raw scores not
comparable to this work's own metric numbers. Recomputing from the raw
generation/ground-truth text with this project's own
``compute_corpus_metrics`` instead gives a genuinely apples-to-apples
comparison: identical metric code, identical BERTScore checkpoint,
identical orientation, on both sides of the table.
"""
import json
from pathlib import Path


def find_dataset_index(dataset_path: Path, title: str) -> int | None:
    """Return the index of the entry whose title matches, or None if the
    file doesn't exist or no entry matches. Works on any package file that
    is a list of dicts with a "title" key (Dataset/ or Generated_ADRs/)."""
    if not dataset_path.exists():
        return None
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    for i, item in enumerate(data):
        if item.get("title", "").strip() == title.strip():
            return i
    return None


def load_generation_pair(generated_adrs_path: Path, index: int) -> tuple[str, str] | None:
    """Return (generation, ground_truth) at `index` in a
    Generated_ADRs/{repo}.json file, or None if the file doesn't exist or
    has fewer entries than `index`."""
    if not generated_adrs_path.exists():
        return None
    data = json.loads(generated_adrs_path.read_text(encoding="utf-8"))
    if index >= len(data):
        return None
    entry = data[index]
    return entry["generation"], entry["ground-truth"]


def mean_scores(per_item: list[dict], metric_keys: tuple[str, ...]) -> dict[str, float | None]:
    """Average each of `metric_keys` across `per_item` entries (each with a
    "scores" dict, possibly None for an unmatched item). A metric missing
    from every matched item (or no items matched at all) reports None
    rather than a misleading 0.0."""
    result: dict[str, float | None] = {}
    for key in metric_keys:
        values = [
            item["scores"][key]
            for item in per_item
            if item.get("scores") is not None and key in item["scores"]
        ]
        result[key] = (sum(values) / len(values)) if values else None
    return result
