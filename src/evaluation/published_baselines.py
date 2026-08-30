"""Match our held-out items against the "Context Matters" replication
package's own published per-item generations and scores, so the manuscript
can compare against real reported frontier-model results (not just our own
re-implementations of baseline-style prompting) on the exact same items.

The package's ``Results/{strategy}/{model}/Dataset/{repo}.json`` files hold
one entry per ADR generated for that repo, in the same order as the
package's own ``Results/{strategy}/{model}/Evaluations/{repo}.json`` scores
-- so matching our held-out item to its index in ``Dataset`` (by title,
since titles are stable across the package's own reruns and ours) gives the
score for that exact item under that model/strategy.
"""
import json
from pathlib import Path


def find_dataset_index(dataset_path: Path, title: str) -> int | None:
    """Return the index of the entry whose title matches, or None if the
    file doesn't exist or no entry matches."""
    if not dataset_path.exists():
        return None
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    for i, item in enumerate(data):
        if item.get("title", "").strip() == title.strip():
            return i
    return None


def load_eval_entry(eval_path: Path, index: int) -> dict | None:
    """Return the Evaluations[]  entry at `index`, or None if the file
    doesn't exist or has fewer entries than `index` (a real mismatch
    between Dataset and Evaluations that should not be silently ignored
    upstream, but is a legitimate "no data" signal here)."""
    if not eval_path.exists():
        return None
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    if index >= len(data):
        return None
    return data[index]


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
