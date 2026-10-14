"""Length-controlled re-scoring of the paired evaluation.

Truncates each single-pass baseline's output to the word count of
Cadence-NC (B=5) on the same item and re-scores it, so the comparison with
Cadence-NC no longer rewards the baselines' longer outputs. Writes
data/processed/evaluation_length_control.json.

Usage: python scripts/length_control.py
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import compute_corpus_metrics, load_bertscorer  # noqa: E402

PROCESSED = _PROJECT_ROOT / "data" / "processed"
ITEMS_PATH = PROCESSED / "evaluation_paired_items.jsonl"
OUT_PATH = PROCESSED / "evaluation_length_control.json"
TARGET = ("5", "cadence_no_critique")


def main():
    rows = [json.loads(l) for l in ITEMS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if "error" not in r]
    lengths = [len(r["budgets"][TARGET[0]][TARGET[1]].split()) for r in rows]
    refs = [r["reference"] for r in rows]
    scorer = load_bertscorer()
    out = {"target": f"{TARGET[1]}@B={TARGET[0]}", "target_words": lengths, "per_item_metrics": {}}
    for system in ("zero_shot", "retrieval_only"):
        truncated = [" ".join(r[system].split()[:n]) for r, n in zip(rows, lengths)]
        scores = compute_corpus_metrics(truncated, refs, scorer)
        out["per_item_metrics"][f"{system}_truncated"] = {r["record_id"]: asdict(s) for r, s in zip(rows, scores)}
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
