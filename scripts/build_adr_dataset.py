"""Parse the real ADR corpus into a compact, git-trackable JSONL dataset.

Run once after fetching the corpus (scripts/fetch_adr_corpus.py). After this
succeeds, data/extracted/ (~11 GB) is no longer needed and can be deleted —
this script's output is the only thing later plans read from.
"""
import dataclasses
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.retrieval.records import ADRRecord, parse_corpus

CORPUS_DATA_DIR = _PROJECT_ROOT / "data" / "extracted" / "Context Matters" / "Data"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


def records_to_jsonl(records: list[ADRRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(dataclasses.asdict(record)) + "\n")


def run_build(data_dir: Path, processed_dir: Path) -> Path:
    records = parse_corpus(data_dir)
    out_path = processed_dir / "adr_records.jsonl"
    records_to_jsonl(records, out_path)
    return out_path


if __name__ == "__main__":
    result_path = run_build(data_dir=CORPUS_DATA_DIR, processed_dir=PROCESSED_DIR)
    print(f"Wrote processed dataset to {result_path}")
