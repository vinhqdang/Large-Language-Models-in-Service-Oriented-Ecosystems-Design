"""Held-out evaluation set construction.

Query construction follows the corpus's own upstream precedent for this
exact task ("ADR Generation from Titles" -- see data/README.md): title as
decision context, full body as ground truth. Filters to the corpus's own
high-confidence Verified subset for a clean reference.
"""
import json
import random
from pathlib import Path

from src.retrieval.records import ADRRecord


def load_verified_records(records_path: Path) -> list[ADRRecord]:
    records = []
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            record = ADRRecord(**json.loads(line))
            if record.extraction_status == "Verified":
                records.append(record)
    return records


def sample_test_set(
    records: list[ADRRecord], n: int, min_length: int = 300, seed: int = 42
) -> list[ADRRecord]:
    eligible = [r for r in records if len(r.raw_text) >= min_length]
    if len(eligible) < n:
        raise ValueError(f"Only {len(eligible)} eligible records, need {n}")
    return random.Random(seed).sample(eligible, n)
