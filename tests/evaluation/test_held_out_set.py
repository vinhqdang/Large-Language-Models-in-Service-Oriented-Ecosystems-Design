import dataclasses
import json

import pytest

from src.retrieval.records import ADRRecord
from src.evaluation.held_out_set import load_verified_records, sample_test_set


def _record(record_id, status, raw_text):
    return ADRRecord(
        record_id=record_id, repo_folder="r", repository_url=None,
        relative_path=record_id, sequence_number=1, title=f"Title {record_id}",
        raw_text=raw_text, extraction_status=status,
    )


def test_load_verified_records_filters_by_status(tmp_path):
    records = [
        _record("a", "Verified", "x" * 400),
        _record("b", "Doubt (name sequence)", "x" * 400),
        _record("c", "Verified", "x" * 400),
    ]
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(json.dumps(dataclasses.asdict(r)) for r in records) + "\n", encoding="utf-8"
    )

    result = load_verified_records(path)

    assert {r.record_id for r in result} == {"a", "c"}


def test_sample_test_set_excludes_short_records():
    records = [_record("short", "Verified", "x" * 50), _record("long", "Verified", "x" * 400)]

    result = sample_test_set(records, n=1, min_length=300)

    assert [r.record_id for r in result] == ["long"]


def test_sample_test_set_is_reproducible_with_same_seed():
    records = [_record(str(i), "Verified", "x" * 400) for i in range(20)]

    first = sample_test_set(records, n=5, seed=7)
    second = sample_test_set(records, n=5, seed=7)

    assert [r.record_id for r in first] == [r.record_id for r in second]


def test_sample_test_set_raises_when_not_enough_eligible_records():
    records = [_record("a", "Verified", "x" * 400)]

    with pytest.raises(ValueError):
        sample_test_set(records, n=5)
