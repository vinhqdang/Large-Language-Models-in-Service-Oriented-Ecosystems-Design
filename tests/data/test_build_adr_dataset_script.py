import json

from scripts.build_adr_dataset import records_to_jsonl, run_build
from src.retrieval.records import ADRRecord


def test_records_to_jsonl_round_trips(tmp_path):
    records = [
        ADRRecord(
            record_id="repoA_adr/0001-x.md",
            repo_folder="repoA_adr",
            repository_url="https://github.com/a/repoA.git",
            relative_path="0001-x.md",
            sequence_number=1,
            title="X",
            raw_text="# X\n",
            extraction_status="Verified",
        ),
    ]
    out_path = tmp_path / "adr_records.jsonl"

    records_to_jsonl(records, out_path)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])
    assert loaded == {
        "record_id": "repoA_adr/0001-x.md",
        "repo_folder": "repoA_adr",
        "repository_url": "https://github.com/a/repoA.git",
        "relative_path": "0001-x.md",
        "sequence_number": 1,
        "title": "X",
        "raw_text": "# X\n",
        "extraction_status": "Verified",
    }


def test_build_dataset_calls_parse_corpus_and_writes_output(tmp_path, monkeypatch):
    fake_records = [
        ADRRecord("r/0001-x.md", "r", None, "0001-x.md", 1, "X", "# X\n", "Verified"),
    ]
    calls = []

    def fake_parse_corpus(data_dir):
        calls.append(data_dir)
        return fake_records

    monkeypatch.setattr("scripts.build_adr_dataset.parse_corpus", fake_parse_corpus)

    out_path = run_build(data_dir=tmp_path / "Data", processed_dir=tmp_path / "processed")

    assert calls == [tmp_path / "Data"]
    assert out_path.exists()
    assert len(out_path.read_text(encoding="utf-8").splitlines()) == 1
