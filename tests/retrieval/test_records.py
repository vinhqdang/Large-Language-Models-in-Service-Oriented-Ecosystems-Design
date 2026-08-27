import json

from src.retrieval.records import ADRRecord, load_records_jsonl, parse_adr_folder, parse_corpus


def test_parse_adr_folder_extracts_title_and_sequence_number(tmp_path):
    folder = tmp_path / "myrepo_doc_adr"
    folder.mkdir()
    (folder / "0001-record-architecture-decisions.md").write_text(
        "# 1. Record architecture decisions\n\nDate: 2019-05-23\n\n"
        "## Status\n\nAccepted\n\n## Context\n\nWe need to record decisions.\n",
        encoding="utf-8",
    )

    records = parse_adr_folder(
        folder, repo_folder="myrepo_doc_adr", repository_url="https://github.com/x/myrepo.git",
        extraction_status="Verified",
    )

    assert len(records) == 1
    record = records[0]
    assert record.record_id == "myrepo_doc_adr/0001-record-architecture-decisions.md"
    assert record.sequence_number == 1
    assert record.title == "1. Record architecture decisions"
    assert "We need to record decisions." in record.raw_text
    assert record.extraction_status == "Verified"
    assert record.repository_url == "https://github.com/x/myrepo.git"


def test_parse_adr_folder_handles_inconsistent_filenames_and_missing_title(tmp_path):
    folder = tmp_path / "weird_repo_adr"
    folder.mkdir()
    (folder / "ADR-1_bootstrap-sentinel.md").write_text(
        "## Status\n\nAccepted\n", encoding="utf-8"
    )
    (folder / "splitting-and-bundling.md").write_text(
        "Just some notes, no heading at all.\n", encoding="utf-8"
    )
    (folder / "0001 - logging.md").write_text(
        "# Use structured logging\n\nDecision text.\n", encoding="utf-8"
    )

    records = {r.relative_path: r for r in parse_adr_folder(
        folder, repo_folder="weird_repo_adr", repository_url=None, extraction_status="Doubt (name sequence)",
    )}

    assert records["ADR-1_bootstrap-sentinel.md"].sequence_number == 1
    assert records["ADR-1_bootstrap-sentinel.md"].title == "ADR-1_bootstrap-sentinel"  # no heading -> filename fallback

    assert records["splitting-and-bundling.md"].sequence_number is None
    assert records["splitting-and-bundling.md"].title == "splitting-and-bundling"

    assert records["0001 - logging.md"].sequence_number == 1
    assert records["0001 - logging.md"].title == "Use structured logging"


def test_parse_adr_folder_skips_section_name_heading_and_uses_next_real_title(tmp_path):
    folder = tmp_path / "repo_adr"
    folder.mkdir()
    (folder / "0001-x.md").write_text(
        "## Status\n\nAccepted\n\n# Use a message queue\n\n## Decision\n\ntext\n",
        encoding="utf-8",
    )

    records = parse_adr_folder(folder, repo_folder="repo_adr", repository_url=None, extraction_status="Verified")

    assert records[0].title == "Use a message queue"


def test_parse_adr_folder_ignores_non_markdown_files(tmp_path):
    folder = tmp_path / "repo_adr"
    folder.mkdir()
    (folder / "0001-decision.md").write_text("# Decision\n", encoding="utf-8")
    (folder / "notes.txt").write_text("not an adr", encoding="utf-8")

    records = parse_adr_folder(folder, repo_folder="repo_adr", repository_url=None, extraction_status="Verified")

    assert len(records) == 1
    assert records[0].relative_path == "0001-decision.md"


def test_parse_corpus_walks_all_folders_and_looks_up_status(tmp_path):
    data_dir = tmp_path / "Data"
    adrs_dir = data_dir / "ADRs"
    adrs_dir.mkdir(parents=True)

    (adrs_dir / "repoA_adr").mkdir()
    (adrs_dir / "repoA_adr" / "0001-x.md").write_text("# X\n", encoding="utf-8")

    (adrs_dir / "repoB_adr").mkdir()
    (adrs_dir / "repoB_adr" / "0001-y.md").write_text("# Y\n", encoding="utf-8")

    (data_dir / "dataset_index.json").write_text(
        json.dumps({
            "repoA_adr": {
                "repository_url": "https://github.com/a/repoA.git",
                "local_folder": "ADRs/repoA_adr",
                "files_count": 1,
                "status": "Verified",
            }
            # repoB_adr intentionally has no index entry
        }),
        encoding="utf-8",
    )

    records = parse_corpus(data_dir)

    by_repo = {r.repo_folder: r for r in records}
    assert by_repo["repoA_adr"].extraction_status == "Verified"
    assert by_repo["repoA_adr"].repository_url == "https://github.com/a/repoA.git"
    assert by_repo["repoB_adr"].extraction_status == "unknown"
    assert by_repo["repoB_adr"].repository_url is None
    assert len(records) == 2


def test_load_records_jsonl_round_trips_and_preserves_file_order(tmp_path):
    records = [
        ADRRecord("r/1.md", "r", None, "1.md", 1, "First", "text 1", "Verified"),
        ADRRecord("r/2.md", "r", None, "2.md", 2, "Second", "text 2", "Verified"),
    ]
    path = tmp_path / "adr_records.jsonl"
    path.write_text(
        "\n".join(json.dumps(r.__dict__) for r in records) + "\n", encoding="utf-8"
    )

    result = load_records_jsonl(path)

    assert result == records
