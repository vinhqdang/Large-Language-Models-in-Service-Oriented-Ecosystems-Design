import json

from scripts.fetch_adr_corpus import report_to_json, run_fetch


def test_report_to_json_round_trips(tmp_path):
    from src.data.inventory import InventoryReport

    report = InventoryReport(
        total_files=3,
        extension_counts={".md": 2, ".json": 1},
        top_level_entries=["repoA", "repoB"],
    )
    out_path = tmp_path / "inventory.json"

    report_to_json(report, out_path)

    loaded = json.loads(out_path.read_text())
    assert loaded == {
        "total_files": 3,
        "extension_counts": {".md": 2, ".json": 1},
        "top_level_entries": ["repoA", "repoB"],
    }


def test_run_fetch_calls_download_extract_inventory_in_order(tmp_path, monkeypatch):
    calls = []

    def fake_download(url, dest_path, expected_md5=None):
        calls.append(("download", url, dest_path, expected_md5))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"fake zip bytes")
        return dest_path

    def fake_extract(zip_path, dest_dir):
        calls.append(("extract", zip_path, dest_dir))
        dest_dir.mkdir(parents=True, exist_ok=True)
        return dest_dir

    def fake_inventory(root_dir):
        calls.append(("inventory", root_dir))
        from src.data.inventory import InventoryReport

        return InventoryReport(total_files=0, extension_counts={}, top_level_entries=[])

    monkeypatch.setattr("scripts.fetch_adr_corpus.download_file", fake_download)
    monkeypatch.setattr("scripts.fetch_adr_corpus.extract_archive", fake_extract)
    monkeypatch.setattr("scripts.fetch_adr_corpus.build_inventory", fake_inventory)

    run_fetch(data_dir=tmp_path)

    assert [c[0] for c in calls] == ["download", "extract", "inventory"]
    assert (tmp_path / "corpus_inventory.json").exists()


def test_run_fetch_deletes_raw_zip_after_successful_extraction(tmp_path, monkeypatch):
    def fake_download(url, dest_path, expected_md5=None):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"fake zip bytes")
        return dest_path

    def fake_extract(zip_path, dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        return dest_dir

    def fake_inventory(root_dir):
        from src.data.inventory import InventoryReport

        return InventoryReport(total_files=0, extension_counts={}, top_level_entries=[])

    monkeypatch.setattr("scripts.fetch_adr_corpus.download_file", fake_download)
    monkeypatch.setattr("scripts.fetch_adr_corpus.extract_archive", fake_extract)
    monkeypatch.setattr("scripts.fetch_adr_corpus.build_inventory", fake_inventory)

    run_fetch(data_dir=tmp_path)

    assert not (tmp_path / "raw" / "context_matters.zip").exists()
    assert not (tmp_path / "raw").exists()
