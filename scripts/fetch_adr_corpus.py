"""Fetch, extract, and inventory the ADR corpus used for retrieval and evaluation.

Source: "Context Matters" replication package (Zenodo DOI 10.5281/zenodo.18370195),
a validated sequential ADR dataset derived from Buchgeher et al.'s GitHub mining
study (IEEE Access, DOI 10.1109/ACCESS.2023.3287654). CC-BY-4.0.
"""
import dataclasses
import json
from pathlib import Path

from src.data.download import download_file
from src.data.inventory import InventoryReport, build_inventory, extract_archive

ZENODO_URL = "https://zenodo.org/records/18370195/files/Context%20Matters.zip?download=1"
EXPECTED_MD5 = "1106da3185ac5ddba0fdfc2f0ace9301"


def report_to_json(report: InventoryReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataclasses.asdict(report), indent=2))


def run_fetch(data_dir: Path) -> InventoryReport:
    """Download, extract, and inventory the corpus.

    The raw zip is deleted immediately after successful extraction: its MD5
    was already verified during download, so keeping both the zip and its
    extracted contents on disk is pure redundancy (~460 MB saved). The much
    larger extracted/ directory (~11 GB) is intentionally kept — later plans
    (retrieval indexing) still need to read from it — and should be deleted
    manually once that follow-on plan has produced its compact processed
    dataset from it.
    """
    raw_dir = data_dir / "raw"
    extracted_dir = data_dir / "extracted"
    zip_path = raw_dir / "context_matters.zip"

    download_file(ZENODO_URL, zip_path, expected_md5=EXPECTED_MD5)
    extract_archive(zip_path, extracted_dir)
    report = build_inventory(extracted_dir)

    report_to_json(report, data_dir / "corpus_inventory.json")

    zip_path.unlink()
    if not any(raw_dir.iterdir()):
        raw_dir.rmdir()

    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    result = run_fetch(data_dir=project_root / "data")
    print(f"Total files: {result.total_files}")
    print(f"Extension counts: {result.extension_counts}")
    print(f"Top-level entries: {result.top_level_entries}")
