import zipfile
from pathlib import Path

from src.data.inventory import build_inventory, extract_archive


def _make_sample_zip(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("repoA/adr/0001-use-microservices.md", "# Decision\nUse microservices.")
        zf.writestr("repoA/adr/0002-use-postgres.md", "# Decision\nUse Postgres.")
        zf.writestr("repoB/decisions/decisions.json", '{"decisions": []}')
        zf.writestr("README.txt", "sample corpus")


def test_extract_archive_unpacks_all_entries(tmp_path):
    zip_path = tmp_path / "sample.zip"
    _make_sample_zip(zip_path)
    dest_dir = tmp_path / "extracted"

    result = extract_archive(zip_path, dest_dir)

    assert result == dest_dir
    assert (dest_dir / "repoA" / "adr" / "0001-use-microservices.md").exists()
    assert (dest_dir / "repoB" / "decisions" / "decisions.json").exists()


def test_build_inventory_counts_files_and_extensions(tmp_path):
    zip_path = tmp_path / "sample.zip"
    _make_sample_zip(zip_path)
    dest_dir = tmp_path / "extracted"
    extract_archive(zip_path, dest_dir)

    report = build_inventory(dest_dir)

    assert report.total_files == 4
    assert report.extension_counts[".md"] == 2
    assert report.extension_counts[".json"] == 1
    assert report.extension_counts[".txt"] == 1
    assert set(report.top_level_entries) == {"repoA", "repoB", "README.txt"}
