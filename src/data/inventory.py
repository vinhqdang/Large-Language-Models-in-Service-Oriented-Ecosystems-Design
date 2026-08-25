"""Archive extraction and structural inventory for the ADR corpus."""
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


def extract_archive(zip_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    return dest_dir


@dataclass
class InventoryReport:
    total_files: int
    extension_counts: dict[str, int] = field(default_factory=dict)
    top_level_entries: list[str] = field(default_factory=list)


def build_inventory(root_dir: Path) -> InventoryReport:
    ext_counter: Counter[str] = Counter()
    total_files = 0

    for path in root_dir.rglob("*"):
        if path.is_file():
            total_files += 1
            ext_counter[path.suffix or "<no-ext>"] += 1

    top_level = sorted(p.name for p in root_dir.iterdir())

    return InventoryReport(
        total_files=total_files,
        extension_counts=dict(ext_counter),
        top_level_entries=top_level,
    )
