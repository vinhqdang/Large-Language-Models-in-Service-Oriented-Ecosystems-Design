"""ADR corpus schema and parser.

The corpus ("Context Matters" replication package, see data/README.md) has
inconsistent filenames across repos (0001-x.md, 0001 - x.md, ADR-1_x.md,
some with no leading number at all) and body formats that don't universally
follow Nygard/MADR headings. This parser deliberately extracts only a
best-effort title and sequence number and keeps everything else in
raw_text, rather than a brittle general section-splitter — see the
retrieval-indexing plan's self-review notes for why.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

_LEADING_NUMBER = re.compile(r"^\D*(\d+)")
_HEADINGS = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
# Common ADR section names that sometimes appear as a file's first heading
# when the file has no actual title heading (e.g. a template missing its
# top-level "# <title>" line) — never worth using as the title itself.
_NON_TITLE_HEADINGS = {
    "status", "context", "decision", "decisions", "consequences",
    "consequence", "options considered", "options", "problem context",
}


@dataclass(frozen=True)
class ADRRecord:
    record_id: str
    repo_folder: str
    repository_url: str | None
    relative_path: str
    sequence_number: int | None
    title: str
    raw_text: str
    extraction_status: str


def _extract_sequence_number(filename: str) -> int | None:
    match = _LEADING_NUMBER.match(filename)
    return int(match.group(1)) if match else None


def _extract_title(raw_text: str, filename: str) -> str:
    for match in _HEADINGS.finditer(raw_text):
        candidate = match.group(1).strip()
        if candidate.lower() not in _NON_TITLE_HEADINGS:
            return candidate
    return Path(filename).stem


def parse_adr_folder(
    folder_path: Path,
    repo_folder: str,
    repository_url: str | None,
    extraction_status: str,
) -> list[ADRRecord]:
    records = []
    for md_path in sorted(folder_path.glob("*.md")):
        raw_text = md_path.read_text(encoding="utf-8", errors="replace")
        records.append(
            ADRRecord(
                record_id=f"{repo_folder}/{md_path.name}",
                repo_folder=repo_folder,
                repository_url=repository_url,
                relative_path=md_path.name,
                sequence_number=_extract_sequence_number(md_path.name),
                title=_extract_title(raw_text, md_path.name),
                raw_text=raw_text,
                extraction_status=extraction_status,
            )
        )
    return records


def load_records_jsonl(records_path: Path) -> list[ADRRecord]:
    """Load a processed-dataset JSONL file (data/processed/adr_records.jsonl)
    back into ADRRecords, unfiltered and in file order -- the order every
    real script relies on to stay aligned with data/processed/adr_embeddings.npy's
    rows (see scripts/build_retrieval_index.py, which built them together).
    """
    records = []
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            records.append(ADRRecord(**json.loads(line)))
    return records


def parse_corpus(data_dir: Path) -> list[ADRRecord]:
    index_path = data_dir / "dataset_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}

    adrs_dir = data_dir / "ADRs"
    records = []
    for folder_path in sorted(p for p in adrs_dir.iterdir() if p.is_dir()):
        repo_folder = folder_path.name
        entry = index.get(repo_folder)
        repository_url = entry.get("repository_url") if entry else None
        extraction_status = entry.get("status", "unknown") if entry else "unknown"
        records.extend(
            parse_adr_folder(folder_path, repo_folder, repository_url, extraction_status)
        )
    return records
