# ADR Corpus Acquisition & Inspection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reliably fetch, verify, and inventory the real ADR dataset we will use for retrieval and evaluation, so the next plan (retrieval indexing) can be written against the dataset's *actual* on-disk schema instead of an assumed one.

**Architecture:** A small `src/data` package with three pure, independently-testable pieces — checksummed download, archive extraction, and structural inventory — wired together by a CLI script. Unit tests use small synthetic fixtures (never the real 11 GB archive) so the suite stays fast; the real download only happens when the CLI script is run for real in Task 4.

**Tech Stack:** Python 3.13 (conda env `py313`), `requests` for HTTP, `tqdm` for progress, stdlib `zipfile`/`hashlib`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-08-25-cadence-adr-algorithm-design.md` (§4 Data and Knowledge Sources) — this plan implements the corpus-acquisition prerequisite for that section. Executors should read the spec's §4 before starting.

**Source dataset:** "Context Matters" replication package (validated sequential ADR corpus derived from Buchgeher et al.'s 921-repo mining study, IEEE Access, DOI [10.1109/ACCESS.2023.3287654](https://doi.org/10.1109/ACCESS.2023.3287654)), hosted on Zenodo, DOI [10.5281/zenodo.18370195](https://doi.org/10.5281/zenodo.18370195), file `Context Matters.zip`, 460.3 MB compressed / 11.0 GB uncompressed, MD5 `1106da3185ac5ddba0fdfc2f0ace9301`, CC-BY-4.0, freely downloadable with no authentication. Download URL pattern: `https://zenodo.org/records/18370195/files/Context%20Matters.zip?download=1`.

## Global Constraints

- Python 3.13, run via `conda activate py313` (per project convention — no other interpreter).
- No secrets in this plan's code; none are needed (public dataset, no API key).
- The real 11 GB archive and its extracted contents are **never committed to git** — add `data/raw/` and `data/extracted/` to `.gitignore` in Task 1.
- Unit tests must not require network access or the real archive — use small synthetic fixtures built in the test itself.
- Commit after every task (per repo's existing convention of pushing after edits).

---

### Task 1: Project scaffolding and environment

**Files:**
- Create: `environment.yml`
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/data/__init__.py`
- Modify: `.gitignore` (append data directories)
- Test: `tests/test_environment.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: importable `src` and `src.data` packages that later tasks add modules to; a working `py313` conda environment with `requests`, `tqdm`, `pytest` installed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_environment.py
import sys

def test_python_version_is_313_or_newer():
    assert sys.version_info >= (3, 13), (
        f"Expected Python 3.13+, got {sys.version_info.major}.{sys.version_info.minor}"
    )

def test_src_package_importable():
    import src  # noqa: F401
    import src.data  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/test_environment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src'` (packages don't exist yet).

- [ ] **Step 3: Create the packages and environment files**

```python
# src/__init__.py
```

```python
# src/data/__init__.py
```

```python
# tests/__init__.py
```

```python
# tests/data/__init__.py
```

```yaml
# environment.yml
name: py313
channels:
  - conda-forge
dependencies:
  - python=3.13
  - pip
  - pip:
      - requests>=2.32
      - tqdm>=4.66
      - pytest>=8.3
```

```text
# requirements.txt
requests>=2.32
tqdm>=4.66
pytest>=8.3
```

Then run: `conda env update -n py313 -f environment.yml --prune` (or `conda env create -n py313 -f environment.yml` if the env does not exist yet).

- [ ] **Step 4: Append data directories to .gitignore**

Add this block to the end of the existing `.gitignore`:

```gitignore

# ADR corpus (large, downloaded — never commit)
data/raw/
data/extracted/
```

- [ ] **Step 5: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/test_environment.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add environment.yml requirements.txt src/__init__.py src/data/__init__.py tests/__init__.py tests/data/__init__.py tests/test_environment.py .gitignore
git commit -m "chore: scaffold Python package and conda env for corpus acquisition"
git push
```

---

### Task 2: Checksummed downloader

**Files:**
- Create: `src/data/download.py`
- Test: `tests/data/test_download.py`

**Interfaces:**
- Consumes: nothing beyond stdlib + `requests`.
- Produces: `download_file(url: str, dest_path: pathlib.Path, expected_md5: str | None = None, chunk_size: int = 1 << 20) -> pathlib.Path` — streams `url` to `dest_path`, verifies MD5 if given, raises `ChecksumMismatchError` on mismatch (deletes the bad file first), returns `dest_path` on success. Later tasks (Task 4) call this with the Zenodo URL and the MD5 above.

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_download.py
import hashlib
from pathlib import Path

import pytest

from src.data.download import ChecksumMismatchError, download_file


class _FakeResponse:
    def __init__(self, content: bytes):
        self._content = content
        self.headers = {"content-length": str(len(content))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]


def test_download_writes_file_and_returns_path(tmp_path, monkeypatch):
    payload = b"hello adr corpus" * 1000
    monkeypatch.setattr(
        "src.data.download.requests.get",
        lambda url, stream, timeout: _FakeResponse(payload),
    )
    dest = tmp_path / "out.bin"

    result = download_file("http://example.test/file", dest)

    assert result == dest
    assert dest.read_bytes() == payload


def test_download_verifies_correct_checksum(tmp_path, monkeypatch):
    payload = b"consistent bytes"
    correct_md5 = hashlib.md5(payload).hexdigest()
    monkeypatch.setattr(
        "src.data.download.requests.get",
        lambda url, stream, timeout: _FakeResponse(payload),
    )
    dest = tmp_path / "out.bin"

    result = download_file("http://example.test/file", dest, expected_md5=correct_md5)

    assert result == dest


def test_download_raises_and_deletes_file_on_checksum_mismatch(tmp_path, monkeypatch):
    payload = b"tampered bytes"
    monkeypatch.setattr(
        "src.data.download.requests.get",
        lambda url, stream, timeout: _FakeResponse(payload),
    )
    dest = tmp_path / "out.bin"

    with pytest.raises(ChecksumMismatchError):
        download_file("http://example.test/file", dest, expected_md5="0" * 32)

    assert not dest.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/data/test_download.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.download'`.

- [ ] **Step 3: Write the implementation**

```python
# src/data/download.py
"""Streamed, checksum-verified file download."""
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file's MD5 does not match the expected value."""


def download_file(
    url: str,
    dest_path: Path,
    expected_md5: str | None = None,
    chunk_size: int = 1 << 20,
) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    md5 = hashlib.md5()

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest_path.name
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                md5.update(chunk)
                bar.update(len(chunk))

    if expected_md5 is not None and md5.hexdigest() != expected_md5:
        actual = md5.hexdigest()
        dest_path.unlink()
        raise ChecksumMismatchError(
            f"{dest_path}: expected MD5 {expected_md5}, got {actual}"
        )

    return dest_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/data/test_download.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/data/download.py tests/data/test_download.py
git commit -m "feat: add checksum-verified streaming downloader"
git push
```

---

### Task 3: Archive extraction and structural inventory

**Files:**
- Create: `src/data/inventory.py`
- Test: `tests/data/test_inventory.py`

**Interfaces:**
- Consumes: nothing beyond stdlib.
- Produces:
  - `extract_archive(zip_path: pathlib.Path, dest_dir: pathlib.Path) -> pathlib.Path` — extracts `zip_path` into `dest_dir`, returns `dest_dir`.
  - `InventoryReport` — dataclass with fields `total_files: int`, `extension_counts: dict[str, int]`, `top_level_entries: list[str]`.
  - `build_inventory(root_dir: pathlib.Path) -> InventoryReport` — walks `root_dir` and populates the report. Task 4's script calls both and serializes the report to JSON; the next plan (retrieval) is written by reading that JSON.

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_inventory.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/data/test_inventory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.inventory'`.

- [ ] **Step 3: Write the implementation**

```python
# src/data/inventory.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/data/test_inventory.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/data/inventory.py tests/data/test_inventory.py
git commit -m "feat: add archive extraction and structural inventory"
git push
```

---

### Task 4: Wire together the real fetch, run it, and record the inventory

**Files:**
- Create: `scripts/fetch_adr_corpus.py`
- Create: `data/README.md`
- Test: `tests/data/test_fetch_script.py` (tests the script's argument wiring and JSON-serialization logic with mocks; does **not** perform the real 11 GB download)

**Interfaces:**
- Consumes: `download_file` (Task 2), `extract_archive` + `build_inventory` + `InventoryReport` (Task 3).
- Produces: `data/corpus_inventory.json` on disk (git-tracked — it's small, just counts/names) after a real run; `data/raw/` and `data/extracted/` on disk (gitignored, not tracked). The next plan (retrieval indexing) is written by reading `data/corpus_inventory.json` plus manual inspection of `data/extracted/`.

- [ ] **Step 1: Write the failing test for the script's wiring logic**

```python
# tests/data/test_fetch_script.py
import json
from pathlib import Path

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/data/test_fetch_script.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.fetch_adr_corpus'`.

- [ ] **Step 3: Write the script**

```python
# scripts/fetch_adr_corpus.py
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
    raw_dir = data_dir / "raw"
    extracted_dir = data_dir / "extracted"
    zip_path = raw_dir / "context_matters.zip"

    download_file(ZENODO_URL, zip_path, expected_md5=EXPECTED_MD5)
    extract_archive(zip_path, extracted_dir)
    report = build_inventory(extracted_dir)

    report_to_json(report, data_dir / "corpus_inventory.json")
    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    result = run_fetch(data_dir=project_root / "data")
    print(f"Total files: {result.total_files}")
    print(f"Extension counts: {result.extension_counts}")
    print(f"Top-level entries: {result.top_level_entries}")
```

```python
# scripts/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/data/test_fetch_script.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the script for real**

This step performs the actual ~460 MB download (11 GB after extraction) — confirm available disk space (~12 GB free) and a stable connection before running.

Run: `conda run -n py313 python scripts/fetch_adr_corpus.py`
Expected: progress bar completes, MD5 verifies, and the script prints total file count, extension counts, and top-level entries. `data/corpus_inventory.json` now exists.

- [ ] **Step 6: Write data/README.md documenting provenance**

```markdown
# ADR Corpus

Source: "Context Matters" replication package, Zenodo DOI
[10.5281/zenodo.18370195](https://doi.org/10.5281/zenodo.18370195), derived from
Buchgeher, Schöberl, Geist, Dorninger, Haindl, Weinreich, "Using Architecture
Decision Records in Open Source Projects — An MSR Study on GitHub," IEEE Access,
vol. 11, pp. 63725-63740, 2023, DOI
[10.1109/ACCESS.2023.3287654](https://doi.org/10.1109/ACCESS.2023.3287654).
License: CC-BY-4.0.

`raw/` and `extracted/` are gitignored (11 GB) — regenerate with
`python scripts/fetch_adr_corpus.py`. `corpus_inventory.json` is committed and
summarizes the extracted structure (file counts by extension, top-level entries)
for downstream planning.
```

- [ ] **Step 7: Commit**

```bash
git add scripts/fetch_adr_corpus.py scripts/__init__.py tests/data/test_fetch_script.py data/README.md data/corpus_inventory.json
git commit -m "feat: fetch and inventory the ADR corpus from Zenodo"
git push
```

---

## Self-Review Notes

- **Spec coverage:** This plan implements only the corpus-acquisition prerequisite of spec §4. It deliberately does **not** implement ADR parsing into the `ADRRecord` schema, embeddings, or the retrieval index (spec §3 Stage 1) — those require knowing the corpus's actual internal structure, which Task 4 discovers. That follow-on work belongs in a separate plan (`retrieval-indexing`) written after this one completes and `data/corpus_inventory.json` / manual inspection of `data/extracted/` are available.
- **Placeholder scan:** no TBD/TODO; all code blocks are complete and runnable.
- **Type consistency:** `InventoryReport` fields (`total_files`, `extension_counts`, `top_level_entries`) are used identically across Task 3 and Task 4; `download_file` / `extract_archive` / `build_inventory` signatures match between definition (Tasks 2–3) and use (Task 4).
