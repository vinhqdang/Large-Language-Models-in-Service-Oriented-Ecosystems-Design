# ADR Retrieval Indexing Implementation Plan

**Goal:** Parse the real ADR corpus into a structured, compact dataset; embed
it; build a vector index; and expose a `retrieve(query_text, k)` function —
concretely implementing CADENCE Stage 1 (spec §3) for use by the next plan
(multi-agent deliberation).

**Architecture:** A `src/retrieval` package with four independently-testable
layers: parsing (markdown → `ADRRecord`), embedding (text → vector, model
injected for testability), a vector index (nearest-neighbor search over
embeddings), and a `Retriever` that composes the three into the public
Stage-1 interface. A script wires the real corpus through parsing +
embedding once and persists the result to `data/processed/`, so later plans
never need to touch `data/extracted/` (~11 GB) again — it can be deleted
after this plan's Task 2 completes, per the standing note in `data/README.md`.

**Tech Stack:** Python 3.13 (conda env `py313`), `sentence-transformers`
(model `all-MiniLM-L6-v2` — small, fast, CPU/GPU-portable, no CADENCE-specific
tuning needed for Stage 1) for embeddings, `torch` (CUDA) as its backend,
`scikit-learn`'s `NearestNeighbors` (cosine metric) for the vector index —
chosen over adding `faiss` as a new dependency since `sklearn` is already
installed and verified working, and the corpus is small enough (~6,173
vectors) that an exact brute-force/cosine search has no real performance
downside here. `numpy` for the embedding matrix. `pytest`.

**Spec:** `docs/superpowers/specs/2026-08-25-cadence-adr-algorithm-design.md`
(§3 Stage 1, §7 repo layout: `src/retrieval/`).

**Real corpus schema this plan is written against** (see `PROGRESS.md`,
"Session 2026-08-26", for the full inspection — summarized here for
executors who don't want to re-derive it):

- Root: `data/extracted/Context Matters/Data/`.
- `ADRs/{repo_folder}/*.md` — 883 folders, 6,173 `.md` files. One folder
  (`markmap_adr`) mixed in an unnumbered file (`splitting-and-bundling.md`);
  filenames are otherwise inconsistently formatted across repos (`0001-x.md`,
  `0001 - x.md`, `adr-0001 x.md`, `ADR-1_x.md`, 3-digit, 0-indexed, etc.) —
  **do not assume a single filename regex will match everything.**
- `dataset_index.json` — dict keyed by folder name (matches `ADRs/` subfolder
  names), each value has `repository_url`, `folder_path`, `local_folder`
  (`"ADRs/{folder}"`), `files_count`, `status` (one of `Verified`,
  `Doubt (name sequence)`, `Doubt (missing file)`, `Doubt (no repo dir)`,
  `Doubt (file contents)`, `Repo Inaccessible`, `No ADR files listed`,
  `No files downloaded`). Not every `ADRs/` folder is guaranteed to have a
  matching key (treat a missing lookup as `status = "unknown"`, don't
  crash).
- ADR body format is usually Nygard/MADR (`# Title`, `Date:`, `## Status`,
  `## Context`, `## Decision`, `## Consequences`) but not universally, so
  `ADRRecord` only extracts a `title` (best-effort) and keeps the rest as
  `raw_text` — no general section-splitter (see Self-Review Notes for why).

## Global Constraints

- Python 3.13, run via `conda activate py313` (project convention).
- **Import order matters:** any module using both `sentence_transformers`
  and `torch` must `import sentence_transformers` *before* `import torch` —
  the reverse order segfaults on this machine (see `PROGRESS.md`). Put the
  `sentence_transformers` import first and add a one-line comment noting why.
- Unit tests must not require the real corpus, a network connection, or a
  real embedding model download — use small synthetic fixtures and an
  injected fake embedding model (matches the existing `test_download.py` /
  `test_inventory.py` convention of never touching the real 11 GB archive in
  tests).
- Commit after every task; push after every commit (per repo convention).
- `data/processed/` is git-tracked but kept small (JSONL text + a compact
  `.npy` embeddings file, not the raw corpus) — check its size before
  committing; if the embeddings file is unexpectedly large, gitignore it and
  document regeneration instead (decide in Task 3 once the real file size is
  known).

---

### Task 1: `ADRRecord` schema and corpus parser

**Files:**
- Create: `src/retrieval/__init__.py`
- Create: `src/retrieval/records.py`
- Test: `tests/retrieval/__init__.py`, `tests/retrieval/test_records.py`

**Interfaces:**
- Consumes: nothing beyond stdlib (`json`, `re`, `dataclasses`, `pathlib`).
- Produces:
  - `ADRRecord` — frozen dataclass: `record_id: str` (`f"{repo_folder}/{filename}"`),
    `repo_folder: str`, `repository_url: str | None`, `relative_path: str`
    (path relative to the `ADRs/` directory), `sequence_number: int | None`,
    `title: str`, `raw_text: str`, `extraction_status: str`.
  - `parse_adr_folder(folder_path: Path, repo_folder: str, repository_url: str | None, extraction_status: str) -> list[ADRRecord]` —
    parses every `.md` file directly inside `folder_path` into an
    `ADRRecord`.
  - `parse_corpus(data_dir: Path) -> list[ADRRecord]` — given the `Data/`
    directory (containing `ADRs/` and `dataset_index.json`), loads the
    index, walks every subfolder of `ADRs/`, and returns the full list of
    records with `extraction_status` looked up per folder (default
    `"unknown"` if the folder has no matching index entry). Task 2's script
    calls this against the real corpus.

- [ ] **Step 1: Write the failing tests**

```python
# tests/retrieval/test_records.py
import json
from pathlib import Path

from src.retrieval.records import ADRRecord, parse_adr_folder, parse_corpus


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/retrieval/test_records.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.retrieval'`.

- [ ] **Step 3: Write the implementation**

```python
# src/retrieval/__init__.py
```

```python
# tests/retrieval/__init__.py
```

```python
# src/retrieval/records.py
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
_FIRST_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


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
    match = _FIRST_HEADING.search(raw_text)
    if match:
        return match.group(1).strip()
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/retrieval/test_records.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/__init__.py src/retrieval/records.py tests/retrieval/__init__.py tests/retrieval/test_records.py
git commit -m "feat: add ADRRecord schema and lenient corpus parser"
git push
```

---

### Task 2: Build the compact processed dataset from the real corpus

**Files:**
- Create: `scripts/build_adr_dataset.py`
- Test: `tests/data/test_build_adr_dataset_script.py` (mocks `parse_corpus`;
  does not touch the real corpus)
- Modify: `data/README.md` (mark `data/extracted/` as safe to delete)

**Interfaces:**
- Consumes: `ADRRecord`, `parse_corpus` (Task 1).
- Produces: `records_to_jsonl(records: list[ADRRecord], out_path: Path) -> None`;
  `data/processed/adr_records.jsonl` on disk after a real run — one JSON
  object per `ADRRecord`, git-tracked (it's ~6,173 lines of the corpus's
  markdown, no embeddings yet, so still reasonably small — check the actual
  size in Step 5 before committing and gitignore instead if it turns out too
  large). Task 3 reads this file to build embeddings, so later tasks and
  future plans never need `data/extracted/` again.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_build_adr_dataset_script.py
import json
from pathlib import Path

from scripts.build_adr_dataset import records_to_jsonl
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
    from src.retrieval.records import ADRRecord

    fake_records = [
        ADRRecord("r/0001-x.md", "r", None, "0001-x.md", 1, "X", "# X\n", "Verified"),
    ]
    calls = []

    def fake_parse_corpus(data_dir):
        calls.append(data_dir)
        return fake_records

    monkeypatch.setattr("scripts.build_adr_dataset.parse_corpus", fake_parse_corpus)

    from scripts.build_adr_dataset import run_build

    out_path = run_build(data_dir=tmp_path / "Data", processed_dir=tmp_path / "processed")

    assert calls == [tmp_path / "Data"]
    assert out_path.exists()
    assert len(out_path.read_text(encoding="utf-8").splitlines()) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/data/test_build_adr_dataset_script.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.build_adr_dataset'`.

- [ ] **Step 3: Write the script**

```python
# scripts/build_adr_dataset.py
"""Parse the real ADR corpus into a compact, git-trackable JSONL dataset.

Run once after fetching the corpus (scripts/fetch_adr_corpus.py). After this
succeeds, data/extracted/ (~11 GB) is no longer needed and can be deleted —
this script's output is the only thing later plans read from.
"""
import dataclasses
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.retrieval.records import ADRRecord, parse_corpus

CORPUS_DATA_DIR = _PROJECT_ROOT / "data" / "extracted" / "Context Matters" / "Data"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


def records_to_jsonl(records: list[ADRRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(dataclasses.asdict(record)) + "\n")


def run_build(data_dir: Path, processed_dir: Path) -> Path:
    records = parse_corpus(data_dir)
    out_path = processed_dir / "adr_records.jsonl"
    records_to_jsonl(records, out_path)
    return out_path


if __name__ == "__main__":
    result_path = run_build(data_dir=CORPUS_DATA_DIR, processed_dir=PROCESSED_DIR)
    print(f"Wrote processed dataset to {result_path}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/data/test_build_adr_dataset_script.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the script for real, check output size, commit or gitignore accordingly**

Run: `conda run -n py313 python scripts/build_adr_dataset.py`
Expected: prints the output path; `data/processed/adr_records.jsonl` exists
with 6,173 lines (one per `.md` file found in the real corpus per the
schema inspection above).

Check its size (`ls -la data/processed/adr_records.jsonl`). If it's a few
MB (expected — the corpus's `.md` content alone is documented as ~50 MB
total in `data/README.md`'s upstream `Data/README.md`, and this is a subset
of that), commit it directly. If it turns out unexpectedly large, add
`data/processed/adr_records.jsonl` to `.gitignore` instead and document
regeneration in `data/README.md` — decide based on the real number, don't
guess.

- [ ] **Step 6: Update data/README.md**

Add a note that `data/extracted/` has now been fully consumed by
`scripts/build_adr_dataset.py` and can be deleted
(`Remove-Item -Recurse -Force data\extracted`), regenerable via
`scripts/fetch_adr_corpus.py` + `scripts/build_adr_dataset.py` if ever
needed again.

- [ ] **Step 7: Delete data/extracted/ and commit**

```bash
Remove-Item -Recurse -Force data\extracted
git add scripts/build_adr_dataset.py tests/data/test_build_adr_dataset_script.py data/README.md data/processed/adr_records.jsonl
git commit -m "feat: parse real ADR corpus into a compact processed dataset"
git push
```

(If Step 5 decided to gitignore the JSONL instead, adjust the `git add` list
and `.gitignore` accordingly, and skip committing that file.)

---

### Task 3: Embeddings

**Files:**
- Create: `src/retrieval/embeddings.py`
- Test: `tests/retrieval/test_embeddings.py` (uses a fake model — no real
  download in unit tests)

**Interfaces:**
- Consumes: nothing beyond `numpy`; the real `load_embedding_model` consumes
  `sentence_transformers`.
- Produces:
  - `embed_texts(texts: list[str], model) -> np.ndarray` — pure function,
    `model` is any object with an `.encode(list[str]) -> np.ndarray` method
    (dependency injection for testability — this is exactly the
    `sentence_transformers.SentenceTransformer` interface, so the real model
    can be passed directly with no adapter).
  - `load_embedding_model(model_name: str = "all-MiniLM-L6-v2")` — the only
    function that actually imports/loads `sentence_transformers`; not
    covered by a fast unit test (would require a real, slow model
    download), exercised for real in Task 4's end-to-end script instead.

- [ ] **Step 1: Write the failing tests**

```python
# tests/retrieval/test_embeddings.py
import numpy as np

from src.retrieval.embeddings import embed_texts


class _FakeModel:
    """Deterministic stand-in for a real sentence-transformers model:
    embeds each text as its length and character-sum, so we can assert on
    exact output without downloading anything."""

    def encode(self, texts, **kwargs):
        return np.array([[len(t), sum(map(ord, t)) % 997] for t in texts], dtype="float32")


def test_embed_texts_returns_one_vector_per_text():
    model = _FakeModel()
    vectors = embed_texts(["hello", "a longer piece of text"], model)

    assert vectors.shape == (2, 2)
    assert vectors[0][0] == 5


def test_embed_texts_handles_empty_list():
    model = _FakeModel()
    vectors = embed_texts([], model)

    assert vectors.shape == (0,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/retrieval/test_embeddings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.retrieval.embeddings'`.

- [ ] **Step 3: Write the implementation**

```python
# src/retrieval/embeddings.py
"""Text embedding for ADR retrieval.

Import order matters on this machine: `import sentence_transformers` before
`import torch` (importing torch first causes a native segfault on import —
see PROGRESS.md, Session 2026-08-26). load_embedding_model is the only
place torch gets imported transitively, so the sentence_transformers import
below must stay first in this module.
"""
import sentence_transformers  # noqa: F401  (import before torch — see docstring)
import numpy as np

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def embed_texts(texts: list[str], model) -> np.ndarray:
    if not texts:
        return np.array([])
    return np.asarray(model.encode(texts))


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/retrieval/test_embeddings.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Smoke-test the real model loads and encodes (manual, not a pytest)**

Run:
```
conda run -n py313 python -c "from src.retrieval.embeddings import load_embedding_model, embed_texts; m = load_embedding_model(); v = embed_texts(['We will use microservices.'], m); print(v.shape)"
```
Expected: downloads the model on first run (small, ~90 MB), prints a shape
like `(1, 384)`. If this segfaults, re-check the import order note above
before doing anything else — do not skip straight to debugging
sentence-transformers itself.

- [ ] **Step 6: Commit**

```bash
git add src/retrieval/embeddings.py tests/retrieval/test_embeddings.py
git commit -m "feat: add injectable text-embedding wrapper for ADR retrieval"
git push
```

---

### Task 4: Vector index and the Stage-1 Retriever

**Files:**
- Create: `src/retrieval/index.py`
- Create: `src/retrieval/retriever.py`
- Create: `scripts/build_retrieval_index.py`
- Test: `tests/retrieval/test_index.py`, `tests/retrieval/test_retriever.py`,
  `tests/data/test_build_retrieval_index_script.py`

**Interfaces:**
- Consumes: `ADRRecord` (Task 1), `embed_texts`/`load_embedding_model`
  (Task 3), `numpy`, `sklearn.neighbors.NearestNeighbors`.
- Produces:
  - `VectorIndex` — wraps `NearestNeighbors(metric="cosine")`;
    `VectorIndex.build(embeddings: np.ndarray) -> VectorIndex` (classmethod);
    `VectorIndex.query(vector: np.ndarray, k: int) -> list[tuple[int, float]]`
    returning `(row_index, similarity)` pairs sorted by descending
    similarity (`similarity = 1 - cosine_distance`).
  - `Retriever` — `Retriever(records: list[ADRRecord], embeddings: np.ndarray, model)`;
    `Retriever.retrieve(query_text: str, k: int = 5) -> list[ADRRecord]` —
    embeds the query with `model`, queries the index, returns the top-k
    `ADRRecord`s. This **is** CADENCE Stage 1 (spec §3): "Embed decision
    context" + "Vector-search top-k precedent ADRs". The next plan
    (multi-agent deliberation) imports `Retriever` directly.
  - `scripts/build_retrieval_index.py` — real end-to-end script: loads
    `data/processed/adr_records.jsonl` (Task 2), loads the real embedding
    model (Task 3), embeds every record's `raw_text`, saves the embeddings
    matrix to `data/processed/adr_embeddings.npy`, and prints a sanity
    retrieval example.

- [ ] **Step 1: Write the failing tests for `VectorIndex`**

```python
# tests/retrieval/test_index.py
import numpy as np

from src.retrieval.index import VectorIndex


def test_query_returns_closest_vectors_first():
    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.9, 0.1],
    ])
    index = VectorIndex.build(embeddings)

    results = index.query(np.array([1.0, 0.0]), k=2)

    assert [idx for idx, _score in results] == [0, 2]


def test_query_k_larger_than_dataset_returns_all():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    index = VectorIndex.build(embeddings)

    results = index.query(np.array([1.0, 0.0]), k=10)

    assert len(results) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/retrieval/test_index.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.retrieval.index'`.

- [ ] **Step 3: Write `VectorIndex`**

```python
# src/retrieval/index.py
"""Nearest-neighbor vector index over ADR embeddings.

Uses sklearn's NearestNeighbors (cosine metric) rather than a dedicated
vector-search library: the corpus is ~6,000 vectors, small enough that
exact brute-force cosine search has no practical downside, and sklearn is
already a project dependency — no need to add faiss for this scale.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


class VectorIndex:
    def __init__(self, embeddings: np.ndarray, model: NearestNeighbors):
        self._embeddings = embeddings
        self._model = model

    @classmethod
    def build(cls, embeddings: np.ndarray) -> "VectorIndex":
        model = NearestNeighbors(metric="cosine")
        model.fit(embeddings)
        return cls(embeddings, model)

    def query(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, len(self._embeddings))
        distances, indices = self._model.kneighbors(vector.reshape(1, -1), n_neighbors=k)
        similarities = 1 - distances[0]
        return list(zip(indices[0].tolist(), similarities.tolist()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/retrieval/test_index.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Write the failing tests for `Retriever`**

```python
# tests/retrieval/test_retriever.py
import numpy as np

from src.retrieval.records import ADRRecord
from src.retrieval.retriever import Retriever


class _FakeModel:
    """Maps fixed strings to fixed vectors so retrieval order is exact and
    deterministic, with no real embedding model involved."""

    _VECTORS = {
        "use microservices": [1.0, 0.0],
        "use a monolith": [0.0, 1.0],
        "use microservices for scale": [0.9, 0.1],
    }

    def encode(self, texts, **kwargs):
        return np.array([self._VECTORS[t] for t in texts])


def _record(record_id, raw_text):
    return ADRRecord(
        record_id=record_id, repo_folder="r", repository_url=None,
        relative_path=record_id, sequence_number=1, title=raw_text,
        raw_text=raw_text, extraction_status="Verified",
    )


def test_retrieve_returns_top_k_records_by_similarity():
    records = [
        _record("a", "use microservices"),
        _record("b", "use a monolith"),
        _record("c", "use microservices for scale"),
    ]
    model = _FakeModel()
    embeddings = np.array([model._VECTORS[r.raw_text] for r in records])
    retriever = Retriever(records, embeddings, model)

    results = retriever.retrieve("use microservices", k=2)

    assert [r.record_id for r in results] == ["a", "c"]
```

- [ ] **Step 6: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/retrieval/test_retriever.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.retrieval.retriever'`.

- [ ] **Step 7: Write `Retriever`**

```python
# src/retrieval/retriever.py
"""CADENCE Stage 1: embed a decision context, retrieve top-k precedent ADRs."""
from src.retrieval.embeddings import embed_texts
from src.retrieval.index import VectorIndex
from src.retrieval.records import ADRRecord


class Retriever:
    def __init__(self, records: list[ADRRecord], embeddings, model):
        self._records = records
        self._model = model
        self._index = VectorIndex.build(embeddings)

    def retrieve(self, query_text: str, k: int = 5) -> list[ADRRecord]:
        query_vector = embed_texts([query_text], self._model)[0]
        results = self._index.query(query_vector, k)
        return [self._records[idx] for idx, _similarity in results]
```

- [ ] **Step 8: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/retrieval/test_retriever.py -v`
Expected: PASS (1 passed).

- [ ] **Step 9: Write the failing test for the real end-to-end script's wiring**

```python
# tests/data/test_build_retrieval_index_script.py
import json
from pathlib import Path

import numpy as np


def test_run_index_build_saves_embeddings_and_returns_retriever(tmp_path, monkeypatch):
    from scripts.build_retrieval_index import run_index_build

    records_path = tmp_path / "adr_records.jsonl"
    records_path.write_text(
        json.dumps({
            "record_id": "r/0001-x.md", "repo_folder": "r", "repository_url": None,
            "relative_path": "0001-x.md", "sequence_number": 1, "title": "X",
            "raw_text": "use microservices", "extraction_status": "Verified",
        }) + "\n",
        encoding="utf-8",
    )

    class _FakeModel:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr(
        "scripts.build_retrieval_index.load_embedding_model", lambda: _FakeModel()
    )

    embeddings_path = tmp_path / "adr_embeddings.npy"
    retriever = run_index_build(records_path=records_path, embeddings_out_path=embeddings_path)

    assert embeddings_path.exists()
    saved = np.load(embeddings_path)
    assert saved.shape == (1, 2)

    results = retriever.retrieve("anything", k=1)
    assert results[0].record_id == "r/0001-x.md"
```

- [ ] **Step 10: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/data/test_build_retrieval_index_script.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.build_retrieval_index'`.

- [ ] **Step 11: Write the script**

```python
# scripts/build_retrieval_index.py
"""Embed the processed ADR dataset and build the real Stage-1 retrieval index.

Run after scripts/build_adr_dataset.py. Saves the embeddings matrix so this
(slow, one-time) embedding step never needs to be repeated — later code
loads data/processed/adr_embeddings.npy directly instead of re-embedding.
"""
import dataclasses
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.retrieval.embeddings import embed_texts, load_embedding_model
from src.retrieval.records import ADRRecord
from src.retrieval.retriever import Retriever

PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
RECORDS_PATH = PROCESSED_DIR / "adr_records.jsonl"
EMBEDDINGS_PATH = PROCESSED_DIR / "adr_embeddings.npy"


def _load_records(records_path: Path) -> list[ADRRecord]:
    records = []
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            records.append(ADRRecord(**json.loads(line)))
    return records


def run_index_build(records_path: Path, embeddings_out_path: Path) -> Retriever:
    records = _load_records(records_path)
    model = load_embedding_model()
    embeddings = embed_texts([r.raw_text for r in records], model)

    embeddings_out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_out_path, embeddings)

    return Retriever(records, embeddings, model)


if __name__ == "__main__":
    retriever = run_index_build(records_path=RECORDS_PATH, embeddings_out_path=EMBEDDINGS_PATH)
    sample = retriever.retrieve("Should we use a message queue for async processing?", k=3)
    print(f"Indexed and embedded {len(sample)}-of-k sample retrieval:")
    for record in sample:
        print(f"  - [{record.repo_folder}] {record.title}")
```

- [ ] **Step 12: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/data/test_build_retrieval_index_script.py -v`
Expected: PASS (1 passed).

- [ ] **Step 13: Run the full suite, then run the real script**

Run: `conda run -n py313 pytest -q` — expect all tests (old + new) passing.

Run: `conda run -n py313 python scripts/build_retrieval_index.py` — this
embeds all 6,173 real ADR texts (downloads the ~90 MB model on first run,
then runs on GPU per the project's CUDA setup). Expect it to print 3 sample
retrieval results and leave `data/processed/adr_embeddings.npy` on disk.
Check its size — at 384 dims × ~6,173 records × 4 bytes it should be
~9.5 MB; commit it if so (small enough to track directly, and it's exactly
the "slow to regenerate, cheap to store" artifact git is fine with), or
gitignore + document regeneration if it's meaningfully larger than
expected.

- [ ] **Step 14: Commit**

```bash
git add src/retrieval/index.py src/retrieval/retriever.py scripts/build_retrieval_index.py tests/retrieval/test_index.py tests/retrieval/test_retriever.py tests/data/test_build_retrieval_index_script.py data/processed/adr_embeddings.npy
git commit -m "feat: build vector index and Stage-1 Retriever over the ADR corpus"
git push
```

---

## Self-Review Notes

- **Spec coverage:** This plan implements exactly CADENCE Stage 1 (spec §3)
  and nothing else — no deliberation, no solver, no critique. Those are
  separate follow-on plans per `PROGRESS.md`'s pipeline order.
- **Why no general section-splitter for Status/Context/Decision/Consequences:**
  the real corpus (see schema inspection) uses those headings inconsistently
  enough (varying heading depth, missing sections, non-standard templates,
  even files with no headings at all) that a general splitter would either
  silently mis-attribute text in many repos or need enough special-casing to
  become its own risky, hard-to-review plan. `raw_text` always has everything;
  a later plan can extract a specific section with a targeted regex *when it
  actually needs that field*, scoped to what it needs rather than guessed
  upfront.
- **Why embed `raw_text` wholesale rather than a curated subset:** Stage 1's
  job is precedent retrieval by topical/decision similarity, not structured
  extraction — embedding the full ADR text maximizes retrievable signal
  (title, rationale, and consequences all contribute to "is this a relevant
  precedent") without depending on the section-splitter this plan
  deliberately doesn't build.
- **Why `sklearn.neighbors.NearestNeighbors` instead of `faiss`:** confirmed
  during environment setup that `faiss` isn't installed and `sklearn` is —
  at ~6,173 vectors, brute-force cosine search is fast enough that adding a
  new native dependency (with its own Windows-install risk, per this
  project's already-documented `sentence_transformers`/`torch` import-order
  segfault) isn't justified. Revisit only if the corpus grows by orders of
  magnitude.
- **Why `data/extracted/` gets deleted in Task 2, not deferred again:** this
  was already flagged twice (in the corpus-acquisition plan's self-review
  and in `PROGRESS.md`'s "Deferred but tracked") as something to not forget;
  Task 2 Step 7 makes it an explicit, scripted step in *this* plan instead of
  leaving it as a third deferred TODO.
- **Placeholder scan:** no TBD/TODO; all code blocks are complete and
  runnable against the schema this plan documents.
- **Type/interface consistency:** `ADRRecord`'s fields are used identically
  in Task 1 (definition), Task 2 (`dataclasses.asdict` round-trip), Task 4
  (`ADRRecord(**json.loads(line))` reconstruction) — field names and order
  don't matter for the dict-based round-trip, but names must match exactly,
  which they do. `embed_texts(texts, model)` and `VectorIndex.query(vector, k)`
  signatures match between definition (Tasks 3–4) and use (`Retriever`,
  Task 4).
