"""Embed the processed ADR dataset and build the real Stage-1 retrieval index.

Run after scripts/build_adr_dataset.py. Saves the embeddings matrix so this
(slow, one-time) embedding step never needs to be repeated — later code
loads data/processed/adr_embeddings.npy directly instead of re-embedding.
"""
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import order matters on this machine: sentence_transformers before torch
# (see PROGRESS.md, Session 2026-08-26). src.retrieval.embeddings enforces
# this internally, so importing it first here keeps that guarantee.
from src.retrieval.embeddings import embed_texts, load_embedding_model
import numpy as np

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
