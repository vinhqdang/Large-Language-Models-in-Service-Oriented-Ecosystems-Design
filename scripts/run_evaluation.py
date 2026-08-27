"""Real evaluation harness run: CADENCE + 3 baselines over a held-out
sample of real ADRs, scored against ground truth (spec §5).

Import order: src.retrieval.embeddings (sentence_transformers) is imported
before anything that triggers torch via the local-HF client, per the
sentence_transformers-before-torch rule in PROGRESS.md.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.retrieval.embeddings import embed_texts, load_embedding_model  # noqa: F401  (import before torch)
import numpy as np

from src.deliberation.knowledge_graph import QUALITY_ATTRIBUTES, TACTICS, build_knowledge_graph
from src.deliberation.llm_client import load_local_hf_client
from src.evaluation.harness import EvaluationReport, run_evaluation
from src.evaluation.held_out_set import sample_test_set
from src.retrieval.records import load_records_jsonl
from src.retrieval.retriever import Retriever

PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
RECORDS_PATH = PROCESSED_DIR / "adr_records.jsonl"
EMBEDDINGS_PATH = PROCESSED_DIR / "adr_embeddings.npy"


def run_evaluation_script(
    records_path: Path,
    embeddings_path: Path,
    n_test_items: int = 3,
    min_length: int = 300,
    k: int = 2,
    max_rounds: int = 1,
    tactic_budget: int = 4,
    max_repair_iterations: int = 2,
) -> EvaluationReport:
    # The retriever must be built over the FULL, unfiltered record set in
    # the same order used to build data/processed/adr_embeddings.npy
    # (see scripts/build_retrieval_index.py) -- embeddings[i] corresponds
    # to all_records[i], so filtering before building the Retriever would
    # silently misalign every row.
    all_records = load_records_jsonl(records_path)
    verified = [r for r in all_records if r.extraction_status == "Verified"]
    test_records = sample_test_set(verified, n=n_test_items, min_length=min_length)

    embeddings = np.load(embeddings_path)
    embedding_model = load_embedding_model()
    retriever = Retriever(all_records, embeddings, embedding_model)

    client = load_local_hf_client()
    graph = build_knowledge_graph(TACTICS)

    return run_evaluation(
        test_records=test_records, retriever=retriever, client=client, graph=graph,
        tactics=TACTICS, quality_attributes=QUALITY_ATTRIBUTES,
        k=k, max_rounds=max_rounds, tactic_budget=tactic_budget,
        max_repair_iterations=max_repair_iterations,
    )


if __name__ == "__main__":
    report = run_evaluation_script(RECORDS_PATH, EMBEDDINGS_PATH)
    print(f"=== Evaluation report ({report.n_items} held-out items) ===")
    for sr in report.system_reports:
        print(f"\n{sr.system_name}:")
        print(f"  BERTScore F1: {sr.average_scores.bertscore_f1:.3f}")
        print(f"  BLEU:         {sr.average_scores.bleu:.2f}")
        print(f"  ROUGE-1 F1:   {sr.average_scores.rouge1_f:.3f}")
        print(f"  METEOR:       {sr.average_scores.meteor:.3f}")
        if sr.feasibility_rate is not None:
            print(f"  Constraint-satisfaction rate: {sr.feasibility_rate:.2f}")
            print(f"  Avg repair iterations:        {sr.average_repair_iterations:.2f}")
