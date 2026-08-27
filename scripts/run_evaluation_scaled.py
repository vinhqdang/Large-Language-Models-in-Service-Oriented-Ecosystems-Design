"""Scaled real evaluation run for the manuscript's Evaluation section (spec
§5): CADENCE vs. 3 baselines over a larger held-out sample, at two
tactic_budget conditions (one achievable, one deliberately tight), per
PROGRESS.md's "Next step" guidance. Writes results to JSON so the manuscript
can be drafted against them without re-running the (slow) real pipeline.

Import order: src.retrieval.embeddings (sentence_transformers) is imported
before anything that triggers torch via the local-HF client, per the
sentence_transformers-before-torch rule in PROGRESS.md.
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.retrieval.embeddings import embed_texts, load_embedding_model  # noqa: F401  (import before torch)
import numpy as np

from src.deliberation.knowledge_graph import QUALITY_ATTRIBUTES, TACTICS, build_knowledge_graph
from src.deliberation.llm_client import load_local_hf_client
from src.evaluation.harness import EvaluationReport, run_multi_budget_evaluation
from src.evaluation.held_out_set import sample_test_set
from src.retrieval.records import load_records_jsonl
from src.retrieval.retriever import Retriever

PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
RECORDS_PATH = PROCESSED_DIR / "adr_records.jsonl"
EMBEDDINGS_PATH = PROCESSED_DIR / "adr_embeddings.npy"
RESULTS_PATH = PROCESSED_DIR / "evaluation_results_scaled.json"


def run_evaluation_scaled_script(
    records_path: Path,
    embeddings_path: Path,
    n_test_items: int = 15,
    min_length: int = 300,
    k: int = 3,
    max_rounds: int = 2,
    tactic_budgets: tuple[int, ...] = (5, 2),
    max_repair_iterations: int = 2,
) -> dict[int, EvaluationReport]:
    # See scripts/run_evaluation.py for why the Retriever must be built over
    # the full unfiltered record set (row-alignment with adr_embeddings.npy).
    all_records = load_records_jsonl(records_path)
    verified = [r for r in all_records if r.extraction_status == "Verified"]
    test_records = sample_test_set(verified, n=n_test_items, min_length=min_length)

    embeddings = np.load(embeddings_path)
    embedding_model = load_embedding_model()
    retriever = Retriever(all_records, embeddings, embedding_model)

    client = load_local_hf_client()
    graph = build_knowledge_graph(TACTICS)

    return run_multi_budget_evaluation(
        test_records=test_records, retriever=retriever, client=client, graph=graph,
        tactics=TACTICS, quality_attributes=QUALITY_ATTRIBUTES,
        tactic_budgets=tactic_budgets, k=k, max_rounds=max_rounds,
        max_repair_iterations=max_repair_iterations,
    )


def _reports_to_json(reports_by_budget: dict[int, EvaluationReport]) -> dict:
    return {
        str(budget): {
            "n_items": report.n_items,
            "system_reports": [asdict(sr) for sr in report.system_reports],
        }
        for budget, report in reports_by_budget.items()
    }


if __name__ == "__main__":
    reports_by_budget = run_evaluation_scaled_script(RECORDS_PATH, EMBEDDINGS_PATH)

    for budget, report in reports_by_budget.items():
        print(f"\n=== tactic_budget={budget} ({report.n_items} held-out items) ===")
        for sr in report.system_reports:
            print(f"\n{sr.system_name}:")
            print(f"  BERTScore F1: {sr.average_scores.bertscore_f1:.3f}")
            print(f"  BLEU:         {sr.average_scores.bleu:.2f}")
            print(f"  ROUGE-1 F1:   {sr.average_scores.rouge1_f:.3f}")
            print(f"  METEOR:       {sr.average_scores.meteor:.3f}")
            if sr.feasibility_rate is not None:
                print(f"  Constraint-satisfaction rate: {sr.feasibility_rate:.2f}")
                print(f"  Avg repair iterations:        {sr.average_repair_iterations:.2f}")

    RESULTS_PATH.write_text(json.dumps(_reports_to_json(reports_by_budget), indent=2), encoding="utf-8")
    print(f"\nResults written to {RESULTS_PATH}")
