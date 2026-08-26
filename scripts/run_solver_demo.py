"""Real end-to-end demo: Stage 1 (retrieval) -> Stage 2 (deliberation) ->
Stage 3 (constraint-solver verification + repair).

Import order: src.retrieval.embeddings (sentence_transformers) is imported
before anything that triggers torch via the local-HF deliberation client,
per the sentence_transformers-before-torch rule in PROGRESS.md.
"""
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.retrieval.embeddings import embed_texts, load_embedding_model  # noqa: F401  (import before torch)
import numpy as np

from src.deliberation.agent import QualityAttributeAgent
from src.deliberation.knowledge_graph import QUALITY_ATTRIBUTES, TACTICS, build_knowledge_graph
from src.deliberation.llm_client import load_local_hf_client
from src.deliberation.orchestrator import DeliberationOrchestrator
from src.retrieval.records import ADRRecord
from src.retrieval.retriever import Retriever
from src.solver.repair import VerifiedDecision, run_repair_loop

PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
RECORDS_PATH = PROCESSED_DIR / "adr_records.jsonl"
EMBEDDINGS_PATH = PROCESSED_DIR / "adr_embeddings.npy"

SAMPLE_CONTEXT = (
    "Our service-oriented system's order-processing service is experiencing "
    "10x read traffic growth from a new mobile client. Requirements: keep "
    "p99 read latency low, keep the change operable by a small team, and "
    "avoid introducing new categories of security risk."
)


def _load_records(records_path: Path) -> list[ADRRecord]:
    records = []
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            records.append(ADRRecord(**json.loads(line)))
    return records


def run_solver_demo(
    records_path: Path,
    embeddings_path: Path,
    context: str,
    max_rounds: int = 2,
    tactic_budget: int = 4,
    max_repair_iterations: int = 2,
) -> VerifiedDecision:
    records = _load_records(records_path)
    embeddings = np.load(embeddings_path)
    embedding_model = load_embedding_model()
    retriever = Retriever(records, embeddings, embedding_model)
    precedents = retriever.retrieve(context, k=3)

    llm_client = load_local_hf_client()
    graph = build_knowledge_graph(TACTICS)
    agents = [QualityAttributeAgent(qa, llm_client, graph) for qa in QUALITY_ATTRIBUTES]
    orchestrator = DeliberationOrchestrator(agents, llm_client, max_rounds=max_rounds)
    deliberation = orchestrator.deliberate(context, precedents)

    return run_repair_loop(
        candidate=deliberation.converged_candidate,
        rationale=deliberation.rationale,
        required_quality_attributes=QUALITY_ATTRIBUTES,
        tactic_budget=tactic_budget,
        quality_attribute_weights={},
        tactics=TACTICS,
        repair_client=llm_client,
        max_repair_iterations=max_repair_iterations,
    )


if __name__ == "__main__":
    result = run_solver_demo(RECORDS_PATH, EMBEDDINGS_PATH, SAMPLE_CONTEXT)
    print("=== Verified decision ===")
    print(f"feasible: {result.is_feasible}")
    print(f"repair iterations: {result.repair_iterations}")
    print(f"candidate: {result.final_candidate}")
    print(f"rationale: {result.rationale}")
    print(f"selected tactics: {result.selected_tactics}")
    print(f"covered quality attributes: {result.covered_quality_attributes}")
    print(f"uncovered quality attributes: {result.uncovered_quality_attributes}")
    if result.caveat:
        print(f"caveat: {result.caveat}")
