"""Real end-to-end demo: Stage 1 retrieval feeding Stage 2 deliberation.

Import order: src.retrieval.embeddings (which imports sentence_transformers)
is imported before anything that triggers a torch import via the local-HF
deliberation client, per the sentence_transformers-before-torch rule in
PROGRESS.md.
"""
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
from src.retrieval.records import load_records_jsonl
from src.retrieval.retriever import Retriever

PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
RECORDS_PATH = PROCESSED_DIR / "adr_records.jsonl"
EMBEDDINGS_PATH = PROCESSED_DIR / "adr_embeddings.npy"

SAMPLE_CONTEXT = (
    "Our service-oriented system's order-processing service is experiencing "
    "10x read traffic growth from a new mobile client. Requirements: keep "
    "p99 read latency low, keep the change operable by a small team, and "
    "avoid introducing new categories of security risk."
)


def run_demo(records_path: Path, embeddings_path: Path, context: str, max_rounds: int = 3):
    records = load_records_jsonl(records_path)
    embeddings = np.load(embeddings_path)
    embedding_model = load_embedding_model()
    retriever = Retriever(records, embeddings, embedding_model)
    precedents = retriever.retrieve(context, k=3)

    llm_client = load_local_hf_client()
    graph = build_knowledge_graph(TACTICS)
    agents = [QualityAttributeAgent(qa, llm_client, graph) for qa in QUALITY_ATTRIBUTES]
    orchestrator = DeliberationOrchestrator(agents, llm_client, max_rounds=max_rounds)

    return orchestrator.deliberate(context, precedents)


if __name__ == "__main__":
    result = run_demo(RECORDS_PATH, EMBEDDINGS_PATH, SAMPLE_CONTEXT, max_rounds=2)
    print("=== Transcript ===")
    for position in result.transcript:
        print(f"[round {position.round_number} | {position.quality_attribute} | {position.stance}]")
        print(position.content)
        print()
    print("=== Converged candidate ===")
    print(result.converged_candidate)
    print("=== Rationale ===")
    print(result.rationale)
