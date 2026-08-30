"""Run the complete, real CADENCE pipeline once on a single decision context
and save the full structured result -- precedents, per-agent deliberation
transcript, solver verdict, and finalized ADR with utility scores -- so the
manuscript can quote a genuine worked example instead of describing the
pipeline only in the abstract.

This is illustrative, not evaluative: the quantitative claims in the paper
come from the held-out evaluation runs (scripts/run_evaluation*.py), not
from this one demo context. Reuses run_cadence_demo's real pipeline
wiring and sample decision context unchanged.

Import order: src.retrieval.embeddings (sentence_transformers) is imported
before anything that triggers torch via the local-HF client, per the
sentence_transformers-before-torch rule in PROGRESS.md.
"""
import dataclasses
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.retrieval.embeddings import embed_texts, load_embedding_model  # noqa: F401  (import before torch)

from scripts.run_cadence_demo import RECORDS_PATH, EMBEDDINGS_PATH, SAMPLE_CONTEXT, run_cadence_demo

RESULTS_PATH = _PROJECT_ROOT / "data" / "processed" / "worked_example.json"


def _adr_to_json(context: str, adr) -> dict:
    return {
        "decision_context": context,
        "precedent_titles": adr.precedent_titles,
        "deliberation_transcript": [dataclasses.asdict(p) for p in adr.deliberation_transcript],
        "selected_tactics": adr.selected_tactics,
        "covered_quality_attributes": adr.covered_quality_attributes,
        "uncovered_quality_attributes": adr.uncovered_quality_attributes,
        "is_feasible": adr.is_feasible,
        "repair_iterations": adr.repair_iterations,
        "solver_caveat": adr.solver_caveat,
        "decision": adr.decision,
        "rationale": adr.rationale,
        "overall_score": adr.overall_score,
        "utility_scores": [dataclasses.asdict(s) for s in adr.utility_scores],
        "residual_weaknesses": adr.residual_weaknesses,
    }


if __name__ == "__main__":
    adr = run_cadence_demo(RECORDS_PATH, EMBEDDINGS_PATH, SAMPLE_CONTEXT)
    result = _adr_to_json(SAMPLE_CONTEXT, adr)
    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote worked example to {RESULTS_PATH}")
    print(json.dumps(result, indent=2))
