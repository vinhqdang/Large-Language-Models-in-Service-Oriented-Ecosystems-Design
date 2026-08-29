"""Extended scaled evaluation run, two deliberate additions over the
committed `data/processed/evaluation_results_scaled.json` (Table III):

1. max_repair_iterations=2 (the design default) instead of 1, to test
   whether giving Stage 3's repair loop its full budget changes the 0%
   constraint-satisfaction finding.
2. `cadence_no_critique` (Stages 1-3 only, no Stage 4) now participates
   per budget alongside `cadence_full`, properly isolating Stage 4's
   (self-critique) marginal contribution -- something no baseline in
   Table III's system set did (`multiagent_no_solver` removes Stage 3
   *and* 4 together).

Both additions are independent of each other (neither confounds the
other's interpretation) and everything else is held constant: the same
seed=43 held-out sample, the same k=3/max_rounds=2, the same two
tactic_budget conditions (5, 2).

Writes to a SEPARATE results file (not overwriting evaluation_results_scaled.json,
which Table III is built from) and checkpoints after each tactic_budget
condition via `on_budget_complete`, since a real ~48-minute-class run has
no other persistence until this script's very end -- and this
environment has previously exhibited unexplained mid-run terminations
under long real end-to-end runs (see PROGRESS.md's "background-task
duration limit" note). A crash after the first budget still leaves that
budget's real result on disk.

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

from scripts.run_evaluation_scaled import (
    EMBEDDINGS_PATH, RECORDS_PATH, _reports_to_json, run_evaluation_scaled_script,
)

RESULTS_PATH = _PROJECT_ROOT / "data" / "processed" / "evaluation_results_scaled_repair2.json"


def _checkpoint(reports_so_far: dict, budget, report) -> None:
    reports_so_far[budget] = report
    RESULTS_PATH.write_text(
        json.dumps(_reports_to_json(reports_so_far), indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n=== checkpointed tactic_budget={budget} ({report.n_items} items) ===")
    for sr in report.system_reports:
        print(f"{sr.system_name}: BERTScore F1={sr.average_scores.bertscore_f1:.3f}", end="")
        if sr.feasibility_rate is not None:
            print(f", feasibility={sr.feasibility_rate:.2f}, avg_repair_iters={sr.average_repair_iterations:.2f}", end="")
        print()


if __name__ == "__main__":
    reports_so_far: dict = {}
    reports_by_budget = run_evaluation_scaled_script(
        RECORDS_PATH,
        EMBEDDINGS_PATH,
        seed=43,  # same held-out sample as the committed Table III run
        n_test_items=3,  # same N as Table III -- isolate max_repair_iterations only
        k=3,  # same as Table III
        max_rounds=2,  # same as Table III
        tactic_budgets=(5, 2),  # same two conditions as Table III
        max_repair_iterations=2,  # <-- the one changed parameter (Table III used 1)
        on_budget_complete=lambda budget, report: _checkpoint(reports_so_far, budget, report),
    )

    print(f"\nAll budgets complete. Final results written to {RESULTS_PATH}")
