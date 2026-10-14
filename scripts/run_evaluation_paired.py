"""Larger, paired evaluation run.

Differs from scripts/run_evaluation_scaled.py in two ways:

1. Paired ablation. For each held-out item, Stages 1-2 (retrieval and
   deliberation) run once. Multi-Agent's output is that deliberation's
   candidate; Stage 3 then runs on the same candidate once per tactic budget,
   giving Cadence-NC; Stage 4 runs on that same verified decision, giving
   Cadence-Full. Each rung of the ladder therefore differs from the previous
   one by exactly one stage on identical input, and the same deliberation is
   shared across budgets, so a budget effect cannot be confounded with a
   different debate.
2. Per-item persistence. Every item's raw outputs and solver diagnostics are
   appended to a JSONL file as soon as they exist, and a rerun skips items
   already present, so an interrupted run resumes instead of restarting.
   Metrics are computed per item at the end, which allows paired statistics.

Usage: python scripts/run_evaluation_paired.py [n_items] [seed]
"""
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.retrieval.embeddings import load_embedding_model  # noqa: E402  (import before torch)

import numpy as np  # noqa: E402

from src.critique.finalize import finalize_decision  # noqa: E402
from src.deliberation.agent import QualityAttributeAgent  # noqa: E402
from src.deliberation.knowledge_graph import QUALITY_ATTRIBUTES, TACTICS, build_knowledge_graph  # noqa: E402
from src.deliberation.llm_client import load_local_hf_client  # noqa: E402
from src.deliberation.orchestrator import DeliberationOrchestrator  # noqa: E402
from src.evaluation.held_out_set import sample_test_set  # noqa: E402
from src.evaluation.metrics import compute_corpus_metrics, load_bertscorer  # noqa: E402
from src.evaluation.systems import retrieve_excluding_self, run_retrieval_only, run_zero_shot  # noqa: E402
from src.retrieval.records import load_records_jsonl  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402
from src.solver.repair import run_repair_loop  # noqa: E402

PROCESSED = _PROJECT_ROOT / "data" / "processed"
RECORDS_PATH = PROCESSED / "adr_records.jsonl"
EMBEDDINGS_PATH = PROCESSED / "adr_embeddings.npy"
ITEMS_PATH = PROCESSED / "evaluation_paired_items.jsonl"
RESULTS_PATH = PROCESSED / "evaluation_paired_results.json"

K_PRECEDENTS = 3
MAX_ROUNDS = 2
MAX_REPAIR_ITERATIONS = 2
TACTIC_BUDGETS = (5, 7)
EARLIER_SAMPLES = ((42, 3), (43, 3))  # pilot and scaled runs; kept disjoint from this one


def select_items(verified, n, seed):
    earlier = set()
    for s, size in EARLIER_SAMPLES:
        earlier.update(r.record_id for r in sample_test_set(verified, n=size, seed=s))
    pool = [r for r in verified if r.record_id not in earlier]
    return sample_test_set(pool, n=n, seed=seed), sorted(earlier)


def run_item(record, retriever, client, graph):
    out = {"record_id": record.record_id, "title": record.title, "reference": record.raw_text}
    t0 = time.time()
    out["zero_shot"] = run_zero_shot(record.title, client).generated_text
    out["retrieval_only"] = run_retrieval_only(
        record.title, retriever, record.record_id, client, k=K_PRECEDENTS
    ).generated_text

    precedents = retrieve_excluding_self(retriever, record.title, record.record_id, K_PRECEDENTS)
    agents = [QualityAttributeAgent(qa, client, graph) for qa in QUALITY_ATTRIBUTES]
    deliberation = DeliberationOrchestrator(agents, client, max_rounds=MAX_ROUNDS).deliberate(
        record.title, precedents
    )
    out["multiagent_no_solver"] = f"{deliberation.converged_candidate}\n\n{deliberation.rationale}"
    out["precedent_titles"] = [p.title for p in precedents]

    out["budgets"] = {}
    for budget in TACTIC_BUDGETS:
        verified = run_repair_loop(
            candidate=deliberation.converged_candidate,
            rationale=deliberation.rationale,
            required_quality_attributes=QUALITY_ATTRIBUTES,
            tactic_budget=budget,
            quality_attribute_weights={},
            tactics=TACTICS,
            repair_client=client,
            max_repair_iterations=MAX_REPAIR_ITERATIONS,
        )
        entry = {
            "cadence_no_critique": f"{verified.final_candidate}\n\n{verified.rationale}",
            "is_feasible": verified.is_feasible,
            "repair_iterations": verified.repair_iterations,
            "selected_tactics": verified.selected_tactics,
            "covered": verified.covered_quality_attributes,
            "uncovered": verified.uncovered_quality_attributes,
        }
        try:
            final = finalize_decision(
                verified_decision=verified, deliberation_result=deliberation, precedents=precedents,
                quality_attributes=QUALITY_ATTRIBUTES, tactics=TACTICS, critique_client=client,
            )
            entry["cadence_full"] = f"{final.decision}\n\n{final.rationale}"
            entry["overall_utility"] = final.overall_score
            entry["utility_scores"] = [asdict(u) for u in final.utility_scores]
        except Exception as exc:  # critique parse failure after its own retries
            entry["cadence_full"] = None
            entry["critique_error"] = repr(exc)
        out["budgets"][str(budget)] = entry
    out["seconds"] = round(time.time() - t0, 1)
    return out


def generate(n, seed):
    all_records = load_records_jsonl(RECORDS_PATH)
    verified = [r for r in all_records if r.extraction_status == "Verified"]
    items, earlier = select_items(verified, n, seed)
    done = set()
    if ITEMS_PATH.exists():
        done = {json.loads(line)["record_id"] for line in ITEMS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()}
    retriever = Retriever(all_records, np.load(EMBEDDINGS_PATH), load_embedding_model())
    client = load_local_hf_client()
    graph = build_knowledge_graph(TACTICS)
    for i, record in enumerate(items, 1):
        if record.record_id in done:
            continue
        try:
            row = run_item(record, retriever, client, graph)
        except Exception as exc:
            row = {"record_id": record.record_id, "title": record.title, "error": repr(exc),
                   "traceback": traceback.format_exc()}
        with ITEMS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[{i}/{len(items)}] {record.record_id} {row.get('seconds', 'ERROR')}s", flush=True)
    return items, earlier


SYSTEMS = ("zero_shot", "retrieval_only", "multiagent_no_solver", "cadence_no_critique", "cadence_full")


def with_consequences(entry):
    """Stage 4 does not rewrite the decision: cadence_full carries the same
    text as cadence_no_critique plus utility scores and residual weaknesses.
    This supplementary variant renders those weaknesses as an ADR
    Consequences section, so the text metrics can see what Stage 4 adds."""
    if not entry["cadence_full"]:
        return None
    weaknesses = [u["weakness"] for u in entry["utility_scores"] if u.get("weakness")]
    if not weaknesses:
        return entry["cadence_full"]
    return entry["cadence_full"] + "\n\nConsequences:\n" + "\n".join(f"- {w}" for w in weaknesses)


def score(n, seed, earlier):
    rows = [json.loads(l) for l in ITEMS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in rows if "error" not in r]
    scorer = load_bertscorer()
    per_item = {}
    for system in ("zero_shot", "retrieval_only", "multiagent_no_solver"):
        scores = compute_corpus_metrics([r[system] for r in ok], [r["reference"] for r in ok], scorer)
        per_item[system] = {r["record_id"]: asdict(s) for r, s in zip(ok, scores)}
    for budget in map(str, TACTIC_BUDGETS):
        for system in ("cadence_no_critique", "cadence_full"):
            usable = [r for r in ok if r["budgets"][budget][system]]
            scores = compute_corpus_metrics(
                [r["budgets"][budget][system] for r in usable], [r["reference"] for r in usable], scorer
            )
            per_item[f"{system}@B={budget}"] = {r["record_id"]: asdict(s) for r, s in zip(usable, scores)}
        usable = [r for r in ok if with_consequences(r["budgets"][budget])]
        scores = compute_corpus_metrics(
            [with_consequences(r["budgets"][budget]) for r in usable], [r["reference"] for r in usable], scorer
        )
        per_item[f"cadence_full_consequences@B={budget}"] = {r["record_id"]: asdict(s) for r, s in zip(usable, scores)}
    result = {
        "config": {"n_requested": n, "seed": seed, "k": K_PRECEDENTS, "max_rounds": MAX_ROUNDS,
                   "max_repair_iterations": MAX_REPAIR_ITERATIONS, "tactic_budgets": list(TACTIC_BUDGETS),
                   "excluded_earlier_items": earlier},
        "n_items_run": len(rows), "n_items_ok": len(ok),
        "failed_items": [{"record_id": r["record_id"], "error": r["error"]} for r in rows if "error" in r],
        "per_item_metrics": per_item,
        "solver": {
            budget: [{"record_id": r["record_id"], **{k: r["budgets"][budget][k] for k in
                      ("is_feasible", "repair_iterations", "selected_tactics", "covered", "uncovered")},
                      "critique_error": r["budgets"][budget].get("critique_error")} for r in ok]
            for budget in map(str, TACTIC_BUDGETS)
        },
        "output_words": {
            s: [len(r[s].split()) for r in ok] for s in ("zero_shot", "retrieval_only", "multiagent_no_solver")
        } | {
            f"{s}@B={b}": [len(r["budgets"][str(b)][s].split()) for r in ok if r["budgets"][str(b)][s]]
            for b in TACTIC_BUDGETS for s in ("cadence_no_critique", "cadence_full")
        } | {
            f"cadence_full_consequences@B={b}": [len(with_consequences(r["budgets"][str(b)]).split()) for r in ok
                                                 if with_consequences(r["budgets"][str(b)])]
            for b in TACTIC_BUDGETS
        } | {"reference": [len(r["reference"].split()) for r in ok]},
    }
    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"scored {len(ok)} items -> {RESULTS_PATH}")


if __name__ == "__main__":
    n_items = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 44
    _, earlier_ids = generate(n_items, seed)
    score(n_items, seed, earlier_ids)
