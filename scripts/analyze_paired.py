"""Paired statistics for the output of scripts/run_evaluation_paired.py.

Reads data/processed/evaluation_paired_results.json and writes
data/processed/evaluation_paired_stats.json with, per system, the mean of each
metric and a 95% percentile-bootstrap interval; for each adjacent rung of the
ablation ladder, the mean paired difference, its bootstrap interval, a
two-sided Wilcoxon signed-rank p-value and the Holm-adjusted p-value within
each metric; the solver's feasibility and repair statistics per budget; and
output length against the reference.

Usage: python scripts/analyze_paired.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = _PROJECT_ROOT / "data" / "processed"
RESULTS_PATH = PROCESSED / "evaluation_paired_results.json"
STATS_PATH = PROCESSED / "evaluation_paired_stats.json"

METRICS = ("bertscore_f1", "bleu", "rouge1_f", "meteor")
N_BOOT = 10_000
RNG = np.random.default_rng(0)


def boot_ci(values):
    values = np.asarray(values, dtype=float)
    idx = RNG.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def ladder(budgets):
    steps = [("zero_shot", "retrieval_only"), ("retrieval_only", "multiagent_no_solver")]
    for b in budgets:
        steps.append(("multiagent_no_solver", f"cadence_no_critique@B={b}"))
        steps.append((f"cadence_no_critique@B={b}", f"cadence_full@B={b}"))
    steps.append(("zero_shot", f"cadence_full@B={budgets[0]}"))
    return steps


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def main():
    res = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    per_item = res["per_item_metrics"]
    budgets = res["config"]["tactic_budgets"]
    out = {"n_items_ok": res["n_items_ok"], "systems": {}, "paired": {}, "solver": {}, "length": {}}

    for system, rows in per_item.items():
        out["systems"][system] = {"n": len(rows)}
        for m in METRICS:
            vals = [r[m] for r in rows.values()]
            out["systems"][system][m] = {"mean": float(np.mean(vals)), "ci95": boot_ci(vals)}

    for m in METRICS:
        entries = []
        for a, b in ladder(budgets):
            common = sorted(set(per_item[a]) & set(per_item[b]))
            diff = np.array([per_item[b][i][m] - per_item[a][i][m] for i in common])
            p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            entries.append({"from": a, "to": b, "n": len(common), "mean_diff": float(diff.mean()),
                            "ci95": boot_ci(diff), "wins": int((diff > 0).sum()),
                            "losses": int((diff < 0).sum()), "p_wilcoxon": p})
        for e, padj in zip(entries, holm([e["p_wilcoxon"] for e in entries])):
            e["p_holm"] = padj
        out["paired"][m] = entries

    for b, rows in res["solver"].items():
        n_cov = [len(r["covered"]) for r in rows]
        out["solver"][b] = {
            "n": len(rows),
            "feasible_rate": float(np.mean([r["is_feasible"] for r in rows])),
            "mean_repair_iterations": float(np.mean([r["repair_iterations"] for r in rows])),
            "mean_covered_attributes": float(np.mean(n_cov)),
            "covered_histogram": {str(k): n_cov.count(k) for k in range(6)},
            "mean_selected_tactics": float(np.mean([len(r["selected_tactics"]) for r in rows])),
            "critique_failures": sum(1 for r in rows if r["critique_error"]),
        }

    words = res["output_words"]
    ref = np.array(words["reference"], dtype=float)
    for system, w in words.items():
        out["length"][system] = {"median_words": float(np.median(w)), "mean_words": float(np.mean(w))}
    # Does the n-gram penalty track output length? Correlate, per system and
    # item, the length ratio to the reference with ROUGE-1.
    ratios, rouge = [], []
    for system, rows in per_item.items():
        lengths = words[system]
        for (rid, r), n in zip(rows.items(), lengths):
            ratios.append(n / ref[list(per_item["zero_shot"]).index(rid)])
            rouge.append(r["rouge1_f"])
    rho = spearmanr(ratios, rouge)
    out["length"]["spearman_length_ratio_vs_rouge1"] = {"rho": float(rho.statistic), "p": float(rho.pvalue),
                                                        "n": len(ratios)}

    STATS_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.exit(main())
