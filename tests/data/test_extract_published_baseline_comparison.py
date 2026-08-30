import json

import numpy as np


class _FakeScorer:
    def score(self, cands, refs, verbose=False):
        import torch
        n = len(cands)
        return torch.zeros(n), torch.zeros(n), torch.full((n,), 0.9)


def test_extract_comparison_recomputes_metrics_from_raw_generation_pairs(tmp_path, monkeypatch):
    from scripts.extract_published_baseline_comparison import extract_comparison

    records_path = tmp_path / "adr_records.jsonl"
    record = {
        "record_id": "myrepo/0001-decision.md", "repo_folder": "myrepo", "repository_url": None,
        "relative_path": "0001-decision.md", "sequence_number": 1, "title": "My Decision",
        "raw_text": "x" * 400, "extraction_status": "Verified",
    }
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    results_root = tmp_path / "Results"
    for exp_dir, sub in [("Baseline", ""), ("RAG_Based", "/K_3")]:
        gen_dir = results_root / exp_dir / "Gemini-2.5-Pro" / f"Generated_ADRs{sub}"
        gen_dir.mkdir(parents=True)
        (gen_dir / "myrepo.json").write_text(
            json.dumps([{"title": "My Decision", "generation": "Use caching.", "ground-truth": "Use caching for speed."}]),
            encoding="utf-8",
        )
        for other_model in ["Gemma3-4b", "GLM-4.6", "Qwen3-235b"]:
            other_dir = results_root / exp_dir / other_model / f"Generated_ADRs{sub}"
            other_dir.mkdir(parents=True)
            (other_dir / "myrepo.json").write_text(
                json.dumps([{"title": "My Decision", "generation": "Use pooling.", "ground-truth": "Use caching for speed."}]),
                encoding="utf-8",
            )

    monkeypatch.setattr(
        "scripts.extract_published_baseline_comparison.sample_test_set",
        lambda records, n, min_length, seed: records,
    )

    comparison = extract_comparison(records_path, results_root, scorer=_FakeScorer())

    assert comparison["held_out_items"] == [
        {"record_id": "myrepo/0001-decision.md", "repo_folder": "myrepo", "title": "My Decision"}
    ]
    baseline_gemini = comparison["results"]["Baseline"]["Gemini-2.5-Pro"]
    assert len(baseline_gemini) == 1
    assert baseline_gemini[0]["dataset_index"] == 0
    scores = baseline_gemini[0]["scores"]
    assert abs(scores["bertscore_f1"] - 0.9) < 1e-4  # from the fake scorer (float32), proves recomputation happened
    assert scores["bleu"] > 0  # real sacrebleu computed from the generation/ground-truth text
    assert set(comparison["results"].keys()) == {"Baseline", "RAG_Based_K3"}
    assert set(comparison["results"]["Baseline"].keys()) == {
        "Gemini-2.5-Pro", "Gemma3-4b", "GLM-4.6", "Qwen3-235b",
    }


def test_extract_comparison_reports_none_scores_for_unmatched_titles(tmp_path, monkeypatch):
    from scripts.extract_published_baseline_comparison import extract_comparison

    records_path = tmp_path / "adr_records.jsonl"
    record = {
        "record_id": "myrepo/0001-decision.md", "repo_folder": "myrepo", "repository_url": None,
        "relative_path": "0001-decision.md", "sequence_number": 1, "title": "My Decision",
        "raw_text": "x" * 400, "extraction_status": "Verified",
    }
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    results_root = tmp_path / "Results"  # no Generated_ADRs files created at all

    monkeypatch.setattr(
        "scripts.extract_published_baseline_comparison.sample_test_set",
        lambda records, n, min_length, seed: records,
    )

    comparison = extract_comparison(records_path, results_root, scorer=_FakeScorer())

    baseline_gemini = comparison["results"]["Baseline"]["Gemini-2.5-Pro"]
    assert baseline_gemini[0]["dataset_index"] is None
    assert baseline_gemini[0]["scores"] is None
