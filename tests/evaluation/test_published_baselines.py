import json

from src.evaluation.published_baselines import find_dataset_index, load_generation_pair, mean_scores


def test_find_dataset_index_matches_by_title(tmp_path):
    dataset_path = tmp_path / "repo.json"
    dataset_path.write_text(
        json.dumps([
            {"title": "First decision", "content": "..."},
            {"title": "Second decision", "content": "..."},
        ]),
        encoding="utf-8",
    )

    assert find_dataset_index(dataset_path, "Second decision") == 1
    assert find_dataset_index(dataset_path, "Third decision") is None


def test_find_dataset_index_returns_none_for_missing_file(tmp_path):
    assert find_dataset_index(tmp_path / "missing.json", "Anything") is None


def test_load_generation_pair_returns_generation_and_ground_truth_at_index(tmp_path):
    path = tmp_path / "repo.json"
    path.write_text(
        json.dumps([
            {"title": "First", "generation": "gen A", "ground-truth": "ref A"},
            {"title": "Second", "generation": "gen B", "ground-truth": "ref B"},
        ]),
        encoding="utf-8",
    )

    assert load_generation_pair(path, 1) == ("gen B", "ref B")
    assert load_generation_pair(path, 5) is None  # index out of range, not an error


def test_load_generation_pair_returns_none_for_missing_file(tmp_path):
    assert load_generation_pair(tmp_path / "missing.json", 0) is None


def test_mean_scores_averages_across_matched_items_only():
    per_item = [
        {"scores": {"bert_f1": 0.8, "bleu_avg": 0.1}},
        {"scores": {"bert_f1": 0.6, "bleu_avg": 0.3}},
        {"scores": None},  # unmatched item -- excluded, not treated as 0
    ]

    result = mean_scores(per_item, ("bert_f1", "bleu_avg"))

    assert result["bert_f1"] == 0.7
    assert result["bleu_avg"] == 0.2


def test_mean_scores_returns_none_for_a_metric_no_item_has():
    per_item = [{"scores": {"bert_f1": 0.8}}]

    result = mean_scores(per_item, ("bert_f1", "missing_metric"))

    assert result["bert_f1"] == 0.8
    assert result["missing_metric"] is None
