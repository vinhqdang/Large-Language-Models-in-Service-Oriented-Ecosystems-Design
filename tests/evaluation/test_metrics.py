from src.evaluation.metrics import MetricScores, average_scores, compute_corpus_metrics, ensure_nltk_data

ensure_nltk_data()  # module-level: METEOR needs this before any test runs


def test_compute_corpus_metrics_returns_one_score_per_pair():
    generated = ["The cat sat on the rug.", "We will use caching."]
    reference = ["The cat sat on the mat.", "We will use caching for performance."]

    scores = compute_corpus_metrics(generated, reference)

    assert len(scores) == 2
    for s in scores:
        assert isinstance(s, MetricScores)
        assert 0.0 <= s.bertscore_f1 <= 1.0
        assert 0.0 <= s.bleu <= 100.0
        assert 0.0 <= s.rouge1_f <= 1.0
        assert 0.0 <= s.meteor <= 1.0


def test_identical_text_scores_near_perfect():
    scores = compute_corpus_metrics(["Use read replicas for scaling."], ["Use read replicas for scaling."])

    assert scores[0].bertscore_f1 > 0.99
    assert scores[0].rouge1_f == 1.0
    assert scores[0].meteor > 0.9


def test_completely_unrelated_text_scores_low():
    scores = compute_corpus_metrics(["The weather is sunny today."], ["We will use caching for performance."])

    assert scores[0].rouge1_f < 0.3
    assert scores[0].bleu < 20.0


def test_average_scores_computes_mean_per_field():
    scores = [
        MetricScores(bertscore_f1=0.8, bleu=20.0, rouge1_f=0.5, meteor=0.4),
        MetricScores(bertscore_f1=0.6, bleu=10.0, rouge1_f=0.3, meteor=0.2),
    ]

    avg = average_scores(scores)

    assert avg.bertscore_f1 == 0.7
    assert avg.bleu == 15.0
    assert avg.rouge1_f == 0.4
    assert abs(avg.meteor - 0.3) < 1e-9
