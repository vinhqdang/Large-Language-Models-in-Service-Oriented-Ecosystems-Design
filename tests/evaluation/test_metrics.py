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


class _FakeScorer:
    """A scorer double: proves compute_corpus_metrics reuses whatever
    scorer it's given, rather than constructing its own per call."""

    def __init__(self):
        self.calls = 0

    def score(self, cands, refs, verbose=False):
        import torch
        self.calls += 1
        n = len(cands)
        return torch.zeros(n), torch.zeros(n), torch.full((n,), 0.9)


def test_compute_corpus_metrics_reuses_an_injected_scorer_across_calls():
    """A caller that builds one BERTScorer via load_bertscorer() and
    injects it explicitly should never trigger a second, hidden
    construction -- this is the preferred, directly-testable path (see
    src/evaluation/harness.py, which injects one scorer for a whole
    evaluation run's worth of system reports)."""
    scorer = _FakeScorer()

    compute_corpus_metrics(["a"], ["a"], scorer=scorer)
    compute_corpus_metrics(["b", "c"], ["b", "c"], scorer=scorer)

    assert scorer.calls == 2  # the same injected instance was used both times


def test_compute_corpus_metrics_falls_back_to_one_cached_default_scorer(monkeypatch):
    """Regression: bert_score.score() instantiates and loads a fresh
    ~1.4GB roberta-large model on every call with no caching of its own.
    Calling it once per system report (5x in a typical evaluation run)
    stacked on top of the resident deliberation/critique LLM reliably
    exhausted memory and silently killed a real scaled-evaluation run (an
    OS-level kill -- exit 127, no Python traceback). This is the safety
    net for a caller that omits `scorer=`: it must still only construct
    the default once per process, not once per call."""
    import bert_score

    import src.evaluation.metrics as metrics_module

    monkeypatch.setattr(metrics_module, "_bertscorer", None)

    construction_count = 0

    class _CountingBERTScorer:
        def __init__(self, *args, **kwargs):
            nonlocal construction_count
            construction_count += 1

        def score(self, cands, refs, verbose=False):
            import torch
            n = len(cands)
            return torch.zeros(n), torch.zeros(n), torch.full((n,), 0.9)

    monkeypatch.setattr(bert_score, "BERTScorer", _CountingBERTScorer)

    metrics_module.compute_corpus_metrics(["a"], ["a"])
    metrics_module.compute_corpus_metrics(["b", "c"], ["b", "c"])

    assert construction_count == 1


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
