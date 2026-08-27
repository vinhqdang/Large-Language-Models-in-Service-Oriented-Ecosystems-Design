"""Standard generation metrics (spec §5) for comparing a system's output
against a held-out ADR's real body text.
"""
from dataclasses import dataclass

import nltk
import sacrebleu
from rouge_score import rouge_scorer


@dataclass(frozen=True)
class MetricScores:
    bertscore_f1: float
    bleu: float
    rouge1_f: float
    meteor: float


def ensure_nltk_data() -> None:
    for resource in ("wordnet", "punkt_tab", "omw-1.4"):
        nltk.download(resource, quiet=True)


def load_bertscorer():
    """Construct a real BERTScorer (loads the ~1.4GB roberta-large model).
    Callers that run compute_corpus_metrics multiple times in one process
    (e.g. src/evaluation/harness.py, once per system report) should call
    this ONCE and pass the result to every compute_corpus_metrics call via
    `scorer=` -- see that function's docstring for why."""
    import bert_score
    return bert_score.BERTScorer(lang="en")


_bertscorer = None


def _get_bertscorer():
    # Fallback used only when a caller doesn't inject a scorer (see
    # compute_corpus_metrics). bert_score.score(...) instantiates and loads
    # a fresh ~1.4GB roberta-large model on every call, with no caching of
    # its own -- calling it once per system report (5x in a typical
    # evaluation run) stacked on top of the resident deliberation/critique
    # LLM reliably exhausted memory and silently killed the process during
    # a real scaled-evaluation run (an OS-level kill -- exit 127, no Python
    # traceback -- not a raised exception). This process-lifetime cache is
    # the safety net so an evaluation run can never regress into per-call
    # reloading just because a caller forgot to inject a scorer explicitly.
    global _bertscorer
    if _bertscorer is None:
        _bertscorer = load_bertscorer()
    return _bertscorer


def compute_corpus_metrics(
    generated_texts: list[str], reference_texts: list[str], scorer=None
) -> list[MetricScores]:
    """`scorer`: an optional pre-built BERTScorer (see load_bertscorer()).
    Callers making multiple calls in one process should build one with
    load_bertscorer() and pass it explicitly, so the ~1.4GB model loads
    once for the whole run rather than once per call. If omitted, falls
    back to a module-cached default (see _get_bertscorer()) -- the
    fallback exists so a caller that forgets to inject still gets the
    same one-load-per-process behavior, not the original per-call-reload
    bug.
    """
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score

    ensure_nltk_data()
    scorer = scorer or _get_bertscorer()

    _, _, bertscore_f1s = scorer.score(generated_texts, reference_texts, verbose=False)

    rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    scores = []
    for i, (generated, reference) in enumerate(zip(generated_texts, reference_texts)):
        bleu = sacrebleu.sentence_bleu(generated, [reference]).score
        rouge1_f = rouge.score(reference, generated)["rouge1"].fmeasure
        meteor = meteor_score([word_tokenize(reference)], word_tokenize(generated))
        scores.append(
            MetricScores(
                bertscore_f1=bertscore_f1s[i].item(),
                bleu=bleu,
                rouge1_f=rouge1_f,
                meteor=meteor,
            )
        )
    return scores


def average_scores(scores: list[MetricScores]) -> MetricScores:
    n = len(scores)
    return MetricScores(
        bertscore_f1=sum(s.bertscore_f1 for s in scores) / n,
        bleu=sum(s.bleu for s in scores) / n,
        rouge1_f=sum(s.rouge1_f for s in scores) / n,
        meteor=sum(s.meteor for s in scores) / n,
    )
