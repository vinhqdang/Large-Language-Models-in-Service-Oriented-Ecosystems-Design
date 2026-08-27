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


def compute_corpus_metrics(generated_texts: list[str], reference_texts: list[str]) -> list[MetricScores]:
    import bert_score
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score

    ensure_nltk_data()

    _, _, bertscore_f1s = bert_score.score(generated_texts, reference_texts, lang="en", verbose=False)

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
