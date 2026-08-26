"""Match known architectural tactics against free-text deliberation output.

Deliberately a lightweight heuristic (4-char word-stem overlap), not a
learned or exact matcher: the deliberation agents were given the exact
tactic-name vocabulary in their system prompts (see
QualityAttributeAgent._system_prompt), so real LLM output tends to
reference it closely — near-verbatim or lightly paraphrased/re-cased —
which this heuristic tolerates. It will miss tactics referenced only by
unrelated synonyms, and could rarely false-positive on an unrelated tactic
that happens to share enough 4-char word-stem prefixes; both are accepted
trade-offs for a research prototype's text-to-symbol bridge, not something
this plan tries to make perfect.
"""
import re

from src.deliberation.knowledge_graph import Tactic

_STOPWORDS = {"a", "an", "the", "of", "for", "via", "and", "or", "in", "on", "to", "over", "with"}
_STEM_LEN = 4
_MIN_WORD_LEN = 3


def _stem(word: str) -> str:
    return word.lower()[:_STEM_LEN]


def _significant_stems(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", text)
    return {_stem(w) for w in words if len(w) > _MIN_WORD_LEN and w.lower() not in _STOPWORDS}


def extract_mentioned_tactics(text: str, tactics: list[Tactic], threshold: float = 0.6) -> list[Tactic]:
    text_stems = _significant_stems(text)
    if not text_stems:
        return []

    mentioned = []
    for tactic in tactics:
        tactic_stems = _significant_stems(tactic.name)
        if not tactic_stems:
            continue
        overlap = len(tactic_stems & text_stems) / len(tactic_stems)
        if overlap >= threshold:
            mentioned.append(tactic)
    return mentioned
