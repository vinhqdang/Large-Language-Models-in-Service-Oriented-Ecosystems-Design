"""LLM-based qualitative critique (CADENCE Stage 4) — the "separate LLM
pass" spec §3 asks for, distinct from Stage 2's deliberation agents and
Stage 3's synthesizer/repair client.
"""
import re
from dataclasses import dataclass


class CritiqueParseError(RuntimeError):
    """Raised when the critique response is missing a score for one or
    more requested quality attributes."""


@dataclass(frozen=True)
class QualitativeScore:
    quality_attribute: str
    score: float
    weakness: str | None


def _label_pattern(quality_attribute: str, field: str) -> re.Pattern:
    # Tolerant of markdown, case, and a short run of extra descriptor
    # words before the colon -- see PROGRESS.md's environment notes on
    # why this must be built in from the start, not added after a crash.
    qa_word = quality_attribute.replace("_", "[ _]?")
    return re.compile(
        rf"^\s*[*_\-\s]*{qa_word}[ _]?{field}[A-Za-z\s]{{0,30}}?[*_\s]*:[*_\s]*(.*)$",
        re.IGNORECASE | re.MULTILINE,
    )


def run_qualitative_critique(
    decision: str, rationale: str, quality_attributes: tuple[str, ...], client
) -> list[QualitativeScore]:
    fields = "\n".join(
        f"{qa.upper()}_SCORE: <0-10>\n{qa.upper()}_WEAKNESS: <specific weakness, or 'none'>"
        for qa in quality_attributes
    )
    prompt = (
        f"Decision: {decision}\n"
        f"Rationale: {rationale}\n\n"
        "Critique this architectural decision from each quality attribute "
        "perspective below. Score each 0 (fails this attribute) to 10 "
        "(exemplary), and name one specific residual weakness if any exist "
        "(or 'none'). Respond in exactly this format, one block per "
        f"attribute:\n{fields}"
    )
    response = client.generate(prompt)

    scores = []
    missing = []
    for qa in quality_attributes:
        score_match = _label_pattern(qa, "SCORE").search(response)
        weakness_match = _label_pattern(qa, "WEAKNESS").search(response)
        if not score_match:
            missing.append(qa)
            continue
        score = float(score_match.group(1).strip())
        weakness_text = weakness_match.group(1).strip() if weakness_match else ""
        weakness = None if not weakness_text or weakness_text.lower() == "none" else weakness_text
        scores.append(QualitativeScore(qa, score, weakness))

    if missing:
        raise CritiqueParseError(
            f"Could not parse a score for: {', '.join(missing)}. Response: {response!r}"
        )
    return scores
