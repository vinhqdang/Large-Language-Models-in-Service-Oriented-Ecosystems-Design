"""LLM-based qualitative critique (CADENCE Stage 4) — the "separate LLM
pass" spec §3 asks for, distinct from Stage 2's deliberation agents and
Stage 3's synthesizer/repair client.
"""
import re
from dataclasses import dataclass

MIN_SCORE = 0.0
MAX_SCORE = 10.0

_NO_WEAKNESS_PHRASES = {"none", "none identified", "no weakness", "n/a", "na"}


class CritiqueParseError(RuntimeError):
    """Raised when the critique response is missing a parseable score for
    one or more requested quality attributes."""


@dataclass(frozen=True)
class QualitativeScore:
    quality_attribute: str
    score: float
    weakness: str | None


def _qa_word(quality_attribute: str) -> str:
    # QUALITY_ATTRIBUTES is a fixed, known vocabulary (no regex
    # metacharacters, no attribute a prefix of another), so this
    # substitution is safe without full re.escape — see the self-critique
    # plan's Self-Review Notes.
    return quality_attribute.replace("_", "[ _]?")


def _strict_pattern(quality_attribute: str, field: str) -> re.Pattern:
    # The field keyword directly followed by a colon (only markdown
    # decoration tolerated in between, no extra English words) -- tried
    # first so a correctly-formatted answer is always found and used even
    # if the response also contains unrelated prose that happens to
    # mention "<attribute> <field>" earlier (see the tolerant fallback
    # below for why that matters).
    return re.compile(
        rf"^\s*[*_\-\s]*{_qa_word(quality_attribute)}[ _]?{field}[*_\s]*:[*_\s]*(.*)$",
        re.IGNORECASE | re.MULTILINE,
    )


def _tolerant_pattern(quality_attribute: str, field: str) -> re.Pattern:
    # Tolerates a short run of extra descriptor words before the colon
    # (e.g. "Weakness Notes:") -- see PROGRESS.md's environment notes.
    # Deliberately only a FALLBACK, not tried first: this gap is loose
    # enough to also match unrelated prose that happens to start with
    # "<attribute> <field>" (e.g. "Performance Score Analysis: ..."), so
    # it must never take priority over a real, strictly-formatted answer
    # elsewhere in the same response.
    return re.compile(
        rf"^\s*[*_\-\s]*{_qa_word(quality_attribute)}[ _]?{field}[A-Za-z\s]{{0,30}}?[*_\s]*:[*_\s]*(.*)$",
        re.IGNORECASE | re.MULTILINE,
    )


def _find_label_value(response: str, quality_attribute: str, field: str) -> str | None:
    match = _strict_pattern(quality_attribute, field).search(response)
    if match is None:
        match = _tolerant_pattern(quality_attribute, field).search(response)
    return match.group(1).strip() if match else None


_FRACTION_SCORE = re.compile(r"^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$")


def _parse_score(raw: str) -> float | None:
    cleaned = raw.strip().strip("*_ ")

    # A real run showed this is this model's default way of answering a
    # "0-10" prompt -- not a rare edge case, so it's handled directly
    # (scaled to a 0-10 scale) rather than just rejected as unparseable.
    fraction_match = _FRACTION_SCORE.match(cleaned)
    if fraction_match:
        numerator, denominator = float(fraction_match.group(1)), float(fraction_match.group(2))
        if denominator == 0:
            return None
        score = (numerator / denominator) * MAX_SCORE
        return max(MIN_SCORE, min(MAX_SCORE, score))

    try:
        score = float(cleaned)
    except ValueError:
        return None
    return max(MIN_SCORE, min(MAX_SCORE, score))


def _parse_weakness(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = raw.strip().strip("*_ ")
    if not text or text.lower().rstrip(".") in _NO_WEAKNESS_PHRASES:
        return None
    return text


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
        raw_score = _find_label_value(response, qa, "SCORE")
        score = _parse_score(raw_score) if raw_score is not None else None
        if score is None:
            missing.append(qa)
            continue
        weakness = _parse_weakness(_find_label_value(response, qa, "WEAKNESS"))
        scores.append(QualitativeScore(qa, score, weakness))

    if missing:
        raise CritiqueParseError(
            f"Could not parse a score for: {', '.join(missing)}. Response: {response!r}"
        )
    return scores
