import pytest

from src.critique.llm_critique import CritiqueParseError, QualitativeScore, run_qualitative_critique


class _FakeClient:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate(self, prompt, system=None):
        self.prompts.append(prompt)
        return self.response


def test_parses_score_and_weakness_per_attribute():
    response = (
        "PERFORMANCE_SCORE: 8\n"
        "PERFORMANCE_WEAKNESS: none\n"
        "SECURITY_SCORE: 6\n"
        "SECURITY_WEAKNESS: Relies on a single authentication factor.\n"
    )
    client = _FakeClient(response)

    result = run_qualitative_critique(
        decision="Use caching and authentication.",
        rationale="Balances performance and security.",
        quality_attributes=("performance", "security"),
        client=client,
    )

    assert result == [
        QualitativeScore("performance", 8.0, None),
        QualitativeScore("security", 6.0, "Relies on a single authentication factor."),
    ]
    assert "Use caching and authentication." in client.prompts[0]


def test_tolerant_of_markdown_case_and_extra_words_before_colon():
    response = (
        "**Performance Score:** 7\n"
        "**Performance Weakness Notes:** none\n"
    )
    client = _FakeClient(response)

    result = run_qualitative_critique(
        decision="d", rationale="r", quality_attributes=("performance",), client=client,
    )

    assert result == [QualitativeScore("performance", 7.0, None)]


def test_tolerates_none_identified_and_other_no_weakness_phrasings():
    """Regression: a real local-model run answered 'None identified'
    instead of the literal 'none' requested, which an exact-match check
    failed to treat as null, silently reporting a fake weakness."""
    response = (
        "SECURITY_SCORE: 9\n"
        "SECURITY_WEAKNESS: None identified\n"
    )
    client = _FakeClient(response)

    result = run_qualitative_critique(
        decision="d", rationale="r", quality_attributes=("security",), client=client,
    )

    assert result == [QualitativeScore("security", 9.0, None)]


def test_prefers_a_strictly_formatted_answer_over_an_earlier_prose_preamble():
    """Regression: a real run's response could plausibly include a prose
    heading like 'Performance Score Analysis: ...' before the actual
    structured answer -- the old single-tier tolerant regex locked onto
    that preamble (crashing on non-numeric text) instead of finding the
    real, strictly-formatted line later in the same response."""
    response = (
        "Performance Score Analysis: The following breaks down each "
        "attribute in detail.\n\n"
        "PERFORMANCE_SCORE: 8\n"
        "PERFORMANCE_WEAKNESS: none\n"
    )
    client = _FakeClient(response)

    result = run_qualitative_critique(
        decision="d", rationale="r", quality_attributes=("performance",), client=client,
    )

    assert result == [QualitativeScore("performance", 8.0, None)]


def test_strips_markdown_wrapped_around_the_score_value():
    response = "**PERFORMANCE_SCORE:** **8**\nPERFORMANCE_WEAKNESS: none\n"
    client = _FakeClient(response)

    result = run_qualitative_critique(
        decision="d", rationale="r", quality_attributes=("performance",), client=client,
    )

    assert result == [QualitativeScore("performance", 8.0, None)]


def test_fraction_score_is_parsed_and_scaled_to_zero_to_ten():
    """Regression: a real run of the full pipeline showed this model's
    DEFAULT way of answering a '0-10' prompt is 'N/10' for every
    attribute, not a rare edge case -- it must be handled directly, not
    just rejected as unparseable."""
    response = "PERFORMANCE_SCORE: 8/10\nPERFORMANCE_WEAKNESS: none\nSECURITY_SCORE: 3/5\nSECURITY_WEAKNESS: none\n"
    client = _FakeClient(response)

    result = run_qualitative_critique(
        decision="d", rationale="r", quality_attributes=("performance", "security"), client=client,
    )

    assert result[0].score == 8.0  # 8/10 -> already on a 0-10 scale
    assert result[1].score == 6.0  # 3/5 scaled to a 0-10 scale


def test_unparseable_score_text_raises_critique_parse_error_not_value_error():
    response = "PERFORMANCE_SCORE: eight out of ten\nPERFORMANCE_WEAKNESS: none\n"
    client = _FakeClient(response)

    with pytest.raises(CritiqueParseError, match="performance"):
        run_qualitative_critique(
            decision="d", rationale="r", quality_attributes=("performance",), client=client,
        )


def test_out_of_range_scores_are_clamped_to_zero_to_ten():
    response = "PERFORMANCE_SCORE: 12\nPERFORMANCE_WEAKNESS: none\nSECURITY_SCORE: -2\nSECURITY_WEAKNESS: none\n"
    client = _FakeClient(response)

    result = run_qualitative_critique(
        decision="d", rationale="r", quality_attributes=("performance", "security"), client=client,
    )

    assert result[0].score == 10.0
    assert result[1].score == 0.0


def test_cost_operability_with_markdown_and_extra_words_via_tolerant_fallback():
    response = "**Cost Operability Score:** 7\n**Cost Operability Weakness Notes:** none\n"
    client = _FakeClient(response)

    result = run_qualitative_critique(
        decision="d", rationale="r", quality_attributes=("cost_operability",), client=client,
    )

    assert result == [QualitativeScore("cost_operability", 7.0, None)]


def test_raises_naming_missing_attributes():
    response = "PERFORMANCE_SCORE: 8\nPERFORMANCE_WEAKNESS: none\n"  # security missing
    client = _FakeClient(response)

    with pytest.raises(CritiqueParseError, match="security"):
        run_qualitative_critique(
            decision="d", rationale="r",
            quality_attributes=("performance", "security"), client=client,
        )
