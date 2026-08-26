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


def test_raises_naming_missing_attributes():
    response = "PERFORMANCE_SCORE: 8\nPERFORMANCE_WEAKNESS: none\n"  # security missing
    client = _FakeClient(response)

    with pytest.raises(CritiqueParseError, match="security"):
        run_qualitative_critique(
            decision="d", rationale="r",
            quality_attributes=("performance", "security"), client=client,
        )
