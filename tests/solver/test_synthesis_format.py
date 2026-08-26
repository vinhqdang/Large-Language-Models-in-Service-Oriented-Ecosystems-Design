import pytest

from src.solver.synthesis_format import CandidateRationaleParseError, parse_candidate_rationale


def test_parses_simple_candidate_and_rationale():
    text = "CANDIDATE: Use read replicas.\nRATIONALE: Balances performance and cost."

    candidate, rationale = parse_candidate_rationale(text)

    assert candidate == "Use read replicas."
    assert rationale == "Balances performance and cost."


def test_tolerant_of_markdown_case_and_multiline_rationale():
    text = (
        "**Candidate:** Use read replicas for scaling reads.\n"
        "- rationale: This balances performance and cost.\n"
        "It also keeps the change operable by a small team."
    )

    candidate, rationale = parse_candidate_rationale(text)

    assert candidate == "Use read replicas for scaling reads."
    assert rationale == (
        "This balances performance and cost. It also keeps the change operable by a small team."
    )


def test_raises_clearly_on_unparseable_text():
    with pytest.raises(CandidateRationaleParseError):
        parse_candidate_rationale("I'm not sure what to recommend here.")
