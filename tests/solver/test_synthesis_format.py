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


def test_tolerates_extra_words_before_the_colon():
    """Regression: a real local-model repair response used 'Candidate
    Decision:' (an extra descriptor word), which the original strict
    label pattern failed to parse — see the same fix applied to
    src/deliberation/orchestrator.py's sibling parser."""
    text = (
        "**Candidate Decision:**\n"
        "Use a message broker for asynchronous processing.\n\n"
        "**Rationale:**\n"
        "This decouples read traffic from the main application server."
    )

    candidate, rationale = parse_candidate_rationale(text)

    assert candidate == "Use a message broker for asynchronous processing."
    assert rationale == "This decouples read traffic from the main application server."


def test_raises_clearly_on_unparseable_text():
    with pytest.raises(CandidateRationaleParseError):
        parse_candidate_rationale("I'm not sure what to recommend here.")
