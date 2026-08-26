from src.deliberation.knowledge_graph import Tactic
from src.solver.tactic_extraction import extract_mentioned_tactics


def test_extracts_tactic_mentioned_near_verbatim():
    tactics = [Tactic("Caching", "performance", "d", {})]

    result = extract_mentioned_tactics("We will use caching to reduce latency.", tactics)

    assert [t.name for t in result] == ["Caching"]


def test_extracts_tactic_mentioned_with_paraphrase_and_morphology():
    tactics = [
        Tactic("Asynchronous processing via message queues", "performance", "d", {}),
    ]
    text = "leveraging message queuing for asynchronous processing as our primary tactic"

    result = extract_mentioned_tactics(text, tactics)

    assert len(result) == 1
    assert result[0].name == "Asynchronous processing via message queues"


def test_does_not_extract_unrelated_tactic():
    tactics = [Tactic("Read replicas", "scalability", "d", {})]

    result = extract_mentioned_tactics("We should use caching to reduce latency.", tactics)

    assert result == []


def test_extracts_multiple_distinct_tactics_from_one_text():
    tactics = [
        Tactic("Caching", "performance", "d", {}),
        Tactic("Authentication", "security", "d", {}),
        Tactic("Read replicas", "scalability", "d", {}),
    ]
    text = "We will add caching and authentication, but not touch replicas yet."

    result = extract_mentioned_tactics(text, tactics)

    assert {t.name for t in result} == {"Caching", "Authentication"}


def test_empty_text_extracts_nothing():
    tactics = [Tactic("Caching", "performance", "d", {})]

    assert extract_mentioned_tactics("", tactics) == []
