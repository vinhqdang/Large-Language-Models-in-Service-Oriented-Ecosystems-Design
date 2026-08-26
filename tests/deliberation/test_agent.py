from src.deliberation.agent import AgentPosition, QualityAttributeAgent
from src.deliberation.knowledge_graph import TACTICS, build_knowledge_graph
from src.retrieval.records import ADRRecord


class _FakeClient:
    def __init__(self, response="fake position"):
        self.response = response
        self.calls = []

    def generate(self, prompt, system=None):
        self.calls.append({"prompt": prompt, "system": system})
        return self.response


def _adr(record_id, title, raw_text):
    return ADRRecord(
        record_id=record_id, repo_folder="r", repository_url=None,
        relative_path=record_id, sequence_number=1, title=title,
        raw_text=raw_text, extraction_status="Verified",
    )


def test_quality_attribute_property_exposes_constructor_value():
    graph = build_knowledge_graph(TACTICS)
    agent = QualityAttributeAgent("maintainability", _FakeClient(), graph)

    assert agent.quality_attribute == "maintainability"


def test_propose_includes_quality_attribute_and_own_tactics_in_system_prompt():
    graph = build_knowledge_graph(TACTICS)
    client = _FakeClient()
    agent = QualityAttributeAgent("performance", client, graph)

    position = agent.propose(
        context="We need to handle 10x read traffic growth.",
        precedents=[_adr("r/1.md", "Use read replicas", "# Use read replicas\n...")],
    )

    assert isinstance(position, AgentPosition)
    assert position.quality_attribute == "performance"
    assert position.stance == "propose"
    assert position.round_number == 1
    assert position.content == "fake position"
    assert client.calls[0]["system"] is not None
    assert "performance" in client.calls[0]["system"].lower()
    assert "Caching" in client.calls[0]["system"]  # a real performance tactic name


def test_propose_includes_precedent_titles_in_the_prompt():
    graph = build_knowledge_graph(TACTICS)
    client = _FakeClient()
    agent = QualityAttributeAgent("security", client, graph)

    agent.propose(
        context="Handle sensitive user data.",
        precedents=[_adr("r/1.md", "Use OAuth2", "..."), _adr("r/2.md", "Encrypt at rest", "...")],
    )

    prompt = client.calls[0]["prompt"]
    assert "Use OAuth2" in prompt
    assert "Encrypt at rest" in prompt


def test_propose_with_no_precedents_still_produces_a_position():
    graph = build_knowledge_graph(TACTICS)
    client = _FakeClient()
    agent = QualityAttributeAgent("scalability", client, graph)

    position = agent.propose(context="A greenfield system with no history.", precedents=[])

    assert position.content == "fake position"


def test_critique_references_other_agents_positions_in_the_prompt():
    graph = build_knowledge_graph(TACTICS)
    client = _FakeClient()
    agent = QualityAttributeAgent("cost_operability", client, graph)
    other = AgentPosition(quality_attribute="scalability", round_number=1, stance="propose",
                           content="Shard the database across 4 nodes.")

    position = agent.critique(
        candidate_text="Shard the database across 4 nodes.",
        other_positions=[other],
        round_number=2,
    )

    assert position.round_number == 2
    assert position.stance == "critique"
    assert position.quality_attribute == "cost_operability"
    prompt = client.calls[0]["prompt"]
    assert "Shard the database across 4 nodes." in prompt
    assert "scalability" in prompt
