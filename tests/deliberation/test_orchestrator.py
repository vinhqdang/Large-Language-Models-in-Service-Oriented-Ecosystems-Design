import pytest

from src.deliberation.agent import AgentPosition, QualityAttributeAgent
from src.deliberation.knowledge_graph import TACTICS, build_knowledge_graph
from src.deliberation.orchestrator import DeliberationOrchestrator, SynthesisParseError, _parse_synthesis


class _FakeAgentClient:
    """Every agent call returns a fixed marker so we can trace call counts
    and round numbers without needing distinct per-agent behavior."""

    def __init__(self, marker):
        self.marker = marker
        self.calls = 0

    def generate(self, prompt, system=None):
        self.calls += 1
        return f"{self.marker} round-response"


class _FakeSynthesizerClient:
    def __init__(self):
        self.last_prompt = None

    def generate(self, prompt, system=None):
        self.last_prompt = prompt
        return "CANDIDATE: use read replicas\nRATIONALE: balances performance and cost"


def test_deliberate_runs_one_propose_round_then_critique_rounds():
    graph = build_knowledge_graph(TACTICS)
    clients = {qa: _FakeAgentClient(qa) for qa in ["performance", "security"]}
    agents = [QualityAttributeAgent(qa, client, graph) for qa, client in clients.items()]
    synthesizer = _FakeSynthesizerClient()
    orchestrator = DeliberationOrchestrator(agents, synthesizer, max_rounds=2)

    result = orchestrator.deliberate(context="Some decision context.", precedents=[])

    assert result.rounds_run == 2
    # 2 agents x 2 rounds (1 propose + 1 critique) = 4 positions in the transcript
    assert len(result.transcript) == 4
    assert [p.round_number for p in result.transcript] == [1, 1, 2, 2]
    assert [p.stance for p in result.transcript] == ["propose", "propose", "critique", "critique"]


def test_deliberate_calls_synthesizer_with_full_transcript_and_parses_result():
    graph = build_knowledge_graph(TACTICS)
    agent = QualityAttributeAgent("performance", _FakeAgentClient("performance"), graph)
    synthesizer = _FakeSynthesizerClient()
    orchestrator = DeliberationOrchestrator([agent], synthesizer, max_rounds=1)

    result = orchestrator.deliberate(context="ctx", precedents=[])

    assert "performance round-response" in synthesizer.last_prompt
    assert result.converged_candidate == "use read replicas"
    assert result.rationale == "balances performance and cost"


def test_deliberate_with_max_rounds_one_only_proposes_no_critique():
    graph = build_knowledge_graph(TACTICS)
    client = _FakeAgentClient("performance")
    agent = QualityAttributeAgent("performance", client, graph)
    orchestrator = DeliberationOrchestrator([agent], _FakeSynthesizerClient(), max_rounds=1)

    result = orchestrator.deliberate(context="ctx", precedents=[])

    assert len(result.transcript) == 1
    assert result.transcript[0].stance == "propose"
    assert client.calls == 1


def test_constructor_rejects_agents_with_duplicate_quality_attributes():
    graph = build_knowledge_graph(TACTICS)
    agents = [
        QualityAttributeAgent("performance", _FakeAgentClient("a"), graph),
        QualityAttributeAgent("performance", _FakeAgentClient("b"), graph),
    ]

    with pytest.raises(ValueError, match="distinct quality attributes"):
        DeliberationOrchestrator(agents, _FakeSynthesizerClient(), max_rounds=1)


def test_parse_synthesis_is_tolerant_of_markdown_case_and_multiline_rationale():
    text = (
        "**Candidate:** Use read replicas for scaling reads.\n"
        "- rationale: This balances performance and cost.\n"
        "It also keeps the change operable by a small team."
    )

    candidate, rationale = _parse_synthesis(text)

    assert candidate == "Use read replicas for scaling reads."
    assert rationale == (
        "This balances performance and cost. It also keeps the change operable by a small team."
    )


def test_parse_synthesis_tolerates_extra_words_before_the_colon():
    """Regression: a real local-model run produced '**Candidate Decision:**'
    and '**Rationale:**' (an extra descriptor word), which the original
    strict label pattern failed to parse, crashing the whole pipeline."""
    text = (
        "**Candidate Decision:**\n"
        "Use a message broker for asynchronous processing.\n\n"
        "**Rationale:**\n"
        "This decouples read traffic from the main application server."
    )

    candidate, rationale = _parse_synthesis(text)

    assert candidate == "Use a message broker for asynchronous processing."
    assert rationale == "This decouples read traffic from the main application server."


def test_parse_synthesis_raises_clearly_on_unparseable_output():
    with pytest.raises(SynthesisParseError):
        _parse_synthesis("I think we should probably use read replicas, but I'm not sure.")


def test_deliberate_raises_when_synthesizer_output_is_unparseable():
    graph = build_knowledge_graph(TACTICS)
    agent = QualityAttributeAgent("performance", _FakeAgentClient("performance"), graph)

    class _BadSynthesizerClient:
        def generate(self, prompt, system=None):
            return "no recognizable format here"

    orchestrator = DeliberationOrchestrator([agent], _BadSynthesizerClient(), max_rounds=1)

    with pytest.raises(SynthesisParseError):
        orchestrator.deliberate(context="ctx", precedents=[])


def test_deliberate_retries_synthesis_and_recovers_on_a_later_attempt():
    """Regression: a real run showed the synthesizer sometimes deviates
    from the requested wording entirely (e.g. 'Candidacy:' instead of
    'Candidate:') -- since generation is stochastic, a bounded retry often
    self-corrects rather than requiring the parser to chase every
    possible wording variant."""
    graph = build_knowledge_graph(TACTICS)
    agent = QualityAttributeAgent("performance", _FakeAgentClient("performance"), graph)

    class _FlakySynthesizerClient:
        def __init__(self, responses):
            self.responses = list(responses)
            self.calls = 0

        def generate(self, prompt, system=None):
            self.calls += 1
            return self.responses.pop(0)

    client = _FlakySynthesizerClient([
        "**Candidacy:** use read replicas\n**Rationale:** balances performance and cost",
        "CANDIDATE: use read replicas\nRATIONALE: balances performance and cost",
    ])
    orchestrator = DeliberationOrchestrator([agent], client, max_rounds=1)

    result = orchestrator.deliberate(context="ctx", precedents=[])

    assert result.converged_candidate == "use read replicas"
    assert client.calls == 2


def test_deliberate_raises_after_exhausting_synthesis_retries():
    graph = build_knowledge_graph(TACTICS)
    agent = QualityAttributeAgent("performance", _FakeAgentClient("performance"), graph)

    class _AlwaysBadSynthesizerClient:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, system=None):
            self.calls += 1
            return "no recognizable format here"

    client = _AlwaysBadSynthesizerClient()
    orchestrator = DeliberationOrchestrator([agent], client, max_rounds=1)

    with pytest.raises(SynthesisParseError):
        orchestrator.deliberate(context="ctx", precedents=[])
    assert client.calls == 3  # MAX_SYNTHESIS_ATTEMPTS
