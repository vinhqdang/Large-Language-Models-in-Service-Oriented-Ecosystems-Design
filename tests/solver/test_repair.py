from src.deliberation.knowledge_graph import Tactic
from src.solver.repair import VerifiedDecision, run_repair_loop


def _catalog():
    return [
        Tactic("Caching", "performance", "d", {}),
        Tactic("Authentication", "security", "d", {}),
        Tactic("Automated regression test suite", "maintainability", "d", {}),
    ]


class _FakeRepairClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, system=None):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_returns_immediately_feasible_without_calling_repair_client():
    client = _FakeRepairClient([])  # would raise IndexError if called

    result = run_repair_loop(
        candidate="We will use caching.",
        rationale="Improves performance.",
        required_quality_attributes=("performance",),
        tactic_budget=2,
        quality_attribute_weights={},
        tactics=_catalog(),
        repair_client=client,
        max_repair_iterations=2,
    )

    assert isinstance(result, VerifiedDecision)
    assert result.is_feasible
    assert result.repair_iterations == 0
    assert result.caveat is None
    assert result.final_candidate == "We will use caching."


def test_repairs_once_and_succeeds():
    client = _FakeRepairClient([
        "CANDIDATE: We will use caching and authentication.\nRATIONALE: Fits the budget.",
    ])

    result = run_repair_loop(
        # Only mentions a performance tactic -- security has zero mentioned
        # supporting tactics, so this is infeasible on the first check.
        candidate="We will use caching.",
        rationale="Improves performance.",
        required_quality_attributes=("performance", "security"),
        tactic_budget=2,
        quality_attribute_weights={},
        tactics=_catalog(),
        repair_client=client,
        max_repair_iterations=2,
    )

    assert result.is_feasible
    assert result.repair_iterations == 1
    assert "budget" in client.prompts[0].lower()
    assert "security" in client.prompts[0]


def test_degrades_gracefully_after_exhausting_repair_attempts():
    # Same unsolvable-within-budget response every time -> never feasible.
    client = _FakeRepairClient([
        "CANDIDATE: We will use caching, authentication, and testing.\nRATIONALE: Still too much.",
        "CANDIDATE: We will use caching, authentication, and testing.\nRATIONALE: Still too much.",
    ])

    result = run_repair_loop(
        candidate="We will use caching, authentication, and an automated regression test suite.",
        rationale="Covers everything.",
        required_quality_attributes=("performance", "security", "maintainability"),
        tactic_budget=1,
        quality_attribute_weights={},
        tactics=_catalog(),
        repair_client=client,
        max_repair_iterations=2,
    )

    assert not result.is_feasible
    assert result.repair_iterations == 2
    assert result.caveat is not None
    assert len(result.uncovered_quality_attributes) == 2
    assert len(result.selected_tactics) == 1


def test_max_repair_iterations_zero_checks_once_without_calling_repair_client():
    client = _FakeRepairClient([])  # would raise IndexError if called

    result = run_repair_loop(
        candidate="We will use nothing in particular.",
        rationale="Vague.",
        required_quality_attributes=("performance", "security"),
        tactic_budget=2,
        quality_attribute_weights={},
        tactics=_catalog(),
        repair_client=client,
        max_repair_iterations=0,
    )

    assert not result.is_feasible
    assert result.repair_iterations == 0
    assert result.caveat is not None


class _RaisingThenSucceedingClient:
    """Raises on its first call (simulating a transient LLM error), then
    succeeds on the next."""

    def __init__(self, success_response):
        self.success_response = success_response
        self.calls = 0

    def generate(self, prompt, system=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient LLM error")
        return self.success_response


def test_recovers_from_a_transient_repair_client_error():
    client = _RaisingThenSucceedingClient(
        "CANDIDATE: We will use caching and authentication.\nRATIONALE: Fits."
    )

    result = run_repair_loop(
        candidate="We will use nothing in particular.",
        rationale="Vague.",
        required_quality_attributes=("performance", "security"),
        tactic_budget=2,
        quality_attribute_weights={},
        tactics=_catalog(),
        repair_client=client,
        max_repair_iterations=2,
    )

    assert result.is_feasible
    assert client.calls == 2


def test_recovers_from_an_unparseable_repair_response():
    client = _FakeRepairClient([
        "I'm not sure what to recommend.",  # unparseable -> CandidateRationaleParseError
        "CANDIDATE: We will use caching and authentication.\nRATIONALE: Fits.",
    ])

    result = run_repair_loop(
        candidate="We will use nothing in particular.",
        rationale="Vague.",
        required_quality_attributes=("performance", "security"),
        tactic_budget=2,
        quality_attribute_weights={},
        tactics=_catalog(),
        repair_client=client,
        max_repair_iterations=2,
    )

    assert result.is_feasible
    assert len(client.prompts) == 2


class _AlwaysRaisingClient:
    def generate(self, prompt, system=None):
        raise RuntimeError("permanent LLM error")


def test_degrades_gracefully_without_crashing_when_repair_client_always_fails():
    result = run_repair_loop(
        candidate="We will use nothing in particular.",
        rationale="Vague.",
        required_quality_attributes=("performance", "security"),
        tactic_budget=2,
        quality_attribute_weights={},
        tactics=_catalog(),
        repair_client=_AlwaysRaisingClient(),
        max_repair_iterations=2,
    )

    assert not result.is_feasible
    assert result.repair_iterations == 2
    assert result.caveat is not None
    # best_result from the original (never-repaired) candidate is preserved
    assert result.final_candidate == "We will use nothing in particular."


def test_keeps_best_attempt_even_if_a_later_repair_is_worse():
    # First repair covers 2/2 required (feasible) -- loop should return
    # immediately without needing a worse second attempt, but this also
    # guards against a regression where a later, worse attempt could
    # overwrite a better one if the loop didn't return early.
    client = _FakeRepairClient([
        "CANDIDATE: We will use caching and authentication.\nRATIONALE: Fits.",
    ])

    result = run_repair_loop(
        candidate="We will use nothing in particular.",
        rationale="Vague.",
        required_quality_attributes=("performance", "security"),
        tactic_budget=2,
        quality_attribute_weights={},
        tactics=_catalog(),
        repair_client=client,
        max_repair_iterations=2,
    )

    assert result.is_feasible
    assert result.repair_iterations == 1
