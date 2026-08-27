import numpy as np
import pytest

from src.deliberation.knowledge_graph import TACTICS, build_knowledge_graph, QUALITY_ATTRIBUTES
from src.retrieval.records import ADRRecord
from src.retrieval.retriever import Retriever
from src.evaluation.systems import (
    SystemOutput, retrieve_excluding_self, run_zero_shot, run_retrieval_only,
    run_multiagent_no_solver, run_cadence_full, run_cadence_no_critique,
)


class _FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0] for _ in texts])


def _make_retriever(record_ids):
    records = [
        ADRRecord(record_id=rid, repo_folder="r", repository_url=None, relative_path=rid,
                   sequence_number=1, title=f"Title {rid}", raw_text=f"text {rid}", extraction_status="Verified")
        for rid in record_ids
    ]
    embeddings = np.array([[1.0, 0.0] for _ in records])
    return Retriever(records, embeddings, _FakeEmbeddingModel())


class _FakeClient:
    def __init__(self, response="generated text"):
        self.response = response
        self.prompts = []

    def generate(self, prompt, system=None):
        self.prompts.append(prompt)
        return self.response


def test_retrieve_excluding_self_never_returns_the_excluded_record():
    retriever = _make_retriever(["self", "a", "b", "c"])

    results = retrieve_excluding_self(retriever, "query", exclude_record_id="self", k=2)

    assert "self" not in [r.record_id for r in results]
    assert len(results) == 2


def test_retrieve_excluding_self_warns_when_fewer_than_k_precedents_remain():
    retriever = _make_retriever(["self", "a"])  # only 1 non-self record exists

    with pytest.warns(UserWarning, match="only 1 precedents available"):
        results = retrieve_excluding_self(retriever, "query", exclude_record_id="self", k=3)

    assert [r.record_id for r in results] == ["a"]


def test_run_zero_shot_calls_client_once_with_only_the_context():
    client = _FakeClient("Use caching.")

    result = run_zero_shot("Handle high read traffic.", client)

    assert isinstance(result, SystemOutput)
    assert result.system_name == "zero_shot"
    assert result.generated_text == "Use caching."
    assert result.is_feasible is None
    assert "Handle high read traffic." in client.prompts[0]


def test_run_retrieval_only_includes_precedents_in_the_prompt():
    retriever = _make_retriever(["self", "a"])
    client = _FakeClient("Use read replicas.")

    result = run_retrieval_only("ctx", retriever, exclude_record_id="self", client=client, k=1)

    assert result.system_name == "retrieval_only"
    assert "Title a" in client.prompts[0]


def test_run_multiagent_no_solver_returns_deliberation_output_with_no_feasibility():
    retriever = _make_retriever(["self", "a"])
    graph = build_knowledge_graph(TACTICS)

    class _SynthesisClient(_FakeClient):
        def generate(self, prompt, system=None):
            self.prompts.append(prompt)
            return "CANDIDATE: Use caching.\nRATIONALE: Improves performance."

    client = _SynthesisClient()
    result = run_multiagent_no_solver(
        "ctx", retriever, exclude_record_id="self", client=client, graph=graph,
        quality_attributes=("performance",), k=1, max_rounds=1,
    )

    assert result.system_name == "multiagent_no_solver"
    assert "Use caching." in result.generated_text
    assert result.is_feasible is None
    assert result.repair_iterations is None


def test_run_cadence_full_returns_feasibility_and_repair_iterations():
    retriever = _make_retriever(["self", "a"])
    graph = build_knowledge_graph(TACTICS)

    class _AllPurposeClient(_FakeClient):
        def generate(self, prompt, system=None):
            self.prompts.append(prompt)
            if "SCORE" in prompt:
                return "\n".join(
                    f"{qa.upper()}_SCORE: 7\n{qa.upper()}_WEAKNESS: none" for qa in QUALITY_ATTRIBUTES
                )
            return "CANDIDATE: Use caching.\nRATIONALE: Improves performance."

    client = _AllPurposeClient()
    result = run_cadence_full(
        "ctx", retriever, exclude_record_id="self", client=client, graph=graph, tactics=TACTICS,
        quality_attributes=QUALITY_ATTRIBUTES, k=1, max_rounds=1, tactic_budget=2, max_repair_iterations=1,
    )

    assert result.system_name == "cadence_full"
    assert result.is_feasible is not None
    assert result.repair_iterations is not None
    assert "Use caching." in result.generated_text


def test_run_cadence_no_critique_returns_feasibility_without_a_critique_call(monkeypatch):
    import src.evaluation.systems as systems_module

    retriever = _make_retriever(["self", "a"])
    graph = build_knowledge_graph(TACTICS)
    client = _FakeClient("CANDIDATE: Use caching.\nRATIONALE: Improves performance.")

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("finalize_decision (Stage 4) must not be called by run_cadence_no_critique")

    monkeypatch.setattr(systems_module, "finalize_decision", _must_not_be_called)

    result = run_cadence_no_critique(
        "ctx", retriever, exclude_record_id="self", client=client, graph=graph, tactics=TACTICS,
        quality_attributes=QUALITY_ATTRIBUTES, k=1, max_rounds=1, tactic_budget=2, max_repair_iterations=1,
    )

    assert result.system_name == "cadence_no_critique"
    assert result.is_feasible is not None
    assert result.repair_iterations is not None
    assert "Use caching." in result.generated_text
