# Multi-Agent Deliberation Implementation Plan

**Goal:** Implement CADENCE Stage 2 (spec §3): given a decision context and
the precedent ADRs retrieved by Stage 1 (`src/retrieval/retriever.py`), run
N quality-attribute advocate agents through bounded-round structured
argumentation, grounded in a knowledge graph of architectural tactics, and
produce a converged candidate decision + rationale + a full transcript
(provenance for the eventual manuscript and for Stage 3's solver input).

**Architecture:** A `src/deliberation` package with four layers, each
independently testable: a knowledge graph (pure data + `networkx`, no
LLM), an LLM-client abstraction (`generate(prompt, system) -> str`,
dependency-injected — mirrors the `embed_texts(texts, model)` pattern
already used in `src/retrieval/embeddings.py`), a `QualityAttributeAgent`
(one per ISO/IEC 25010 concern, uses an injected client + its own KG
tactics), and a `DeliberationOrchestrator` that runs the bounded-round
protocol and a `Synthesizer` that produces the final converged text. A
script wires Stage 1 + Stage 2 together for a real, observable run.

**Tech Stack:** Python 3.13 (conda env `py313`), `networkx` (already
installed) for the knowledge graph, `google-genai` (installed this
session) for the primary Gemini backend, `transformers`+`torch` (already
installed) for the local open-weight secondary backend, `pytest`.

**Spec:** `docs/superpowers/specs/2026-08-25-cadence-adr-algorithm-design.md`
(§3 Stage 2, §4 quality-attribute taxonomy + knowledge graph, §6 LLM
backend, §9 open risk: "knowledge graph construction method to be detailed
in the implementation plan" — that detail is below).

**Quality attributes (fixed set, per spec §4):** `performance`, `security`,
`maintainability`, `scalability`, `cost_operability` — one agent per
attribute, five agents total.

**Knowledge graph construction method** (resolving spec §9's open risk):
a curated catalog of ~25 architectural tactics, each tagged with the
quality attribute it primarily supports and 0–2 trade-off relationships to
*other* quality attributes it measurably worsens. This is an original,
short, structured synthesis of tactics that are common, widely-taught
software-architecture knowledge (caching, load balancing, circuit
breakers, rate limiting, horizontal scaling, etc. — the same *category* of
concept the Bass/Clements/Kazman tactics catalog and SEI technical reports
describe, cited in the manuscript as prior art for the general idea, but
not reproducing their specific text) rather than a scraped or copied
source. It is intentionally small and hand-curated, not mined from a
corpus — the ADR corpus (Stage 1) already supplies real-world grounding;
the knowledge graph's job is to supply the *structured trade-off relations*
between tactics and quality attributes that free-text ADRs don't encode
explicitly. Represented as a `networkx.DiGraph`: quality-attribute names
and tactic names as nodes, `supports` edges (tactic → its primary QA) and
`trade_off` edges (tactic → another QA it worsens, with a `note` describing
the effect) — chosen over a bespoke graph structure since `networkx` is
already an installed dependency and provides free traversal/query
utilities if a later plan needs them (e.g. Stage 3's solver encoding).

## Global Constraints

- Python 3.13, run via `conda activate py313` (project convention).
- **Import order:** any module importing both `sentence_transformers` and
  `torch` must import `sentence_transformers` first (see `PROGRESS.md`).
  This plan's local-HF client only imports `torch`/`transformers` lazily
  *inside* its loader function, and never imports `sentence_transformers`
  itself — so it's only a concern for the Task 4 end-to-end script, which
  combines Stage 1 (retrieval) and Stage 2 (this plan) in one process and
  must import `src.retrieval.embeddings` (or `sentence_transformers`
  directly) before triggering the local-HF loader.
- Unit tests must not call a real LLM API or load a real local model —
  every LLM-consuming class takes its client via dependency injection, and
  tests use a fake client. Real model loading/generation is exercised only
  in the explicit "run for real" steps below, using the free, no-API-key
  local backend (no `GEMINI_API_KEY` is configured on this machine as of
  this plan — see `PROGRESS.md`). **The `GeminiClient` implementation in
  Task 2 is therefore unit-tested against a mocked SDK call but has NOT
  been exercised against the real Gemini API in this session — verify it
  with a real key before relying on it for evaluation runs.**
- Commit after every task; push after every commit.
- Use a small, fast local instruct model (`Qwen2.5-1.5B-Instruct`) for this
  plan's real end-to-end verification — the point here is proving the
  *protocol* works, not picking the final paper-quality model. Model
  choice for actual evaluation runs is a decision for the evaluation-
  harness plan, not this one.

---

### Task 1: Knowledge graph of architectural tactics

**Files:**
- Create: `src/deliberation/__init__.py`
- Create: `src/deliberation/knowledge_graph.py`
- Create: `tests/deliberation/__init__.py`
- Create: `tests/deliberation/test_knowledge_graph.py`

**Interfaces:**
- Consumes: `networkx`.
- Produces:
  - `QUALITY_ATTRIBUTES: tuple[str, ...]` — the fixed 5-tuple above.
  - `Tactic` — frozen dataclass: `name: str`, `category: str` (one of
    `QUALITY_ATTRIBUTES` — the QA it primarily supports),
    `description: str`, `trade_offs: dict[str, str]` (other QA name →
    effect note).
  - `TACTICS: list[Tactic]` — the curated catalog (~25 entries, ≥3 per
    quality attribute), module-level constant.
  - `build_knowledge_graph(tactics: list[Tactic]) -> networkx.DiGraph`.
  - `supporting_tactics_for(graph, quality_attribute: str) -> list[str]`.
  - `trade_offs_for_tactic(graph, tactic_name: str) -> dict[str, str]`.
  - Task 3's agents call `supporting_tactics_for`/`trade_offs_for_tactic`
    against a graph built from `TACTICS`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/deliberation/test_knowledge_graph.py
from src.deliberation.knowledge_graph import (
    QUALITY_ATTRIBUTES,
    TACTICS,
    Tactic,
    build_knowledge_graph,
    supporting_tactics_for,
    trade_offs_for_tactic,
)


def test_build_knowledge_graph_contains_all_quality_attribute_nodes():
    graph = build_knowledge_graph(TACTICS)

    for qa in QUALITY_ATTRIBUTES:
        assert qa in graph.nodes


def test_supporting_tactics_for_returns_only_tactics_with_a_supports_edge():
    tactics = [
        Tactic(name="Caching", category="performance", description="d",
               trade_offs={"maintainability": "cache invalidation complexity"}),
        Tactic(name="Authentication", category="security", description="d",
               trade_offs={}),
    ]
    graph = build_knowledge_graph(tactics)

    assert supporting_tactics_for(graph, "performance") == ["Caching"]
    assert supporting_tactics_for(graph, "security") == ["Authentication"]
    assert supporting_tactics_for(graph, "scalability") == []


def test_trade_offs_for_tactic_returns_trade_off_edges_with_notes():
    tactics = [
        Tactic(name="Caching", category="performance", description="d",
               trade_offs={"maintainability": "cache invalidation complexity"}),
    ]
    graph = build_knowledge_graph(tactics)

    assert trade_offs_for_tactic(graph, "Caching") == {
        "maintainability": "cache invalidation complexity"
    }


def test_trade_offs_for_tactic_with_no_trade_offs_returns_empty_dict():
    tactics = [Tactic(name="X", category="performance", description="d", trade_offs={})]
    graph = build_knowledge_graph(tactics)

    assert trade_offs_for_tactic(graph, "X") == {}


def test_default_tactics_catalog_covers_every_quality_attribute_meaningfully():
    graph = build_knowledge_graph(TACTICS)

    for qa in QUALITY_ATTRIBUTES:
        supporting = supporting_tactics_for(graph, qa)
        assert len(supporting) >= 3, f"{qa} has too few supporting tactics: {supporting}"


def test_default_tactics_catalog_has_no_self_trade_offs():
    for tactic in TACTICS:
        assert tactic.category not in tactic.trade_offs, (
            f"{tactic.name} lists a trade-off against its own category {tactic.category}"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/deliberation/test_knowledge_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.deliberation'`.

- [ ] **Step 3: Write the implementation**

```python
# src/deliberation/__init__.py
```

```python
# tests/deliberation/__init__.py
```

```python
# src/deliberation/knowledge_graph.py
"""Knowledge graph of architectural tactics linked to quality attributes.

A small, hand-curated catalog (not mined from a corpus — Stage 1's ADR
corpus already supplies real-world case grounding; this graph's job is the
structured tactic -> quality-attribute trade-off relations that free-text
ADRs don't encode explicitly). The tactics themselves describe widely
documented, common architectural concepts (caching, load balancing, rate
limiting, circuit breakers, etc.) in original wording — see the
multi-agent-deliberation plan's header for the construction-method
rationale and citation note.
"""
from dataclasses import dataclass, field

import networkx as nx

QUALITY_ATTRIBUTES = (
    "performance",
    "security",
    "maintainability",
    "scalability",
    "cost_operability",
)


@dataclass(frozen=True)
class Tactic:
    name: str
    category: str
    description: str
    trade_offs: dict[str, str] = field(default_factory=dict)


TACTICS: list[Tactic] = [
    # --- performance ---
    Tactic("Caching", "performance",
           "Store frequently-accessed results to avoid recomputation or repeated I/O.",
           {"maintainability": "cache invalidation and staleness become an ongoing concern"}),
    Tactic("Connection pooling", "performance",
           "Reuse expensive connections (DB, network) instead of opening one per request.",
           {"maintainability": "pool sizing and lifecycle tuning adds operational complexity"}),
    Tactic("Asynchronous processing via message queues", "performance",
           "Decouple slow work from the request path via a queue and background workers.",
           {"maintainability": "harder to trace an end-to-end request across queue boundaries",
            "security": "the queue and its consumers become additional attack surface"}),
    Tactic("Data denormalization", "performance",
           "Duplicate data to avoid expensive joins or cross-service calls at read time.",
           {"maintainability": "duplicated data must be kept consistent across writers"}),
    Tactic("Read-through/write-behind data access", "performance",
           "Batch or defer writes and prefetch reads to reduce per-operation latency.",
           {"maintainability": "failure windows can lose or reorder deferred writes"}),

    # --- security ---
    Tactic("Authentication", "security",
           "Verify the identity of a caller before granting access.",
           {"performance": "adds per-request latency for credential verification",
            "cost_operability": "requires identity infrastructure to run and maintain"}),
    Tactic("Fine-grained authorization / access control", "security",
           "Restrict operations to callers with the specific permissions they need.",
           {"maintainability": "permission policies must be kept in sync with the system's evolution"}),
    Tactic("Encryption in transit and at rest", "security",
           "Protect data from interception or unauthorized disk-level access.",
           {"performance": "adds CPU overhead for encrypt/decrypt operations",
            "cost_operability": "key management and rotation is an ongoing operational task"}),
    Tactic("Rate limiting / throttling", "security",
           "Cap how much load a single caller can place on the system.",
           {"performance": "adds bookkeeping overhead to every request, including legitimate ones"}),
    Tactic("Least-privilege service boundaries", "security",
           "Give each service only the access it strictly needs, isolating blast radius.",
           {"maintainability": "more, narrower services means more boundaries to coordinate changes across"}),
    Tactic("Input validation at trust boundaries", "security",
           "Reject malformed or malicious input as early as possible.",
           {"performance": "adds validation overhead on every incoming request"}),

    # --- maintainability ---
    Tactic("Modular decomposition by business capability", "maintainability",
           "Split the system along business boundaries so each module changes independently.",
           {"performance": "cross-module calls add latency that a single module wouldn't have",
            "cost_operability": "more independently deployable units means more to operate"}),
    Tactic("Versioned, well-defined API contracts", "maintainability",
           "Make module/service boundaries explicit and evolve them without breaking consumers.",
           {"cost_operability": "contract review and versioning tooling adds process overhead"}),
    Tactic("Automated regression test suite", "maintainability",
           "Catch behavioral regressions automatically before they reach production.",
           {"cost_operability": "CI compute time and suite upkeep is a recurring cost"}),
    Tactic("Centralized structured logging and observability", "maintainability",
           "Make system behavior inspectable across components for debugging and audits.",
           {"cost_operability": "log storage and processing infrastructure has an ongoing cost",
            "security": "logs can leak sensitive data if not deliberately scrubbed"}),
    Tactic("Dependency injection / inversion of control", "maintainability",
           "Decouple components from concrete implementations of their dependencies.",
           {"performance": "the extra indirection has a small but nonzero runtime cost"}),

    # --- scalability ---
    Tactic("Horizontal scaling with stateless instances", "scalability",
           "Add more identical instances behind a load balancer instead of growing one instance.",
           {"cost_operability": "more running instances means more infrastructure to pay for and operate",
            "maintainability": "session/request state must be externalized, not held in-process"}),
    Tactic("Database sharding", "scalability",
           "Partition data across multiple database instances by a shard key.",
           {"maintainability": "cross-shard queries and rebalancing become genuinely hard problems",
            "cost_operability": "operating multiple database instances multiplies operational surface"}),
    Tactic("Event-driven architecture", "scalability",
           "Components communicate via events rather than direct synchronous calls.",
           {"maintainability": "reasoning about system state requires tracing asynchronous event flows",
            "security": "each event channel is an additional integration point to secure"}),
    Tactic("Auto-scaling infrastructure", "scalability",
           "Automatically add or remove capacity in response to observed load.",
           {"performance": "newly-started instances suffer cold-start latency during scale-up"}),
    Tactic("Read replicas", "scalability",
           "Serve read traffic from replicated copies of the primary data store.",
           {"maintainability": "callers must tolerate and reason about replication lag"}),

    # --- cost_operability ---
    Tactic("Managed cloud services over self-hosted infrastructure", "cost_operability",
           "Use a vendor-operated service instead of running and patching the equivalent yourself.",
           {"security": "trust and blast radius now partly depend on the vendor's shared-responsibility model",
            "maintainability": "the system becomes coupled to that vendor's specific service semantics"}),
    Tactic("Infrastructure as code", "cost_operability",
           "Define infrastructure declaratively so environments are reproducible and cheap to recreate.",
           {}),
    Tactic("Multi-tenancy", "cost_operability",
           "Share infrastructure across multiple customers/tenants instead of provisioning per-tenant.",
           {"security": "a boundary failure between tenants has a much larger blast radius",
            "performance": "a noisy tenant can degrade performance for others sharing the same resources"}),
    Tactic("Serverless / function-as-a-service deployment", "cost_operability",
           "Pay only for actual invocation time instead of provisioning always-on capacity.",
           {"performance": "cold starts add latency to infrequently-invoked functions",
            "maintainability": "execution model constraints (timeouts, statelessness) shape how code must be written"}),
    Tactic("Circuit breaker for downstream dependencies", "cost_operability",
           "Stop calling a failing dependency instead of paying for retries against it.",
           {"maintainability": "failure-handling logic and fallback paths add real code complexity"}),
]


def build_knowledge_graph(tactics: list[Tactic]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(QUALITY_ATTRIBUTES)
    for tactic in tactics:
        graph.add_node(tactic.name)
        graph.add_edge(tactic.name, tactic.category, relation="supports")
        for other_qa, note in tactic.trade_offs.items():
            graph.add_edge(tactic.name, other_qa, relation="trade_off", note=note)
    return graph


def supporting_tactics_for(graph: nx.DiGraph, quality_attribute: str) -> list[str]:
    return sorted(
        tactic for tactic, qa, data in graph.edges(data=True)
        if qa == quality_attribute and data.get("relation") == "supports"
    )


def trade_offs_for_tactic(graph: nx.DiGraph, tactic_name: str) -> dict[str, str]:
    return {
        qa: data["note"]
        for _tactic, qa, data in graph.edges(tactic_name, data=True)
        if data.get("relation") == "trade_off"
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/deliberation/test_knowledge_graph.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/deliberation/__init__.py src/deliberation/knowledge_graph.py tests/deliberation/__init__.py tests/deliberation/test_knowledge_graph.py
git commit -m "feat: add architectural-tactics knowledge graph for deliberation"
git push
```

---

### Task 2: LLM client abstraction (Gemini + local open-weight backends)

**Files:**
- Create: `src/deliberation/llm_client.py`
- Create: `tests/deliberation/test_llm_client.py`

**Interfaces:**
- Consumes: `google.genai` (Gemini backend), `transformers` (local
  backend, imported lazily inside the loader function only).
- Produces:
  - `GeminiClient` — `GeminiClient(client, model_name=DEFAULT_GEMINI_MODEL)`;
    `.generate(prompt: str, system: str | None = None) -> str`. `client`
    is any object shaped like `google.genai.Client` (duck-typed for
    testability — tests inject a fake with a fake `.models.generate_content`).
  - `load_gemini_client(api_key: str | None = None) -> GeminiClient` — real
    constructor, reads `GEMINI_API_KEY` from the environment if `api_key`
    isn't passed. Not covered by a fast unit test (would need a real key);
    **not exercised for real in this plan** (see Global Constraints).
  - `LocalHFClient` — `LocalHFClient(generator)`; `.generate(prompt, system=None) -> str`.
    `generator` is any `callable(messages: list[dict]) -> str` (dependency
    injection — this is exactly the shape of the closure
    `load_local_hf_client` builds around a real HF `pipeline`, so tests
    inject a fake closure instead).
  - `load_local_hf_client(model_name: str = DEFAULT_LOCAL_MODEL) -> LocalHFClient` —
    real constructor; loads the model via `transformers.pipeline`. This
    **is** exercised for real in Task 4 (no API key needed).

- [ ] **Step 1: Write the failing tests**

```python
# tests/deliberation/test_llm_client.py
from src.deliberation.llm_client import GeminiClient, LocalHFClient


class _FakeGenAIResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _FakeGenAIResponse(f"response to: {contents}")


class _FakeGenAIClient:
    def __init__(self):
        self.models = _FakeModels()


def test_gemini_client_generate_sends_prompt_and_returns_text():
    fake_client = _FakeGenAIClient()
    client = GeminiClient(fake_client, model_name="gemini-3.5-flash-lite")

    result = client.generate("Should we use microservices?")

    assert result == "response to: Should we use microservices?"
    assert fake_client.models.calls[0]["model"] == "gemini-3.5-flash-lite"
    assert fake_client.models.calls[0]["contents"] == "Should we use microservices?"
    assert fake_client.models.calls[0]["config"] is None


def test_gemini_client_generate_passes_system_instruction_when_given():
    fake_client = _FakeGenAIClient()
    client = GeminiClient(fake_client, model_name="gemini-3.5-flash-lite")

    client.generate("prompt text", system="You are a performance advocate.")

    config = fake_client.models.calls[0]["config"]
    assert config.system_instruction == "You are a performance advocate."


def test_local_hf_client_generate_builds_chat_messages_and_returns_generator_output():
    seen_messages = []

    def fake_generator(messages):
        seen_messages.append(messages)
        return "generated text"

    client = LocalHFClient(fake_generator)

    result = client.generate("What should we decide?", system="You are an advocate.")

    assert result == "generated text"
    assert seen_messages == [[
        {"role": "system", "content": "You are an advocate."},
        {"role": "user", "content": "What should we decide?"},
    ]]


def test_local_hf_client_generate_without_system_omits_system_message():
    seen_messages = []
    client = LocalHFClient(lambda messages: seen_messages.append(messages) or "x")

    client.generate("no system prompt here")

    assert seen_messages == [[{"role": "user", "content": "no system prompt here"}]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/deliberation/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.deliberation.llm_client'`.

- [ ] **Step 3: Write the implementation**

```python
# src/deliberation/llm_client.py
"""LLM client abstraction for deliberation agents.

Both clients implement the same duck-typed interface:
    generate(prompt: str, system: str | None = None) -> str

Real model/SDK loading happens only in the load_* factory functions below,
never at import time, and (for the local backend) never at module level —
so importing this module never pulls in torch, and never conflicts with
the sentence_transformers-before-torch import-order rule documented in
PROGRESS.md. Callers that use both retrieval (Stage 1) and the local
backend (this module) in the same process must still import
src.retrieval.embeddings (or sentence_transformers directly) before
calling load_local_hf_client.
"""
import os

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class GeminiClient:
    def __init__(self, client, model_name: str = DEFAULT_GEMINI_MODEL):
        self._client = client
        self._model_name = model_name

    def generate(self, prompt: str, system: str | None = None) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(system_instruction=system) if system else None
        response = self._client.models.generate_content(
            model=self._model_name, contents=prompt, config=config,
        )
        return response.text


def load_gemini_client(api_key: str | None = None, model_name: str = DEFAULT_GEMINI_MODEL) -> GeminiClient:
    from google import genai

    api_key = api_key or os.environ["GEMINI_API_KEY"]
    return GeminiClient(genai.Client(api_key=api_key), model_name=model_name)


class LocalHFClient:
    def __init__(self, generator):
        self._generator = generator

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._generator(messages)


def load_local_hf_client(model_name: str = DEFAULT_LOCAL_MODEL) -> LocalHFClient:
    from transformers import pipeline

    hf_pipeline = pipeline("text-generation", model=model_name, torch_dtype="auto", device_map="auto")

    def generator(messages):
        outputs = hf_pipeline(messages, max_new_tokens=400, do_sample=True, temperature=0.7)
        return outputs[0]["generated_text"][-1]["content"]

    return LocalHFClient(generator)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/deliberation/test_llm_client.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Smoke-test the real local backend loads and generates (manual, not a pytest)**

Run (as a script file, not `python -c`, per the `conda run` argument-wrapping
quirk noted in `PROGRESS.md`):

```python
# scratch smoke-test — not committed
from src.deliberation.llm_client import load_local_hf_client

client = load_local_hf_client()
print(client.generate("In one sentence, what is a circuit breaker?"))
```

Expected: downloads `Qwen/Qwen2.5-1.5B-Instruct` on first run (~3 GB) and
prints a coherent one-sentence answer. Confirms the pipeline wiring and
message format are correct before Task 3/4 build on it.

- [ ] **Step 6: Commit**

```bash
git add src/deliberation/llm_client.py tests/deliberation/test_llm_client.py
git commit -m "feat: add Gemini and local-HF LLM client backends for deliberation"
git push
```

---

### Task 3: Quality-attribute advocate agent

**Files:**
- Create: `src/deliberation/agent.py`
- Create: `tests/deliberation/test_agent.py`

**Interfaces:**
- Consumes: `Tactic`, `supporting_tactics_for`, `trade_offs_for_tactic`
  (Task 1); any object shaped like `GeminiClient`/`LocalHFClient` (Task 2,
  duck-typed — tests inject a fake); `ADRRecord` (`src/retrieval/records.py`,
  already committed).
- Produces:
  - `AgentPosition` — frozen dataclass: `quality_attribute: str`,
    `round_number: int`, `stance: str` (`"propose"` or `"critique"`),
    `content: str`.
  - `QualityAttributeAgent` —
    `QualityAttributeAgent(quality_attribute: str, llm_client, knowledge_graph)`;
    `.propose(context: str, precedents: list[ADRRecord]) -> AgentPosition`
    (round 1: build a prompt from the agent's own supporting tactics +
    precedent titles/excerpts + the decision context, call the client);
    `.critique(candidate_text: str, other_positions: list[AgentPosition], round_number: int) -> AgentPosition`
    (later rounds: react to the current candidate and other agents' most
    recent positions, referencing this agent's own trade-off knowledge).
    Task 4's orchestrator calls both methods.

- [ ] **Step 1: Write the failing tests**

```python
# tests/deliberation/test_agent.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/deliberation/test_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.deliberation.agent'`.

- [ ] **Step 3: Write the implementation**

```python
# src/deliberation/agent.py
"""Quality-attribute advocate agent (CADENCE Stage 2)."""
from dataclasses import dataclass

from src.deliberation.knowledge_graph import supporting_tactics_for, trade_offs_for_tactic
from src.retrieval.records import ADRRecord


@dataclass(frozen=True)
class AgentPosition:
    quality_attribute: str
    round_number: int
    stance: str
    content: str


class QualityAttributeAgent:
    def __init__(self, quality_attribute: str, llm_client, knowledge_graph):
        self._quality_attribute = quality_attribute
        self._client = llm_client
        self._graph = knowledge_graph

    def _system_prompt(self) -> str:
        tactics = supporting_tactics_for(self._graph, self._quality_attribute)
        lines = [
            f"You are the {self._quality_attribute} advocate in an architectural "
            "decision-making deliberation. Argue for the decision that best serves "
            f"{self._quality_attribute}, while acknowledging real trade-offs.",
            f"Tactics you can draw on for {self._quality_attribute}: " + ", ".join(tactics),
        ]
        return "\n".join(lines)

    def propose(self, context: str, precedents: list[ADRRecord]) -> AgentPosition:
        precedent_lines = "\n".join(f"- {p.title}" for p in precedents) or "(no precedents retrieved)"
        prompt = (
            f"Decision context:\n{context}\n\n"
            f"Precedent decisions from similar past projects:\n{precedent_lines}\n\n"
            f"Propose and justify a position from the {self._quality_attribute} "
            "perspective for this decision."
        )
        content = self._client.generate(prompt, system=self._system_prompt())
        return AgentPosition(self._quality_attribute, round_number=1, stance="propose", content=content)

    def critique(
        self, candidate_text: str, other_positions: list["AgentPosition"], round_number: int
    ) -> AgentPosition:
        others_lines = "\n".join(
            f"- [{p.quality_attribute}] {p.content}" for p in other_positions
        ) or "(no other positions yet)"
        prompt = (
            f"Current candidate decision:\n{candidate_text}\n\n"
            f"Other advocates' positions this round:\n{others_lines}\n\n"
            f"From the {self._quality_attribute} perspective, critique or refine "
            "this candidate, citing specific trade-offs where relevant."
        )
        content = self._client.generate(prompt, system=self._system_prompt())
        return AgentPosition(self._quality_attribute, round_number=round_number, stance="critique", content=content)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/deliberation/test_agent.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/deliberation/agent.py tests/deliberation/test_agent.py
git commit -m "feat: add quality-attribute advocate agent"
git push
```

---

### Task 4: Deliberation orchestrator, synthesizer, and real end-to-end demo

**Files:**
- Create: `src/deliberation/orchestrator.py`
- Create: `scripts/run_deliberation_demo.py`
- Create: `tests/deliberation/test_orchestrator.py`
- Create: `tests/data/test_run_deliberation_demo_script.py`

**Interfaces:**
- Consumes: `QualityAttributeAgent`, `AgentPosition` (Task 3);
  `QUALITY_ATTRIBUTES`, `build_knowledge_graph`, `TACTICS` (Task 1);
  `Retriever` (`src/retrieval/retriever.py`, already committed);
  `load_local_hf_client` (Task 2).
- Produces:
  - `DeliberationResult` — frozen dataclass: `converged_candidate: str`,
    `rationale: str`, `transcript: list[AgentPosition]`, `rounds_run: int`.
  - `DeliberationOrchestrator` —
    `DeliberationOrchestrator(agents: list[QualityAttributeAgent], synthesizer_client, max_rounds: int = 3)`;
    `.deliberate(context: str, precedents: list[ADRRecord]) -> DeliberationResult` —
    round 1: every agent proposes independently; rounds 2..max_rounds:
    every agent critiques the current candidate (the most recent round's
    positions, joined) given every other agent's latest position; after
    the final round, `synthesizer_client.generate(...)` (a plain LLM
    client, not tied to any one quality attribute) produces the converged
    candidate text and rationale from the full transcript.
  - `scripts/run_deliberation_demo.py` — real end-to-end script: builds
    the 5 agents (one `LocalHFClient` shared across all of them — same
    model, different system prompts — and reused as the synthesizer too,
    to avoid loading multiple models), retrieves precedents via `Retriever`
    (loading `data/processed/adr_records.jsonl` +
    `data/processed/adr_embeddings.npy` from Stage 1), runs
    `DeliberationOrchestrator.deliberate(...)` on a sample decision
    context, and prints the full transcript + converged decision.

- [ ] **Step 1: Write the failing tests**

```python
# tests/deliberation/test_orchestrator.py
from src.deliberation.agent import AgentPosition, QualityAttributeAgent
from src.deliberation.knowledge_graph import TACTICS, build_knowledge_graph
from src.deliberation.orchestrator import DeliberationOrchestrator


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n py313 pytest tests/deliberation/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.deliberation.orchestrator'`.

- [ ] **Step 3: Write the orchestrator**

```python
# src/deliberation/orchestrator.py
"""Bounded-round multi-agent deliberation orchestrator (CADENCE Stage 2)."""
from dataclasses import dataclass

from src.deliberation.agent import AgentPosition, QualityAttributeAgent
from src.retrieval.records import ADRRecord


@dataclass(frozen=True)
class DeliberationResult:
    converged_candidate: str
    rationale: str
    transcript: list[AgentPosition]
    rounds_run: int


def _parse_synthesis(text: str) -> tuple[str, str]:
    candidate, rationale = "", ""
    for line in text.splitlines():
        if line.startswith("CANDIDATE:"):
            candidate = line[len("CANDIDATE:"):].strip()
        elif line.startswith("RATIONALE:"):
            rationale = line[len("RATIONALE:"):].strip()
    return candidate, rationale


class DeliberationOrchestrator:
    def __init__(self, agents: list[QualityAttributeAgent], synthesizer_client, max_rounds: int = 3):
        self._agents = agents
        self._synthesizer = synthesizer_client
        self._max_rounds = max_rounds

    def deliberate(self, context: str, precedents: list[ADRRecord]) -> DeliberationResult:
        transcript: list[AgentPosition] = []

        round_1 = [agent.propose(context, precedents) for agent in self._agents]
        transcript.extend(round_1)
        latest = round_1

        for round_number in range(2, self._max_rounds + 1):
            candidate_text = "\n".join(p.content for p in latest)
            next_round = []
            for agent in self._agents:
                others = [p for p in latest if p.quality_attribute != agent._quality_attribute]
                next_round.append(agent.critique(candidate_text, others, round_number))
            transcript.extend(next_round)
            latest = next_round

        transcript_text = "\n".join(f"[{p.quality_attribute} r{p.round_number}] {p.content}" for p in transcript)
        synthesis_prompt = (
            f"Decision context:\n{context}\n\n"
            f"Full deliberation transcript:\n{transcript_text}\n\n"
            "Synthesize the deliberation into a final decision. Respond in exactly this format:\n"
            "CANDIDATE: <one or two sentence decision>\n"
            "RATIONALE: <one paragraph rationale>"
        )
        synthesis = self._synthesizer.generate(synthesis_prompt)
        candidate, rationale = _parse_synthesis(synthesis)

        return DeliberationResult(
            converged_candidate=candidate,
            rationale=rationale,
            transcript=transcript,
            rounds_run=self._max_rounds,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n py313 pytest tests/deliberation/test_orchestrator.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Write the failing test for the real end-to-end script's wiring**

```python
# tests/data/test_run_deliberation_demo_script.py
def test_run_demo_wires_retriever_and_orchestrator_together(tmp_path, monkeypatch):
    import json
    import numpy as np

    from scripts.run_deliberation_demo import run_demo

    records_path = tmp_path / "adr_records.jsonl"
    records_path.write_text(
        json.dumps({
            "record_id": "r/1.md", "repo_folder": "r", "repository_url": None,
            "relative_path": "1.md", "sequence_number": 1, "title": "Use read replicas",
            "raw_text": "text", "extraction_status": "Verified",
        }) + "\n",
        encoding="utf-8",
    )
    embeddings_path = tmp_path / "adr_embeddings.npy"
    np.save(embeddings_path, np.array([[1.0, 0.0]]))

    class _FakeClient:
        def generate(self, prompt, system=None):
            return "CANDIDATE: c\nRATIONALE: r"

    class _FakeEmbeddingModel:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr("scripts.run_deliberation_demo.load_local_hf_client", lambda: _FakeClient())
    monkeypatch.setattr("scripts.run_deliberation_demo.load_embedding_model", lambda: _FakeEmbeddingModel())

    result = run_demo(
        records_path=records_path, embeddings_path=embeddings_path,
        context="Sample decision context.", max_rounds=1,
    )

    assert result.converged_candidate == "c"
    assert result.rationale == "r"
    assert len(result.transcript) == 5  # 5 quality-attribute agents, 1 propose round each
```

- [ ] **Step 6: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/data/test_run_deliberation_demo_script.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_deliberation_demo'`.

- [ ] **Step 7: Write the script**

```python
# scripts/run_deliberation_demo.py
"""Real end-to-end demo: Stage 1 retrieval feeding Stage 2 deliberation.

Import order: src.retrieval.embeddings (which imports sentence_transformers)
is imported before anything that triggers a torch import via the local-HF
deliberation client, per the sentence_transformers-before-torch rule in
PROGRESS.md.
"""
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.retrieval.embeddings import embed_texts, load_embedding_model  # noqa: F401  (import before torch)
import numpy as np

from src.deliberation.agent import QualityAttributeAgent
from src.deliberation.knowledge_graph import QUALITY_ATTRIBUTES, TACTICS, build_knowledge_graph
from src.deliberation.llm_client import load_local_hf_client
from src.deliberation.orchestrator import DeliberationOrchestrator
from src.retrieval.records import ADRRecord
from src.retrieval.retriever import Retriever

PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
RECORDS_PATH = PROCESSED_DIR / "adr_records.jsonl"
EMBEDDINGS_PATH = PROCESSED_DIR / "adr_embeddings.npy"

SAMPLE_CONTEXT = (
    "Our service-oriented system's order-processing service is experiencing "
    "10x read traffic growth from a new mobile client. Requirements: keep "
    "p99 read latency low, keep the change operable by a small team, and "
    "avoid introducing new categories of security risk."
)


def _load_records(records_path: Path) -> list[ADRRecord]:
    records = []
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            records.append(ADRRecord(**json.loads(line)))
    return records


def run_demo(records_path: Path, embeddings_path: Path, context: str, max_rounds: int = 3):
    records = _load_records(records_path)
    embeddings = np.load(embeddings_path)
    embedding_model = load_embedding_model()
    retriever = Retriever(records, embeddings, embedding_model)
    precedents = retriever.retrieve(context, k=3)

    llm_client = load_local_hf_client()
    graph = build_knowledge_graph(TACTICS)
    agents = [QualityAttributeAgent(qa, llm_client, graph) for qa in QUALITY_ATTRIBUTES]
    orchestrator = DeliberationOrchestrator(agents, llm_client, max_rounds=max_rounds)

    return orchestrator.deliberate(context, precedents)


if __name__ == "__main__":
    result = run_demo(RECORDS_PATH, EMBEDDINGS_PATH, SAMPLE_CONTEXT, max_rounds=2)
    print("=== Transcript ===")
    for position in result.transcript:
        print(f"[round {position.round_number} | {position.quality_attribute} | {position.stance}]")
        print(position.content)
        print()
    print("=== Converged candidate ===")
    print(result.converged_candidate)
    print("=== Rationale ===")
    print(result.rationale)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/data/test_run_deliberation_demo_script.py -v`
Expected: PASS (1 passed).

- [ ] **Step 9: Run the full suite, then run the real demo**

Run: `conda run -n py313 pytest -q` — expect all tests (old + new) passing.

Run: `conda run -n py313 python scripts/run_deliberation_demo.py` (or the
env's `python.exe` directly if `conda run` mis-wraps the invocation, per
`PROGRESS.md`) — this loads the real embedding model, the real corpus
index, and `Qwen/Qwen2.5-1.5B-Instruct` (downloads ~3 GB on first run),
then runs 5 agents through 2 rounds plus synthesis (11 real LLM calls).
Expect it to print a full transcript and a converged candidate + rationale
that are topically coherent (won't be publication-quality — that's a
model/prompt-tuning question for the evaluation-harness plan, not this
one) and reference at least some of the retrieved precedents or tactics.

- [ ] **Step 10: Commit**

```bash
git add src/deliberation/orchestrator.py scripts/run_deliberation_demo.py tests/deliberation/test_orchestrator.py tests/data/test_run_deliberation_demo_script.py
git commit -m "feat: add deliberation orchestrator and real end-to-end demo"
git push
```

---

## Self-Review Notes

- **Spec coverage:** implements exactly CADENCE Stage 2 (spec §3) plus the
  knowledge-graph construction method spec §9 flagged as needing its own
  design pass. Does not implement Stage 3 (solver) or Stage 4 (critique) —
  separate follow-on plans. `DeliberationResult` is intentionally a plain
  text candidate + rationale + transcript, not yet a structured
  "trade-off commitments" object Stage 3 could encode directly into
  MaxSAT/SMT — deciding that structure now, before Stage 3's design pass
  has even started, would be premature; Stage 3's plan should read the
  transcript/candidate text and decide what structured extraction (if any)
  it actually needs.
- **Why a hand-curated knowledge graph, not a mined one:** see the header's
  "Knowledge graph construction method" section — the ADR corpus (Stage 1)
  already supplies real-world precedent grounding; this graph exists to
  supply structured trade-off relations that free text doesn't carry
  explicitly, so mining it from the same corpus would be redundant with
  Stage 1 and wouldn't produce the structured (tactic, quality-attribute,
  trade-off) triples this stage needs.
- **Why `GeminiClient` ships untested-for-real:** no `GEMINI_API_KEY` is
  configured on this machine (see `PROGRESS.md`) — implementing it against
  the real, verified `google.genai` SDK shape (confirmed by reading the
  installed package's source, not guessed) and unit-testing it against a
  mock is the most that's honestly achievable without a key. **Whoever
  next has a key must run one real smoke-test call before trusting this
  client for an actual evaluation run.**
- **Why a small local model for this plan's real run:** `Qwen2.5-1.5B-
  Instruct` is fast to download/run and enough to prove the protocol
  produces coherent, on-topic multi-round output. Spec §6's suggested
  7B/14B models are heavier and their added quality is a question for the
  evaluation-harness plan, which will need to make a deliberate
  model-choice decision anyway (informed by actual metric comparisons) —
  baking that decision into this plan would be premature and would slow
  down iterating on the protocol itself.
- **Why agents share one `LocalHFClient`/model instance:** loading 5+
  separate model instances would multiply GPU memory and load time for no
  benefit — the quality-attribute distinction lives entirely in each
  agent's system prompt, not in having a physically separate model.
- **Why the synthesizer isn't its own quality-attribute agent:** it needs
  to weigh all five concerns neutrally, not advocate for one — reusing the
  same underlying `llm_client` (just called directly, without a QA-specific
  system prompt) is the simplest correct design; no separate class needed.
- **Placeholder scan:** no TBD/TODO; all code blocks are complete and
  runnable.
- **Type/interface consistency:** `AgentPosition` fields match between
  `agent.py` (definition) and `orchestrator.py` (consumption via
  `p.quality_attribute`/`p.round_number`/`p.stance`/`p.content`).
  `QualityAttributeAgent.propose`/`.critique` signatures match between
  Task 3 (definition) and Task 4 (`DeliberationOrchestrator.deliberate`
  call sites). `llm_client.generate(prompt, system=None)` signature is
  used identically by `GeminiClient`, `LocalHFClient`, `QualityAttributeAgent`,
  and `DeliberationOrchestrator`'s synthesizer call.
