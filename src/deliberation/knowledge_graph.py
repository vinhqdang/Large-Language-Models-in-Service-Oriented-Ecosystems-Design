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
