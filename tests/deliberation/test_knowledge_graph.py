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
