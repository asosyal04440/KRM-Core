from krm.concepts.concept_card import ConceptCard
from krm.graph.ghost_edges import GhostEdge
from krm.graph.resonance import ResonanceConfig, ResonanceEngine
from krm.reasoning.router import QueryRouter
from krm.runtime.memory_budget import MemoryBudget


def test_resonance_activation_propagates_and_is_deterministic() -> None:
    concepts = [
        ConceptCard(1, "coal", domain_id=1, source_refs=["src"]),
        ConceptCard(2, "steam engine", domain_id=1, source_refs=["src"]),
    ]
    edges = [GhostEdge(1, 2, "CAUSES", 1.0, 1.0, "test")]
    intent = QueryRouter().classify("why coal steam engine")
    budget = MemoryBudget.for_profile("local_core")
    graph_a = ResonanceEngine().run("q", concepts, {1: 2.0}, edges, intent, budget, ResonanceConfig(max_rounds=2))
    graph_b = ResonanceEngine().run("q", concepts, {1: 2.0}, edges, intent, MemoryBudget.for_profile("local_core"), ResonanceConfig(max_rounds=2))
    assert graph_a.activation_scores[2] > 0
    assert graph_a.top_concepts()[0].canonical_name == graph_b.top_concepts()[0].canonical_name


def test_broad_query_gets_pruned_under_tiny_budget() -> None:
    concepts = [ConceptCard(i, f"concept {i}", domain_id=0, source_refs=["src"]) for i in range(1, 50)]
    intent = QueryRouter().classify("explain everything")
    tiny = MemoryBudget.for_profile("ultra_lite")
    graph = ResonanceEngine().run(
        "q",
        concepts,
        {card.concept_id: 1.0 for card in concepts},
        [],
        intent,
        tiny,
        ResonanceConfig(max_rounds=10, max_active_concepts=100),
    )
    assert graph.resonance_rounds <= tiny.profile.default_rounds
    assert len(graph.selected_concepts) <= tiny.profile.default_candidates
