from krm.concepts.concept_card import ConceptCard
from krm.graph.ghost_edges import GhostEdgeGenerator, SourceContext
from krm.reasoning.router import QueryRouter
from krm.runtime.memory_budget import MemoryBudget


def test_ghost_edges_generate_caps_and_reasons() -> None:
    concepts = [
        ConceptCard(1, "coal", domain_id=1, source_refs=["src"]),
        ConceptCard(2, "steam engine", domain_id=1, source_refs=["src"]),
        ConceptCard(3, "textile industry", domain_id=1, source_refs=["src"]),
    ]
    intent = QueryRouter().classify("Why did the Industrial Revolution start in Britain?")
    edges = GhostEdgeGenerator().generate(
        concepts,
        [SourceContext("src", [1, 2, 3])],
        intent,
        MemoryBudget.for_profile("local_core"),
        max_edges=2,
    )
    assert len(edges) <= 2
    assert edges
    assert all(edge.reason for edge in edges)
    assert any(edge.edge_type == "CAUSES" for edge in edges)
