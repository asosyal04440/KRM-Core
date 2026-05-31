from krm.concepts.concept_card import ConceptCard
from krm.graph.ghost_edges import GhostEdgeGenerator, SourceContext
from krm.reasoning.router import QueryRouter
from krm.runtime.memory_budget import MemoryBudget


def _edge_types(names: list[str], query: str, text: str = "") -> set[str]:
    concepts = [ConceptCard(i + 1, name, domain_id=1, source_refs=["s"]) for i, name in enumerate(names)]
    edges = GhostEdgeGenerator().generate(
        concepts,
        [SourceContext("s", [card.concept_id for card in concepts], text)],
        QueryRouter().classify(query),
        MemoryBudget.for_profile("local_core"),
        max_edges=20,
    )
    return {edge.edge_type for edge in edges}


def test_cause_input_output_and_counterfactual_patterns() -> None:
    assert "CAUSES" in _edge_types(["coal", "steam engine", "textile industry"], "why industrial revolution")
    assert "INPUT_OUTPUT" in _edge_types(["photosynthesis", "glucose", "oxygen"], "compare photosynthesis respiration")
    assert "COUNTERFACTUAL_LINK" in _edge_types(["printing press", "Ottoman Empire", "literacy"], "what if ottoman printing")


def test_edge_caps_are_respected() -> None:
    concepts = [ConceptCard(i, f"concept {i}", domain_id=1, source_refs=["s"]) for i in range(1, 20)]
    edges = GhostEdgeGenerator().generate(
        concepts,
        [SourceContext("s", [card.concept_id for card in concepts])],
        QueryRouter().classify("why"),
        MemoryBudget.for_profile("local_core"),
        max_edges=5,
    )
    assert len(edges) <= 5
