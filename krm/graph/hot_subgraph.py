from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from krm.concepts.concept_card import ConceptCard

if TYPE_CHECKING:
    from krm.graph.ghost_edges import GhostEdge


@dataclass(slots=True)
class HotSubgraph:
    query: str
    selected_concepts: list[ConceptCard]
    selected_ghost_edges: list["GhostEdge"]
    activation_scores: dict[int, float]
    attractor_clusters: list[list[int]] = field(default_factory=list)
    strongest_paths: list[list[int]] = field(default_factory=list)
    source_pointers: list[str] = field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    resonance_rounds: int = 0
    trace: dict[str, Any] = field(default_factory=dict)

    def top_concepts(self, limit: int = 10) -> list[ConceptCard]:
        by_id = {card.concept_id: card for card in self.selected_concepts}
        ranked = sorted(self.activation_scores.items(), key=lambda item: (-item[1], item[0]))
        return [by_id[cid] for cid, _ in ranked[:limit] if cid in by_id]

    def estimate_size_bytes(self) -> int:
        return sum(card.estimate_size_bytes() for card in self.selected_concepts) + len(self.selected_ghost_edges) * 192

    def to_compact_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "concepts": [c.to_compact_dict() for c in self.selected_concepts],
            "edges": [asdict(edge) for edge in self.selected_ghost_edges],
            "activation_scores": self.activation_scores,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "resonance_rounds": self.resonance_rounds,
            "trace": self.trace,
        }

    def pretty(self, limit: int = 10) -> str:
        top = ", ".join(card.canonical_name for card in self.top_concepts(limit))
        return f"HotSubgraph(confidence={self.confidence:.2f}, top=[{top}])"
