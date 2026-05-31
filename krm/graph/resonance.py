from __future__ import annotations

from dataclasses import dataclass

from krm.concepts.concept_card import ConceptCard
from krm.graph.ghost_edges import GhostEdge
from krm.graph.hot_subgraph import HotSubgraph
from krm.reasoning.router import QueryIntent
from krm.runtime.memory_budget import MemoryBudget


EDGE_WEIGHTS = {
    "CAUSES": 1.25,
    "CAUSED_BY": 1.20,
    "ENABLES": 1.15,
    "DEPENDS_ON": 1.10,
    "INPUT_OUTPUT": 1.15,
    "CONTRASTS": 1.05,
    "DOMAIN_NEAR": 0.70,
    "CO_OCCURS": 0.62,
    "SAME_ARTICLE": 0.72,
    "SOURCE_LINKED": 0.78,
    "TEMPORAL_NEAR": 0.80,
    "COUNTERFACTUAL_LINK": 0.85,
    "EDUCATIONAL_PAIR": 0.90,
}


@dataclass(slots=True)
class ResonanceConfig:
    max_rounds: int = 4
    max_active_concepts: int = 300
    max_edges: int = 2000
    activation_decay: float = 0.62
    min_activation: float = 0.01
    max_cluster_count: int = 5


class ResonanceEngine:
    def run(
        self,
        query: str,
        concepts: list[ConceptCard],
        initial_scores: dict[int, float],
        ghost_edges: list[GhostEdge],
        query_intent: QueryIntent,
        budget: MemoryBudget,
        config: ResonanceConfig | None = None,
    ) -> HotSubgraph:
        cfg = config or ResonanceConfig()
        rounds = budget.clamp_rounds(cfg.max_rounds)
        max_concepts = budget.clamp_candidates(cfg.max_active_concepts)
        max_edges = budget.clamp_edges(cfg.max_edges)
        concepts = sorted(concepts, key=lambda c: (-initial_scores.get(c.concept_id, 0.0), c.concept_id))[:max_concepts]
        active_ids = {card.concept_id for card in concepts}
        edges = [edge for edge in ghost_edges if edge.src_id in active_ids and edge.dst_id in active_ids][:max_edges]
        activation = {cid: max(0.0, score) for cid, score in initial_scores.items() if cid in active_ids}
        names = {card.concept_id: card.canonical_name for card in concepts}
        trace: dict[str, object] = {
            "initial_activations": {names.get(cid, str(cid)): round(value, 4) for cid, value in activation.items()},
            "rounds": [],
            "top_edges_used": [],
            "pruning_decisions": [],
            "degradation_decisions": list(budget.degradation_decisions),
        }
        if len(concepts) < len(initial_scores):
            trace["pruning_decisions"].append(f"pruned active concepts to {len(concepts)}")
        if len(edges) < len(ghost_edges):
            trace["pruning_decisions"].append(f"pruned ghost edges to {len(edges)}")
        for card in concepts:
            activation.setdefault(card.concept_id, 0.01)
            if card.domain_id in query_intent.needed_domains:
                activation[card.concept_id] += 0.10
        for _ in range(rounds):
            next_activation = {cid: value * cfg.activation_decay for cid, value in activation.items()}
            edge_uses: list[tuple[float, GhostEdge]] = []
            for edge in edges:
                edge_weight = self._edge_weight(edge.edge_type, query_intent) * edge.weight * edge.confidence
                transfer = activation.get(edge.src_id, 0.0) * edge_weight * 0.20
                if transfer >= cfg.min_activation:
                    next_activation[edge.dst_id] = next_activation.get(edge.dst_id, 0.0) + transfer
                    edge_uses.append((transfer, edge))
            activation = {cid: value for cid, value in next_activation.items() if value >= cfg.min_activation}
            round_top = sorted(activation.items(), key=lambda item: (-item[1], item[0]))[:8]
            trace["rounds"].append(
                {
                    "top_concepts": [
                        {"concept": names.get(cid, str(cid)), "activation": round(value, 4)}
                        for cid, value in round_top
                    ],
                    "activation_count": len(activation),
                }
            )
            for transfer, edge in sorted(edge_uses, key=lambda item: -item[0])[:5]:
                trace["top_edges_used"].append(
                    {
                        "src": names.get(edge.src_id, str(edge.src_id)),
                        "dst": names.get(edge.dst_id, str(edge.dst_id)),
                        "edge_type": edge.edge_type,
                        "transfer": round(transfer, 4),
                        "reason": edge.reason,
                    }
                )
        top_values = sorted(activation.values(), reverse=True)
        confidence = min(1.0, sum(top_values[:5]) / 8.0) if top_values else 0.0
        if query_intent.intent_type.value == "COUNTERFACTUAL" and len(sources := {src for card in concepts for src in card.source_refs}) < 2:
            confidence *= 0.85
        warnings = list(budget.degradation_decisions)
        if confidence < 0.35:
            warnings.append("low confidence from sparse activated evidence")
        sources = sorted({src for card in concepts for src in card.source_refs})
        trace["final_attractor_clusters"] = self._clusters(concepts, activation)
        return HotSubgraph(
            query=query,
            selected_concepts=concepts,
            selected_ghost_edges=edges,
            activation_scores=activation,
            attractor_clusters=self._clusters(concepts, activation),
            strongest_paths=self._paths(edges, activation),
            source_pointers=sources,
            confidence=confidence,
            warnings=warnings,
            resonance_rounds=rounds,
            trace=trace,
        )

    def _edge_weight(self, edge_type: str, query_intent: QueryIntent) -> float:
        weight = EDGE_WEIGHTS.get(edge_type, 0.55)
        intent = query_intent.intent_type.value
        if intent == "CAUSE_EFFECT" and edge_type in {"CAUSES", "CAUSED_BY", "ENABLES", "DEPENDS_ON"}:
            return weight * 1.35
        if intent == "COMPARISON" and edge_type in {"CONTRASTS", "INPUT_OUTPUT", "EDUCATIONAL_PAIR"}:
            return weight * 1.35
        if intent == "COUNTERFACTUAL" and edge_type in {"CAUSES", "ENABLES", "TEMPORAL_NEAR", "COUNTERFACTUAL_LINK"}:
            return weight * 1.25
        return weight

    def _clusters(self, concepts: list[ConceptCard], activation: dict[int, float]) -> list[list[int]]:
        by_domain: dict[int, list[int]] = {}
        for card in concepts:
            if card.concept_id in activation:
                by_domain.setdefault(card.domain_id, []).append(card.concept_id)
        return [ids[:8] for _, ids in sorted(by_domain.items())[:5]]

    def _paths(self, edges: list[GhostEdge], activation: dict[int, float]) -> list[list[int]]:
        ranked = sorted(edges, key=lambda e: (-(activation.get(e.src_id, 0.0) + activation.get(e.dst_id, 0.0)), e.src_id, e.dst_id))
        return [[edge.src_id, edge.dst_id] for edge in ranked[:8]]
