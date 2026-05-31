from __future__ import annotations

from dataclasses import dataclass

from krm.concepts.concept_card import ConceptCard
from krm.reasoning.router import QueryIntent
from krm.runtime.memory_budget import MemoryBudget


@dataclass(slots=True)
class SourceContext:
    source_pointer_id: str
    concept_ids: list[int]
    text: str = ""


@dataclass(slots=True)
class GhostEdge:
    src_id: int
    dst_id: int
    edge_type: str
    weight: float
    confidence: float
    reason: str
    source_pointer_id: str | None = None


CAUSE_HINTS = {
    ("coal", "steam engine"),
    ("steam engine", "textile industry"),
    ("trade", "capital"),
    ("agriculture", "labor"),
    ("britain", "coal"),
    ("printing press", "literacy"),
    ("printing press", "bureaucracy"),
    ("printing press", "reform"),
}
ENABLES_HINTS = {
    ("coal", "steam engine"),
    ("capital", "textile industry"),
    ("trade", "capital"),
    ("transport", "trade"),
}
INPUT_OUTPUT_HINTS = {
    ("photosynthesis", "carbon dioxide"),
    ("photosynthesis", "water"),
    ("photosynthesis", "glucose"),
    ("photosynthesis", "oxygen"),
    ("cellular respiration", "glucose"),
    ("cellular respiration", "oxygen"),
    ("cellular respiration", "carbon dioxide"),
    ("cellular respiration", "energy"),
    ("cellular respiration", "mitochondria"),
    ("photosynthesis", "chloroplast"),
}
CONTRAST_HINTS = {
    ("photosynthesis", "cellular respiration"),
}
COUNTERFACTUAL_HINTS = {
    ("printing press", "Ottoman Empire"),
    ("printing press", "literacy"),
    ("printing press", "bureaucracy"),
    ("printing press", "education"),
    ("printing press", "reform"),
    ("printing press", "resistance"),
}
TEMPORAL_HINTS = {
    ("printing press", "earlier"),
    ("Ottoman Empire", "earlier"),
    ("Industrial Revolution", "Britain"),
}


class GhostEdgeGenerator:
    def generate(
        self,
        active_concepts: list[ConceptCard],
        source_contexts: list[SourceContext],
        query_intent: QueryIntent,
        budget: MemoryBudget,
        max_edges: int = 1000,
    ) -> list[GhostEdge]:
        cap = budget.clamp_edges(max_edges)
        cards = {card.concept_id: card for card in active_concepts}
        edges: list[GhostEdge] = []
        seen: set[tuple[int, int, str]] = set()
        per_type: dict[str, int] = {}
        for ctx in source_contexts:
            ids = [cid for cid in ctx.concept_ids if cid in cards]
            for i, src in enumerate(ids):
                per_node = 0
                for dst in ids[i + 1 :]:
                    if per_node >= 4 or len(edges) >= cap:
                        break
                    edge = self._edge(cards[src], cards[dst], ctx.source_pointer_id, query_intent, ctx.text)
                    key = (edge.src_id, edge.dst_id, edge.edge_type)
                    if key not in seen and per_type.get(edge.edge_type, 0) < max(8, cap // 2):
                        edges.append(edge)
                        seen.add(key)
                        per_type[edge.edge_type] = per_type.get(edge.edge_type, 0) + 1
                        per_node += 1
                if len(edges) >= cap:
                    break
            if len(edges) >= cap:
                break
        return edges

    def _edge(self, src: ConceptCard, dst: ConceptCard, pointer_id: str, intent: QueryIntent, source_text: str = "") -> GhostEdge:
        left = src.canonical_name.lower()
        right = dst.canonical_name.lower()
        pair = (left, right)
        reverse = (right, left)
        text = source_text.lower()
        if pair in CAUSE_HINTS or reverse in CAUSE_HINTS:
            return self._directed(src, dst, pair, "CAUSES", 1.45, 0.86, "cause-effect pattern relation", pointer_id)
        if pair in ENABLES_HINTS or reverse in ENABLES_HINTS:
            return self._directed(src, dst, pair, "ENABLES", 1.30, 0.80, "enables/dependency relation", pointer_id)
        if pair in INPUT_OUTPUT_HINTS or reverse in INPUT_OUTPUT_HINTS:
            return self._directed(src, dst, pair, "INPUT_OUTPUT", 1.35, 0.84, "biology input-output relation", pointer_id)
        if pair in CONTRAST_HINTS or reverse in CONTRAST_HINTS or any(w in text for w in ["unlike", "whereas", "both", "difference"]):
            return GhostEdge(src.concept_id, dst.concept_id, "CONTRASTS", 1.20, 0.78, "comparison pattern relation", pointer_id)
        if pair in COUNTERFACTUAL_HINTS or reverse in COUNTERFACTUAL_HINTS or intent.intent_type.value == "COUNTERFACTUAL":
            return GhostEdge(src.concept_id, dst.concept_id, "COUNTERFACTUAL_LINK", 1.15, 0.66, "counterfactual query-local link", pointer_id)
        if pair in TEMPORAL_HINTS or reverse in TEMPORAL_HINTS or any(w in text for w in ["earlier", "later", "began"]):
            return GhostEdge(src.concept_id, dst.concept_id, "TEMPORAL_NEAR", 0.95, 0.70, "temporal proximity pattern", pointer_id)
        if src.domain_id == dst.domain_id:
            if src.source_refs and set(src.source_refs).intersection(dst.source_refs):
                return GhostEdge(src.concept_id, dst.concept_id, "SAME_ARTICLE", 0.90, 0.74, "same source article", pointer_id)
            return GhostEdge(src.concept_id, dst.concept_id, "DOMAIN_NEAR", 0.72, 0.68, "same domain candidate set", pointer_id)
        return GhostEdge(src.concept_id, dst.concept_id, "CO_OCCURS", 0.58, 0.62, "same active source context", pointer_id)

    def _directed(
        self,
        src: ConceptCard,
        dst: ConceptCard,
        pair: tuple[str, str],
        edge_type: str,
        weight: float,
        confidence: float,
        reason: str,
        pointer_id: str,
    ) -> GhostEdge:
        if pair in CAUSE_HINTS or pair in ENABLES_HINTS or pair in INPUT_OUTPUT_HINTS:
            return GhostEdge(src.concept_id, dst.concept_id, edge_type, weight, confidence, reason, pointer_id)
        return GhostEdge(dst.concept_id, src.concept_id, edge_type, weight, confidence, reason, pointer_id)
