from __future__ import annotations

from dataclasses import dataclass, field

from krm.concepts.concept_card import ConceptCard
from krm.concepts.domain import domain_name
from krm.index.fingerprint_index import fingerprint_similarity, fingerprint_text
from krm.index.lexical_index import LexicalIndex, tokenize
from krm.reasoning.router import QueryIntent
from krm.runtime.memory_budget import MemoryBudget


@dataclass(frozen=True, slots=True)
class RetrievalScoring:
    exact_title: float = 8.0
    alias: float = 5.0
    lexical: float = 1.4
    domain_prior: float = 1.0
    fingerprint: float = 2.0
    source_title_proximity: float = 0.75
    intent_boost: float = 0.65
    min_score: float = 1.0
    pre_rerank_multiplier: int = 3


@dataclass(slots=True)
class RetrievalHit:
    concept: ConceptCard
    score: float
    reasons: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    source_pointer_ids: list[str] = field(default_factory=list)
    domain: str = "general"
    confidence: float = 0.0


@dataclass(slots=True)
class RetrievalResult:
    hits: list[RetrievalHit]
    estimated_ram_bytes: int
    degradation_decisions: list[str] = field(default_factory=list)


class HybridRetriever:
    def __init__(self, concepts: list[ConceptCard], scoring: RetrievalScoring | None = None) -> None:
        self.concepts = {card.concept_id: card for card in concepts}
        self.lexical = LexicalIndex.from_concepts(concepts)
        self.scoring = scoring or RetrievalScoring()

    def search(self, query: str, budget: MemoryBudget, limit: int = 100, intent: QueryIntent | None = None) -> RetrievalResult:
        adjusted_limit = budget.clamp_candidates(limit)
        decisions: list[str] = []
        if adjusted_limit < limit:
            decisions.append(f"reduced candidate concepts from {limit} to {adjusted_limit}")
        pre_limit = min(len(self.concepts), max(adjusted_limit, adjusted_limit * self.scoring.pre_rerank_multiplier))
        lexical = self.lexical.search(query, pre_limit)
        q_fp = fingerprint_text(query)
        q_terms = set(tokenize(query))
        q_lower = query.lower()
        hits: list[RetrievalHit] = []
        for cid, (lex_score, reasons) in lexical.items():
            card = self.concepts[cid]
            name_terms = set(tokenize(card.canonical_name + " " + " ".join(card.aliases)))
            matched = sorted(q_terms.intersection(name_terms))
            score = lex_score * self.scoring.lexical
            hit_reasons = list(reasons)
            if card.canonical_name.lower() in q_lower:
                score += self.scoring.exact_title
                hit_reasons.append("exact title match")
            if any(alias.lower() in q_lower for alias in card.aliases):
                score += self.scoring.alias
                hit_reasons.append("alias match")
            if intent is not None and card.domain_id in intent.needed_domains:
                score += self.scoring.domain_prior
                hit_reasons.append("domain prior")
                if intent.intent_type.value in {"CAUSE_EFFECT", "COMPARISON", "COUNTERFACTUAL"}:
                    score += self.scoring.intent_boost
                    hit_reasons.append("query intent boost")
            if matched and any(term in card.canonical_name.lower() for term in matched):
                score += self.scoring.source_title_proximity
                hit_reasons.append("source/title proximity")
            sim = fingerprint_similarity(q_fp, card.short_fingerprint)
            if budget.profile.max_ram_bytes >= 128 * 1024 * 1024:
                score += sim * self.scoring.fingerprint
            if sim >= 0.50:
                hit_reasons.append("fingerprint match")
            confidence = min(1.0, max(0.05, score / 15.0))
            if score >= self.scoring.min_score:
                hits.append(
                    RetrievalHit(
                        concept=card,
                        score=score,
                        reasons=sorted(set(hit_reasons)),
                        matched_terms=matched,
                        source_pointer_ids=card.source_refs[:8],
                        domain=domain_name(card.domain_id),
                        confidence=confidence,
                    )
                )
        hits.sort(key=lambda hit: (-hit.score, hit.concept.concept_id))
        hits = hits[:adjusted_limit]
        estimated = budget.estimate_query_memory(len(hits), 0, 1)
        return RetrievalResult(hits=hits, estimated_ram_bytes=estimated, degradation_decisions=decisions)
