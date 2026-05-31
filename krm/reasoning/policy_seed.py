from __future__ import annotations

from dataclasses import dataclass

from krm.reasoning.router import IntentType, QueryIntent


@dataclass(slots=True)
class PolicySeed:
    selected_domains: list[int]
    active_reasoning_modes: list[str]
    preferred_edge_types: list[str]
    answer_style: str
    source_strictness: str
    allowed_speculation_level: str
    confidence_threshold: float
    max_answer_length: int
    explanation_depth: str
    include_uncertainty: bool
    ask_for_clarification: bool
    verifier_needed: bool


class PolicySeedGenerator:
    def from_intent(self, intent: QueryIntent) -> PolicySeed:
        if intent.intent_type == IntentType.COUNTERFACTUAL:
            speculation = "medium_labeled"
            modes = ["scenario", "uncertainty"]
            edges = ["CAUSES", "ENABLES", "TEMPORAL_NEAR", "COUNTERFACTUAL_LINK"]
        elif intent.intent_type == IntentType.COMPARISON:
            speculation = "low"
            modes = ["compare", "contrast"]
            edges = ["CONTRASTS", "INPUT_OUTPUT", "EDUCATIONAL_PAIR", "DOMAIN_NEAR"]
        else:
            speculation = "low"
            modes = ["cause_effect", "synthesis"]
            edges = ["CAUSES", "CAUSED_BY", "ENABLES", "DEPENDS_ON", "DOMAIN_NEAR", "CO_OCCURS"]
        return PolicySeed(
            selected_domains=intent.needed_domains,
            active_reasoning_modes=modes,
            preferred_edge_types=edges,
            answer_style="clear, compact, grounded",
            source_strictness="medium_high",
            allowed_speculation_level=speculation,
            confidence_threshold=0.45,
            max_answer_length=650,
            explanation_depth="medium",
            include_uncertainty=True,
            ask_for_clarification=False,
            verifier_needed=intent.need_verifier,
        )
