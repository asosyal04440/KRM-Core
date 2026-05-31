from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from krm.concepts.domain import DOMAIN_IDS, detect_domains


class IntentType(str, Enum):
    FACTUAL = "FACTUAL"
    EXPLANATION = "EXPLANATION"
    COMPARISON = "COMPARISON"
    CAUSE_EFFECT = "CAUSE_EFFECT"
    TIMELINE = "TIMELINE"
    HOW_TO = "HOW_TO"
    CREATIVE = "CREATIVE"
    EDUCATIONAL = "EDUCATIONAL"
    COUNTERFACTUAL = "COUNTERFACTUAL"
    PLANNING = "PLANNING"
    CODE = "CODE"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class QueryIntent:
    intent_type: IntentType
    needed_domains: list[int]
    broadness: float
    hallucination_risk: float
    need_source_grounding: bool
    need_verifier: bool
    expected_answer_structure: str
    memory_intensity: float


class QueryRouter:
    def classify(self, query: str) -> QueryIntent:
        q = query.lower()
        domains = {score.domain_id for score in detect_domains(query, max_domains=4)}
        domains.add(DOMAIN_IDS["general"])
        if "compare" in q or "versus" in q or "difference" in q:
            intent = IntentType.COMPARISON
            structure = "comparison"
        elif "if " in q or "might have" in q or "counterfactual" in q:
            intent = IntentType.COUNTERFACTUAL
            structure = "labeled scenarios"
        elif q.startswith("why") or "cause" in q:
            intent = IntentType.CAUSE_EFFECT
            structure = "multi-factor explanation"
        elif q.startswith("how"):
            intent = IntentType.HOW_TO
            structure = "steps"
        else:
            intent = IntentType.EXPLANATION
            structure = "grounded explanation"
        risk = 0.75 if intent == IntentType.COUNTERFACTUAL else 0.45
        return QueryIntent(
            intent_type=intent,
            needed_domains=sorted(domains),
            broadness=min(1.0, len(query.split()) / 24.0),
            hallucination_risk=risk,
            need_source_grounding=True,
            need_verifier=False,
            expected_answer_structure=structure,
            memory_intensity=0.35 + 0.1 * len(domains),
        )
