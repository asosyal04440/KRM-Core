from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RubricResult:
    query: str
    score: float
    passed: bool
    missing_concepts: list[str]
    missing_structure: list[str]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "score": self.score,
            "passed": self.passed,
            "missing_concepts": self.missing_concepts,
            "missing_structure": self.missing_structure,
            "recommendations": self.recommendations,
        }


def is_demo_query(query: str) -> bool:
    q = query.lower()
    return "industrial revolution" in q or "photosynthesis" in q or "ottoman empire" in q


def evaluate_demo_answer(query: str, answer: str, plan: dict[str, Any]) -> RubricResult:
    q = query.lower()
    text = (answer + " " + " ".join(plan.get("key_concepts", [])) + " " + plan.get("thesis", "")).lower()
    if "industrial revolution" in q:
        required = ["britain", "coal", "steam engine", "textile industry", "capital", "agriculture", "labor", "trade", "transport"]
        structure = {
            "multiple causes": any(term in text for term in ["several", "multiple", "reinforcing", "system"]),
            "no single-cause framing": any(term in text for term in ["no single", "rather than one", "not one"]),
        }
    elif "photosynthesis" in q:
        required = ["photosynthesis", "cellular respiration", "glucose", "oxygen", "carbon dioxide", "energy"]
        structure = {
            "defines both": "photosynthesis" in text and "cellular respiration" in text,
            "compares input/output": "input" in text or ("uses" in text and "outputs" in text) or "produces" in text,
            "explains relationship": "relationship" in text or "complementary" in text,
        }
    else:
        required = ["printing press", "ottoman empire"]
        structure = {
            "speculation label": "speculative" in text or "not a confirmed" in text,
            "multiple scenarios": "scenario a" in text and "scenario b" in text,
            "avoids certainty": "not a certain" in text or "could" in text or "might" in text,
        }
    missing_concepts = [concept for concept in required if concept not in text]
    missing_structure = [name for name, ok in structure.items() if not ok]
    concept_score = (len(required) - len(missing_concepts)) / max(1, len(required))
    structure_score = (len(structure) - len(missing_structure)) / max(1, len(structure))
    score = round((concept_score * 0.6 + structure_score * 0.4) * 100, 1)
    recs = []
    if missing_concepts:
        recs.append("boost missing required concepts in retrieval/resonance")
    if missing_structure:
        recs.append("strengthen answer plan section requirements")
    return RubricResult(query, score, score >= 80.0, missing_concepts, missing_structure, recs)
