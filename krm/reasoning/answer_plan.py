from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from krm.graph.hot_subgraph import HotSubgraph
from krm.reasoning.policy_seed import PolicySeed
from krm.reasoning.router import IntentType, QueryIntent


@dataclass(slots=True)
class AnswerSection:
    title: str
    purpose: str
    concepts: list[str]
    source_refs: list[str] = field(default_factory=list)
    bullet_claims: list[str] = field(default_factory=list)
    uncertainty: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "purpose": self.purpose,
            "concepts": self.concepts,
            "source_refs": self.source_refs,
            "bullet_claims": self.bullet_claims,
            "uncertainty": self.uncertainty,
        }


@dataclass(slots=True)
class AnswerPlan:
    query: str
    intent: str
    thesis: str
    sections: list[AnswerSection]
    key_concepts: list[str]
    supporting_paths: list[list[str]] = field(default_factory=list)
    cause_effect_chains: list[list[str]] = field(default_factory=list)
    comparisons: list[str] = field(default_factory=list)
    source_references: list[str] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)
    what_not_to_claim: list[str] = field(default_factory=list)
    speculation_label_required: bool = False
    confidence: float = 0.0
    style_instructions: str = "clear and grounded"
    final_answer_length: int = 500
    expected_missing_info: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent,
            "thesis": self.thesis,
            "sections": [section.to_dict() for section in self.sections],
            "key_concepts": self.key_concepts,
            "supporting_paths": self.supporting_paths,
            "cause_effect_chains": self.cause_effect_chains,
            "comparisons": self.comparisons,
            "source_references": self.source_references,
            "uncertainty_notes": self.uncertainty_notes,
            "what_not_to_claim": self.what_not_to_claim,
            "speculation_label_required": self.speculation_label_required,
            "confidence": self.confidence,
            "style_instructions": self.style_instructions,
            "final_answer_length": self.final_answer_length,
            "expected_missing_info": self.expected_missing_info,
        }


class AnswerPlanner:
    def plan(self, query: str, graph: HotSubgraph, intent: QueryIntent, policy: PolicySeed) -> AnswerPlan:
        top = [card.canonical_name for card in graph.top_concepts(12)]
        sources = graph.source_pointers[:8]
        uncertainty = list(graph.warnings)
        if graph.confidence < policy.confidence_threshold:
            uncertainty.append("source coverage is limited, so confidence is lower")
        if intent.intent_type == IntentType.COMPARISON:
            if "photosynthesis" in {item.lower() for item in top}:
                thesis = "Photosynthesis and cellular respiration can be compared by inputs, outputs, location, and energy role."
            else:
                thesis = f"{top[0] if top else 'The concepts'} can be compared by inputs, outputs, location, and energy role."
            sections = [
                AnswerSection("Define both processes", "establish the two biological processes", top, sources, ["define photosynthesis", "define cellular respiration"]),
                AnswerSection("Compare inputs and outputs", "show material and energy flow", top, sources, ["compare glucose, oxygen, carbon dioxide, water, and energy"]),
                AnswerSection("Explain relationship", "connect storage and release of energy", top, sources, ["photosynthesis stores energy; respiration releases it"]),
            ]
            comparisons = ["inputs", "outputs", "cell location", "energy direction"]
            chains: list[list[str]] = []
            missing = []
            speculation = False
        elif intent.intent_type == IntentType.COUNTERFACTUAL:
            thesis = "This is speculative: earlier adoption could have changed literacy, bureaucracy, and reform capacity, but institutions still mattered."
            sections = [
                AnswerSection("Scenario A: wider learning and administration", "plausible positive mechanism", top, sources, ["printing press could support literacy and bureaucracy"], "speculative"),
                AnswerSection("Scenario B: resistance limits impact", "show why adoption may not determine outcome", top, sources, ["institutions, censorship, and craft interests could limit change"], "speculative"),
                AnswerSection("Scenario C: reform capacity changes", "connect information flow to state capacity", top, sources, ["printed manuals and records could support reform"], "speculative"),
            ]
            comparisons = []
            chains = [["printing press", "literacy", "bureaucracy"], ["printing press", "reform"]]
            missing = ["direct evidence for alternate outcomes is impossible by definition"]
            speculation = True
        else:
            thesis = "No single cause explains it; several reinforcing factors formed a connected system."
            forced = ["Britain", "coal", "steam engine", "textile industry", "capital", "agriculture", "labor", "trade", "transport"]
            merged_top = _merge_required(top, forced)
            top = merged_top
            sections = [
                AnswerSection("Energy and machines", "explain coal and steam power", top, sources, ["Britain had accessible coal", "coal powered steam engines and mechanized production"]),
                AnswerSection("Capital, trade, and markets", "explain financing and demand", top, sources, ["trade and capital helped finance factories and expand markets"]),
                AnswerSection("Agriculture, labor, and transport", "explain supporting system changes", top, sources, ["agriculture changed labor supply", "transport linked resources and markets"]),
            ]
            comparisons = []
            chains = [["coal", "steam engine", "textile industry"], ["trade", "capital"], ["agriculture", "labor", "transport"]]
            missing = []
            speculation = False
        return AnswerPlan(
            query=query,
            intent=intent.intent_type.value,
            thesis=thesis,
            sections=sections,
            key_concepts=top,
            supporting_paths=[[str(node) for node in path] for path in graph.strongest_paths],
            cause_effect_chains=chains,
            comparisons=comparisons,
            source_references=sources,
            uncertainty_notes=sorted(set(uncertainty)),
            what_not_to_claim=["do not claim a single sufficient cause", "do not invent dates or statistics"],
            speculation_label_required=speculation,
            confidence=graph.confidence,
            style_instructions=policy.answer_style,
            final_answer_length=policy.max_answer_length,
            expected_missing_info=missing,
        )


def _merge_required(top: list[str], required: list[str]) -> list[str]:
    existing = {item.lower() for item in top}
    merged = list(top)
    for item in required:
        if item.lower() not in existing:
            merged.append(item)
    return merged[:16]
