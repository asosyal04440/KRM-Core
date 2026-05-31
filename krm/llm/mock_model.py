from __future__ import annotations

from krm.reasoning.answer_plan import AnswerPlan
from krm.reasoning.router import IntentType, QueryIntent


class MockModel:
    def generate(self, query: str, plan: AnswerPlan, intent: QueryIntent, snippets: list[str]) -> str:
        concepts = {name.lower(): name for name in plan.key_concepts}
        def concept(name: str) -> str:
            return concepts.get(name.lower(), name)

        source_note = "Based on the available local sample sources, " if snippets else "With limited local source coverage, "
        uncertainty = ""
        if plan.uncertainty_notes:
            uncertainty = "\n\nUncertainty: " + "; ".join(plan.uncertainty_notes[:2]) + "."
        if intent.intent_type == IntentType.COMPARISON:
            return (
                f"{plan.thesis}\n\n"
                f"{source_note}"
                f"Photosynthesis uses light, carbon dioxide, and water to make glucose and oxygen, usually in chloroplasts. "
                f"Cellular respiration uses glucose and oxygen to release usable energy, producing carbon dioxide and water, often in mitochondria. "
                f"The relationship is complementary: one stores energy in glucose, while the other releases energy from glucose. "
                f"Grounding concepts: {', '.join(plan.key_concepts[:8])}."
                f"{uncertainty}"
            )
        if intent.intent_type == IntentType.COUNTERFACTUAL:
            return (
                f"{plan.thesis}\n\n"
                "This is a speculative scenario, not a confirmed historical outcome. "
                "Scenario A: wider use of the printing press could have increased literacy, education, bureaucracy, and technical exchange. "
                "Scenario B: social resistance, censorship, or craft interests could have limited the impact. "
                "Scenario C: reform capacity might have improved if printed manuals and records spread through administration. "
                "This remains speculation, not a certain alternate history."
                f"{uncertainty}"
            )
        needed = ["Britain", "coal", "steam engine", "textile industry", "capital", "agriculture", "labor", "trade", "transport"]
        grounded = [concept(name) for name in needed]
        return (
            f"{plan.thesis}\n\n"
            f"{source_note}the strongest explanation is a cluster: {grounded[0]} had accessible {grounded[1]}, and {grounded[1]} powered the "
            f"{grounded[2]}, which helped mines, {grounded[3]}, and later {grounded[8]}. "
            f"{grounded[7]} and {grounded[4]} helped finance factories and expand markets. "
            f"Changes in {grounded[5]} increased food output and moved some {grounded[6]} toward towns. "
            "Together these factors reinforced each other, so the answer is a system of causes rather than one magic trigger."
            f"{uncertainty}"
        )
