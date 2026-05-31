from __future__ import annotations

from krm.llm.mock_model import MockModel
from krm.reasoning.answer_plan import AnswerPlan
from krm.reasoning.router import QueryIntent


class Composer:
    def __init__(self, model: MockModel | None = None) -> None:
        self.model = model or MockModel()

    def compose(self, query: str, plan: AnswerPlan, intent: QueryIntent, snippets: list[str]) -> str:
        return self.model.generate(query, plan, intent, snippets)
