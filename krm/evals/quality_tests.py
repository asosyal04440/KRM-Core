from __future__ import annotations


def has_grounded_answer_shape(answer: str) -> bool:
    return len(answer.strip()) > 40
