from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ConfidenceEstimate:
    score: float
    reasons: list[str]
