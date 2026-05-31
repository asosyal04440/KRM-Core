from __future__ import annotations

from enum import Enum


class ConceptType(str, Enum):
    ENTITY = "ENTITY"
    EVENT = "EVENT"
    PROCESS = "PROCESS"
    PROPERTY = "PROPERTY"
    CAUSE = "CAUSE"
    EFFECT = "EFFECT"
    FIELD = "FIELD"
    PERSON = "PERSON"
    PLACE = "PLACE"
    TIME_PERIOD = "TIME_PERIOD"
    TOOL = "TOOL"
    THEORY = "THEORY"
    UNKNOWN = "UNKNOWN"
