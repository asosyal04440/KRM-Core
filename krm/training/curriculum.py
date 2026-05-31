from __future__ import annotations

from collections import Counter

from krm.training.record_format import TrainingRecord, TrainingTaskType


STAGE_BY_TASK: dict[TrainingTaskType, str] = {
    TrainingTaskType.LM_CONTINUATION: "STRUCTURE",
    TrainingTaskType.DO_NOT_CLAIM: "STRUCTURE",
    TrainingTaskType.CONCEPT_EXTRACTION: "CONCEPTS",
    TrainingTaskType.DOMAIN_CLASSIFICATION: "CONCEPTS",
    TrainingTaskType.INTENT_ROUTING: "CONCEPTS",
    TrainingTaskType.SHARD_ROUTING: "CONCEPTS",
    TrainingTaskType.EDGE_TYPING: "EDGES",
    TrainingTaskType.EDGE_SCORING: "EDGES",
    TrainingTaskType.RETRIEVAL_SCORING: "EDGES",
    TrainingTaskType.ANSWER_PLANNING: "PLANNING",
    TrainingTaskType.UNCERTAINTY_DETECTION: "PLANNING",
    TrainingTaskType.COUNTERFACTUAL_LABELING: "GROUNDED_ANSWERS",
    TrainingTaskType.GROUNDED_ANSWER: "GROUNDED_ANSWERS",
}

PROFILE_TASKS: dict[str, set[TrainingTaskType]] = {
    "tiny": set(TrainingTaskType),
    "small": set(TrainingTaskType),
    "router_only": {TrainingTaskType.INTENT_ROUTING, TrainingTaskType.DOMAIN_CLASSIFICATION, TrainingTaskType.SHARD_ROUTING},
    "planner_only": {TrainingTaskType.ANSWER_PLANNING, TrainingTaskType.UNCERTAINTY_DETECTION, TrainingTaskType.DO_NOT_CLAIM},
    "composer_only": {TrainingTaskType.GROUNDED_ANSWER, TrainingTaskType.COUNTERFACTUAL_LABELING, TrainingTaskType.DO_NOT_CLAIM},
}


def assign_curriculum_stage(record: TrainingRecord) -> str:
    return STAGE_BY_TASK.get(record.task, "STRUCTURE")


def build_curriculum_mix(records: list[TrainingRecord], profile: str = "tiny") -> list[TrainingRecord]:
    allowed = PROFILE_TASKS.get(profile, PROFILE_TASKS["tiny"])
    filtered = [record for record in records if record.task in allowed]
    filtered.sort(key=lambda record: (assign_curriculum_stage(record), record.task.value, record.record_id))
    if profile == "small":
        planner = [r for r in filtered if r.task in {TrainingTaskType.ANSWER_PLANNING, TrainingTaskType.GROUNDED_ANSWER}]
        rest = [r for r in filtered if r not in planner]
        return planner + rest
    return filtered


def curriculum_statistics(records: list[TrainingRecord]) -> dict[str, dict[str, int]]:
    return {
        "stages": dict(Counter(assign_curriculum_stage(record) for record in records)),
        "tasks": dict(Counter(record.task.value for record in records)),
    }
