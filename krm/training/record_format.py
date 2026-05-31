from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrainingRecordError(ValueError):
    pass


class TrainingTaskType(str, Enum):
    LM_CONTINUATION = "LM_CONTINUATION"
    CONCEPT_EXTRACTION = "CONCEPT_EXTRACTION"
    DOMAIN_CLASSIFICATION = "DOMAIN_CLASSIFICATION"
    INTENT_ROUTING = "INTENT_ROUTING"
    SHARD_ROUTING = "SHARD_ROUTING"
    EDGE_TYPING = "EDGE_TYPING"
    EDGE_SCORING = "EDGE_SCORING"
    RETRIEVAL_SCORING = "RETRIEVAL_SCORING"
    ANSWER_PLANNING = "ANSWER_PLANNING"
    GROUNDED_ANSWER = "GROUNDED_ANSWER"
    UNCERTAINTY_DETECTION = "UNCERTAINTY_DETECTION"
    COUNTERFACTUAL_LABELING = "COUNTERFACTUAL_LABELING"
    DO_NOT_CLAIM = "DO_NOT_CLAIM"


@dataclass(slots=True)
class TrainingRecord:
    task: TrainingTaskType
    input: str
    target: str
    record_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.task, str):
            self.task = TrainingTaskType(self.task)
        if not self.record_id:
            self.record_id = deterministic_record_id(self.task, self.input, self.target, self.metadata)
        self.validate()

    def validate(self, max_input_chars: int = 20000, max_target_chars: int = 20000) -> None:
        if not self.input.strip():
            raise TrainingRecordError("training record input must be non-empty")
        if not self.target.strip():
            raise TrainingRecordError("training record target must be non-empty")
        if len(self.input) > max_input_chars:
            raise TrainingRecordError(f"training record input exceeds {max_input_chars} characters")
        if len(self.target) > max_target_chars:
            raise TrainingRecordError(f"training record target exceeds {max_target_chars} characters")
        try:
            json.dumps(self.metadata, ensure_ascii=True, sort_keys=True)
        except TypeError as exc:
            raise TrainingRecordError("training record metadata must be JSON serializable") from exc
        expected = deterministic_record_id(self.task, self.input, self.target, self.metadata)
        if self.record_id != expected:
            raise TrainingRecordError("training record_id is not deterministic for the record content")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "task": self.task.value,
            "input": self.input,
            "target": self.target,
            "metadata": self.metadata,
        }

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True)

    @classmethod
    def from_jsonl_line(cls, line: str, max_input_chars: int = 20000, max_target_chars: int = 20000) -> "TrainingRecord":
        data = json.loads(line)
        try:
            task = TrainingTaskType(data["task"])
        except ValueError as exc:
            raise TrainingRecordError(f"unknown training task: {data.get('task')}") from exc
        record = cls(
            record_id=str(data.get("record_id") or ""),
            task=task,
            input=str(data.get("input") or ""),
            target=str(data.get("target") or ""),
            metadata=dict(data.get("metadata") or {}),
        )
        record.validate(max_input_chars=max_input_chars, max_target_chars=max_target_chars)
        return record


def deterministic_record_id(task: TrainingTaskType | str, input_text: str, target: str, metadata: dict[str, Any] | None = None) -> str:
    task_value = task.value if isinstance(task, TrainingTaskType) else str(task)
    payload = json.dumps(
        {
            "task": task_value,
            "input": input_text,
            "target": target,
            "metadata": metadata or {},
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()


def make_record(
    task: TrainingTaskType,
    input_text: str,
    target: str,
    metadata: dict[str, Any] | None = None,
    max_input_chars: int = 20000,
    max_target_chars: int = 20000,
) -> TrainingRecord:
    record = TrainingRecord(task=task, input=input_text[:max_input_chars], target=target[:max_target_chars], metadata=metadata or {})
    record.validate(max_input_chars=max_input_chars, max_target_chars=max_target_chars)
    return record
