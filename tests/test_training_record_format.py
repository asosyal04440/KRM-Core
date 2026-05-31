import pytest

from krm.training.record_format import TrainingRecord, TrainingRecordError, TrainingTaskType, make_record


def test_training_record_serialization_and_deterministic_id() -> None:
    record = make_record(TrainingTaskType.INTENT_ROUTING, "query", '{"intent":"FACTUAL"}', {"confidence": 0.9})
    loaded = TrainingRecord.from_jsonl_line(record.to_jsonl_line())
    assert loaded.record_id == record.record_id
    assert loaded.task == TrainingTaskType.INTENT_ROUTING


def test_training_record_rejects_invalid_task_and_length() -> None:
    with pytest.raises(TrainingRecordError):
        TrainingRecord.from_jsonl_line('{"record_id":"x","task":"NOPE","input":"a","target":"b","metadata":{}}')
    record = make_record(TrainingTaskType.LM_CONTINUATION, "abc", "target")
    with pytest.raises(TrainingRecordError):
        record.validate(max_input_chars=2)
