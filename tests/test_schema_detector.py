from pathlib import Path

from krm.source.schema_detector import DatasetTaskType, detect_schema


def test_schema_detector_detects_core_mappings(tmp_path: Path) -> None:
    qa = tmp_path / "qa.jsonl"
    qa.write_text('{"question":"Q?","answer":"A","title":"T"}\n', encoding="utf-8")
    qa_schema = detect_schema(qa)
    assert qa_schema.guessed_question_field == "question"
    assert qa_schema.guessed_answer_field == "answer"
    assert qa_schema.guessed_title_field == "title"
    assert qa_schema.task_type == DatasetTaskType.QA

    inst = tmp_path / "inst.json"
    inst.write_text('{"examples":[{"instruction":"Do it","input":"X","output":"Y"}]}', encoding="utf-8")
    inst_schema = detect_schema(inst)
    assert inst_schema.guessed_instruction_field == "instruction"
    assert inst_schema.guessed_input_field == "input"
    assert inst_schema.guessed_output_field == "output"
    assert inst_schema.task_type == DatasetTaskType.INSTRUCTION


def test_schema_detector_unknown_warns(tmp_path: Path) -> None:
    path = tmp_path / "unknown.jsonl"
    path.write_text('{"x":"y"}\n', encoding="utf-8")
    schema = detect_schema(path)
    assert schema.task_type == DatasetTaskType.UNKNOWN
    assert schema.warnings
