from pathlib import Path

from krm.source.dataset_readers import DatasetRow
from krm.source.dataset_to_article import dataset_row_to_article
from krm.source.schema_detector import DatasetTaskType
from krm.source.source_pointer import SourcePointer


def _row(task: DatasetTaskType) -> DatasetRow:
    pointer = SourcePointer("datasets", "dataset_jsonl", "T", "row-1", extra={"dataset_path": "x", "row_id": "row-1", "task_type": task.value})
    return DatasetRow("datasets", "row-1", "Title", "Body", "Question?", "Answer.", "Instruction.", "Input.", "Output.", {}, pointer, task)


def test_dataset_row_to_article_shapes_and_pointer_metadata() -> None:
    qa = dataset_row_to_article(_row(DatasetTaskType.QA))
    assert qa.title == "Question?"
    assert "Answer:" in qa.text
    assert qa.metadata["row_id"] == "row-1"
    inst = dataset_row_to_article(_row(DatasetTaskType.INSTRUCTION))
    assert "Instruction." in inst.text
    plain = dataset_row_to_article(_row(DatasetTaskType.PLAIN_TEXT))
    assert plain.title == "Title"
    assert plain.text == "Body"
