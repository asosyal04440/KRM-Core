from __future__ import annotations

from krm.source.dataset_readers import DatasetRow
from krm.source.schema_detector import DatasetTaskType
from krm.source.source_pointer import SourceArticle


def dataset_row_to_article(row: DatasetRow) -> SourceArticle:
    if row.task_type == DatasetTaskType.QA:
        title = row.question or row.title or row.row_id
        text = f"{row.question or ''}\n\nAnswer:\n{row.answer or ''}".strip()
    elif row.task_type == DatasetTaskType.INSTRUCTION:
        title = row.instruction or row.title or row.row_id
        parts = [row.instruction or "", row.input_text or "", row.output_text or row.answer or ""]
        text = "\n\n".join(part for part in parts if part).strip()
    elif row.task_type == DatasetTaskType.CLASSIFICATION:
        title = (row.text or row.input_text or row.title or row.row_id)[:80]
        label = row.output_text or row.answer or row.raw_fields.get("label") or row.raw_fields.get("class") or ""
        text = "\n\n".join(part for part in [row.text or row.input_text or "", f"Label: {label}" if label else ""] if part).strip()
    else:
        title = row.title or row.question or row.instruction or row.row_id
        text = row.text or row.input_text or row.question or row.instruction or ""
        if row.task_type == DatasetTaskType.UNKNOWN and row.answer and row.question:
            text = f"{row.question}\n\nAnswer:\n{row.answer}"
    return SourceArticle(
        source_id=row.source_id,
        source_type=row.source_pointer.source_type,
        article_id=row.row_id,
        title=title,
        text=text,
        path=None,
        metadata={**row.source_pointer.extra, "dataset_title": row.title},
    )
