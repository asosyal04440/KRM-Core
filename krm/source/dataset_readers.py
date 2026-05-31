from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from krm.source.schema_detector import DatasetSchema, DatasetTaskType, detect_schema
from krm.source.source_pointer import SourcePointer


@dataclass(slots=True)
class DatasetRow:
    source_id: str
    row_id: str
    title: str | None
    text: str | None
    question: str | None
    answer: str | None
    instruction: str | None
    input_text: str | None
    output_text: str | None
    raw_fields: dict[str, str]
    source_pointer: SourcePointer
    task_type: DatasetTaskType


@dataclass(slots=True)
class DatasetReaderConfig:
    max_rows: int = 10_000
    max_field_chars: int = 20_000
    title_field: str | None = None
    text_field: str | None = None
    question_field: str | None = None
    answer_field: str | None = None
    instruction_field: str | None = None
    input_field: str | None = None
    output_field: str | None = None
    task_type: DatasetTaskType | None = None


class BaseDatasetReader:
    def __init__(self, path: Path, config: DatasetReaderConfig | None = None, schema: DatasetSchema | None = None) -> None:
        self.path = Path(path)
        self.config = config or DatasetReaderConfig()
        self.schema = schema or detect_schema(self.path)
        self.warnings: list[str] = []

    def iter_rows(self) -> Iterator[DatasetRow]:
        count = 0
        for idx, raw in enumerate(self._raw_rows(), start=1):
            if count >= self.config.max_rows:
                self.warnings.append(f"stopped at max rows cap: {self.config.max_rows}")
                return
            row = self._to_row(idx, raw)
            if row is None:
                continue
            count += 1
            yield row

    def _raw_rows(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError

    def _to_row(self, idx: int, raw: dict[str, Any]) -> DatasetRow | None:
        fields = {str(key): self._truncate(value) for key, value in raw.items() if value is not None}
        title_field = self.config.title_field or self.schema.guessed_title_field
        text_field = self.config.text_field or self.schema.guessed_text_field
        question_field = self.config.question_field or self.schema.guessed_question_field
        answer_field = self.config.answer_field or self.schema.guessed_answer_field
        instruction_field = self.config.instruction_field or self.schema.guessed_instruction_field
        input_field = self.config.input_field or self.schema.guessed_input_field
        output_field = self.config.output_field or self.schema.guessed_output_field
        title = fields.get(title_field or "")
        text = fields.get(text_field or "")
        question = fields.get(question_field or "")
        answer = fields.get(answer_field or "")
        instruction = fields.get(instruction_field or "")
        input_text = fields.get(input_field or "")
        output_text = fields.get(output_field or "")
        if not any([title, text, question, answer, instruction, input_text, output_text]):
            self.warnings.append(f"{self.path}: row {idx} skipped because mapped fields are empty")
            return None
        task = self.config.task_type or self.schema.task_type
        row_id = f"{self.path.stem}-{idx}"
        pointer = SourcePointer(
            source_id=self.path.parent.name,
            source_type=f"dataset_{self.path.suffix.lower().lstrip('.')}",
            title=title or question or instruction or row_id,
            article_id=row_id,
            extra={"dataset_path": str(self.path.resolve()), "row_id": row_id, "task_type": task.value},
        )
        return DatasetRow(
            source_id=self.path.parent.name,
            row_id=row_id,
            title=title,
            text=text,
            question=question,
            answer=answer,
            instruction=instruction,
            input_text=input_text,
            output_text=output_text,
            raw_fields=fields,
            source_pointer=pointer,
            task_type=task,
        )

    def _truncate(self, value: Any) -> str:
        return str(value)[: self.config.max_field_chars]


class JsonlDatasetReader(BaseDatasetReader):
    def _raw_rows(self) -> Iterator[dict[str, Any]]:
        with self.path.open("r", encoding="utf-8", errors="replace") as fh:
            for line_number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    self.warnings.append(f"{self.path}: line {line_number} invalid JSON")
                    continue
                if isinstance(obj, dict):
                    yield obj


class JsonDatasetReader(BaseDatasetReader):
    def _raw_rows(self) -> Iterator[dict[str, Any]]:
        obj = json.loads(self.path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(obj, list):
            for row in obj:
                if isinstance(row, dict):
                    yield row
            return
        if isinstance(obj, dict):
            for key in ["data", "rows", "examples"]:
                value = obj.get(key)
                if isinstance(value, list):
                    for row in value:
                        if isinstance(row, dict):
                            yield row
                    return
            yield obj


class CsvDatasetReader(BaseDatasetReader):
    delimiter = ","

    def _raw_rows(self) -> Iterator[dict[str, Any]]:
        with self.path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=self.delimiter)
            if not reader.fieldnames:
                self.warnings.append(f"{self.path}: missing header row")
                return
            for row in reader:
                yield dict(row)


class TsvDatasetReader(CsvDatasetReader):
    delimiter = "\t"


def reader_for_path(path: Path, config: DatasetReaderConfig | None = None, schema: DatasetSchema | None = None) -> BaseDatasetReader:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return JsonlDatasetReader(path, config, schema)
    if suffix == ".json":
        return JsonDatasetReader(path, config, schema)
    if suffix == ".csv":
        return CsvDatasetReader(path, config, schema)
    if suffix == ".tsv":
        return TsvDatasetReader(path, config, schema)
    raise ValueError(f"unsupported dataset reader suffix: {suffix}")
