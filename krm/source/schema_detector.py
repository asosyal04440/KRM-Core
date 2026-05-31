from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class DatasetTaskType(str, Enum):
    PLAIN_TEXT = "PLAIN_TEXT"
    QA = "QA"
    INSTRUCTION = "INSTRUCTION"
    CHAT = "CHAT"
    CLASSIFICATION = "CLASSIFICATION"
    UNKNOWN = "UNKNOWN"


TITLE_FIELDS = ["title", "heading", "name", "topic", "subject"]
TEXT_FIELDS = ["text", "content", "body", "passage", "document", "article", "context", "paragraph"]
QUESTION_FIELDS = ["question", "query", "prompt", "q"]
ANSWER_FIELDS = ["answer", "response", "completion", "output", "a"]
INSTRUCTION_FIELDS = ["instruction", "task", "system", "prompt"]
INPUT_FIELDS = ["input", "context", "source"]
OUTPUT_FIELDS = ["output", "completion", "response", "answer"]
LABEL_FIELDS = ["label", "class", "category", "target"]


@dataclass(slots=True)
class DatasetSchema:
    file_path: Path
    format: str
    fields: list[str]
    row_count_estimate: int | None
    guessed_title_field: str | None
    guessed_text_field: str | None
    guessed_question_field: str | None
    guessed_answer_field: str | None
    guessed_instruction_field: str | None
    guessed_input_field: str | None
    guessed_output_field: str | None
    confidence: float
    warnings: list[str] = field(default_factory=list)
    task_type: DatasetTaskType = DatasetTaskType.UNKNOWN
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["file_path"] = str(self.file_path)
        data["task_type"] = self.task_type.value
        return data


def detect_schema(path: Path, sample_rows: int = 5, max_file_mb: float = 50.0) -> DatasetSchema:
    suffix = path.suffix.lower().lstrip(".")
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    max_bytes = int(max_file_mb * 1024 * 1024)
    if path.stat().st_size > max_bytes and suffix == "json":
        warnings.append("JSON file exceeds inspection cap; use JSONL or increase cap")
        return _schema(path, suffix, [], None, warnings, [])
    try:
        if suffix == "jsonl":
            rows = _sample_jsonl(path, sample_rows, warnings)
        elif suffix == "json":
            rows = _sample_json(path, sample_rows, warnings)
        elif suffix in {"csv", "tsv"}:
            rows = _sample_delimited(path, sample_rows, "\t" if suffix == "tsv" else ",", warnings)
        elif suffix in {"txt", "md", "markdown", "html", "htm"}:
            fields = ["text"]
            return _schema(path, suffix, fields, 1, warnings, ["plain text file mapped to text"], DatasetTaskType.PLAIN_TEXT)
        else:
            warnings.append(f"unsupported schema format: {suffix}")
    except Exception as exc:
        warnings.append(f"schema inspection failed: {exc}")
    fields = sorted({str(key) for row in rows for key in row.keys()})
    return _schema(path, suffix, fields, len(rows) if rows else None, warnings, rows)


def _schema(
    path: Path,
    fmt: str,
    fields: list[str],
    row_count: int | None,
    warnings: list[str],
    rows_or_reasons: list[Any],
    forced_task: DatasetTaskType | None = None,
) -> DatasetSchema:
    lower_map = {field.lower(): field for field in fields}
    reasons: list[str] = [item for item in rows_or_reasons if isinstance(item, str)]
    title = _pick(lower_map, TITLE_FIELDS, reasons, "title")
    text = _pick(lower_map, TEXT_FIELDS, reasons, "text")
    question = _pick(lower_map, QUESTION_FIELDS, reasons, "question")
    answer = _pick(lower_map, ANSWER_FIELDS, reasons, "answer")
    instruction = _pick(lower_map, INSTRUCTION_FIELDS, reasons, "instruction")
    input_field = _pick(lower_map, INPUT_FIELDS, reasons, "input")
    output = _pick(lower_map, OUTPUT_FIELDS, reasons, "output")
    task = forced_task or _task_type(fields, question, answer, instruction, input_field, output, text)
    guessed = [value for value in [title, text, question, answer, instruction, input_field, output] if value]
    confidence = round(min(1.0, 0.2 + 0.12 * len(set(guessed))), 2) if fields else 0.0
    if not fields:
        warnings.append("no fields detected")
    if task == DatasetTaskType.UNKNOWN:
        warnings.append("unable to infer dataset task type")
    return DatasetSchema(
        file_path=path,
        format=fmt,
        fields=fields,
        row_count_estimate=row_count,
        guessed_title_field=title,
        guessed_text_field=text,
        guessed_question_field=question,
        guessed_answer_field=answer,
        guessed_instruction_field=instruction,
        guessed_input_field=input_field,
        guessed_output_field=output,
        confidence=confidence,
        warnings=sorted(set(warnings)),
        task_type=task,
        reasons=reasons,
    )


def _pick(lower_map: dict[str, str], candidates: list[str], reasons: list[str], label: str) -> str | None:
    for candidate in candidates:
        if candidate in lower_map:
            field = lower_map[candidate]
            reasons.append(f"guessed {label} field from `{field}`")
            return field
    return None


def _task_type(
    fields: list[str],
    question: str | None,
    answer: str | None,
    instruction: str | None,
    input_field: str | None,
    output: str | None,
    text: str | None,
) -> DatasetTaskType:
    lower = {field.lower() for field in fields}
    if {"messages", "conversation", "chat"}.intersection(lower):
        return DatasetTaskType.CHAT
    if question and answer:
        return DatasetTaskType.QA
    if instruction and (output or answer):
        return DatasetTaskType.INSTRUCTION
    if text and lower.intersection(LABEL_FIELDS):
        return DatasetTaskType.CLASSIFICATION
    if text or input_field:
        return DatasetTaskType.PLAIN_TEXT
    return DatasetTaskType.UNKNOWN


def _sample_jsonl(path: Path, limit: int, warnings: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_number, line in enumerate(fh, start=1):
            if len(rows) >= limit:
                break
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                warnings.append(f"line {line_number} is invalid JSON")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _sample_json(path: Path, limit: int, warnings: list[str]) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(obj, list):
        return [row for row in obj[:limit] if isinstance(row, dict)]
    if isinstance(obj, dict):
        for key in ["data", "rows", "examples"]:
            value = obj.get(key)
            if isinstance(value, list):
                return [row for row in value[:limit] if isinstance(row, dict)]
        return [obj]
    warnings.append("JSON root is not an object/list")
    return []


def _sample_delimited(path: Path, limit: int, delimiter: str, warnings: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        if not reader.fieldnames:
            warnings.append("missing header row")
            return rows
        for row in reader:
            if len(rows) >= limit:
                break
            rows.append(dict(row))
    return rows
