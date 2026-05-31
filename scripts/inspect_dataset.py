from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.source.dataset_bridge import DatasetBridge, summarize_dataset_files
from krm.source.schema_detector import detect_schema


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect local dataset files without downloading or ingesting.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--no-recursive", action="store_false", dest="recursive")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--sample-rows", type=int, default=5)
    parser.add_argument("--max-file-mb", type=float, default=50.0)
    args = parser.parse_args()
    files = DatasetBridge().scan(args.source, recursive=args.recursive)
    if args.max_files is not None:
        files = files[: args.max_files]
    records = []
    for item in files:
        schema = None
        preview: list[dict[str, str]] = []
        warnings: list[str] = []
        if item.ingestible:
            schema = detect_schema(item.path, sample_rows=args.sample_rows, max_file_mb=args.max_file_mb)
            warnings.extend(schema.warnings)
            preview = _preview_rows(item.path, item.suffix, args.sample_rows, args.max_file_mb)
        records.append(
            {
                **item.to_dict(),
                "schema": schema.to_dict() if schema else None,
                "preview": preview,
                "warnings": warnings,
                "recommendation": _recommendation(item, schema),
            }
        )
    payload = {"source": str(args.source), "summary": summarize_dataset_files(files), "files": records}
    if args.json_output:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0
    print(f"source: {args.source}")
    for key, value in payload["summary"].items():
        print(f"{key}: {value}")
    print("detected dataset files:")
    for record in records:
        print(f"- {record['path']} suffix={record['suffix']} size_mb={record['size_mb']} ingestible={record['ingestible']} reason={record['reason']}")
        if record["schema"]:
            schema = record["schema"]
            print(f"  fields: {', '.join(schema['fields'])}")
            print(f"  task_type: {schema['task_type']} confidence={schema['confidence']}")
            print(f"  guessed mapping: title={schema['guessed_title_field']} text={schema['guessed_text_field']} question={schema['guessed_question_field']} answer={schema['guessed_answer_field']} instruction={schema['guessed_instruction_field']} input={schema['guessed_input_field']} output={schema['guessed_output_field']}")
        for warning in record["warnings"]:
            print(f"  warning: {warning}")
        print(f"  recommendation: {record['recommendation']}")
    return 0


def _preview_rows(path: Path, suffix: str, sample_rows: int, max_file_mb: float) -> list[dict[str, str]]:
    if path.stat().st_size > int(max_file_mb * 1024 * 1024):
        return []
    try:
        if suffix == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if len(rows) >= sample_rows:
                        break
                    if line.strip():
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            rows.append(_truncate_row(obj))
            return rows
        if suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(obj, dict):
                for key in ["data", "rows", "examples"]:
                    if isinstance(obj.get(key), list):
                        return [_truncate_row(row) for row in obj[key][:sample_rows] if isinstance(row, dict)]
                return [_truncate_row(obj)]
            if isinstance(obj, list):
                return [_truncate_row(row) for row in obj[:sample_rows] if isinstance(row, dict)]
        if suffix in {".csv", ".tsv"}:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                reader = csv.DictReader(fh, delimiter="\t" if suffix == ".tsv" else ",")
                return [_truncate_row(row) for _, row in zip(range(sample_rows), reader, strict=False)]
    except Exception:
        return []
    return []


def _truncate_row(row: dict[str, Any], max_chars: int = 120) -> dict[str, str]:
    return {str(key): str(value)[:max_chars] for key, value in row.items()}


def _recommendation(item, schema) -> str:
    if not item.ingestible:
        return item.reason
    if schema is None:
        return "inspect schema before ingest"
    if schema.confidence < 0.5:
        return "use explicit field mapping overrides"
    return "ready for dry-run ingestion"


if __name__ == "__main__":
    raise SystemExit(main())
