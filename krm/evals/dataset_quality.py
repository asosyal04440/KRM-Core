from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from krm.source.dataset_bridge import DatasetBridge, summarize_dataset_files
from krm.source.dataset_readers import DatasetReaderConfig, reader_for_path
from krm.source.schema_detector import detect_schema


def dataset_quality_report(source: Path, sample_rows: int = 100, max_file_mb: float = 50.0) -> dict[str, Any]:
    files = DatasetBridge().scan(source)
    summary = summarize_dataset_files(files)
    reports = []
    for item in files:
        if not item.ingestible:
            reports.append({"path": str(item.path), "supported": item.supported, "ingestible": False, "reason": item.reason})
            continue
        schema = detect_schema(item.path, sample_rows=5, max_file_mb=max_file_mb)
        row_count = 0
        empty_count = 0
        total_len = 0
        seen: set[str] = set()
        duplicates = 0
        coverage: dict[str, int] = {field: 0 for field in schema.fields}
        if item.suffix in {".jsonl", ".json", ".csv", ".tsv"}:
            try:
                reader = reader_for_path(item.path, DatasetReaderConfig(max_rows=sample_rows), schema)
                for row in reader.iter_rows():
                    row_count += 1
                    joined = " ".join(value for value in row.raw_fields.values() if value)
                    total_len += len(joined)
                    if not joined.strip():
                        empty_count += 1
                    digest = hashlib.blake2b(joined.encode("utf-8"), digest_size=8).hexdigest()
                    if digest in seen:
                        duplicates += 1
                    seen.add(digest)
                    for field, value in row.raw_fields.items():
                        if value:
                            coverage[field] = coverage.get(field, 0) + 1
            except Exception as exc:
                schema.warnings.append(f"quality sample failed: {exc}")
        reports.append(
            {
                "path": str(item.path),
                "suffix": item.suffix,
                "ingestible": item.ingestible,
                "schema": schema.to_dict(),
                "rows_sampled": row_count,
                "empty_row_rate": round(empty_count / row_count, 3) if row_count else 0.0,
                "duplicate_row_estimate": duplicates,
                "average_text_length": round(total_len / row_count, 1) if row_count else 0.0,
                "field_coverage": coverage,
                "warnings": schema.warnings,
                "recommended_mapping_overrides": _recommended_overrides(schema),
            }
        )
    return {"source": str(source), "summary": summary, "files": reports}


def _recommended_overrides(schema) -> list[str]:
    recs = []
    if schema.confidence < 0.5:
        recs.append("use --title-field/--text-field or QA/instruction field overrides")
    if schema.task_type.value == "UNKNOWN":
        recs.append("set --task-type explicitly")
    return recs
