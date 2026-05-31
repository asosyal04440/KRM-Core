from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.pipeline import ingest_articles
from krm.source.dataset_bridge import DatasetBridge, summarize_dataset_files
from krm.source.dataset_readers import DatasetReaderConfig, reader_for_path
from krm.source.dataset_to_article import dataset_row_to_article
from krm.source.schema_detector import DatasetTaskType, detect_schema
from krm.source.zim_reader import LocalFolderSourceReader


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest local dataset files into KRM-Core artifacts without downloads.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--profile", default="tiny")
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--no-recursive", action="store_false", dest="recursive")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--max-file-mb", type=float, default=100.0)
    parser.add_argument("--max-field-chars", type=int, default=20_000)
    parser.add_argument("--format", default="auto")
    parser.add_argument("--title-field")
    parser.add_argument("--text-field")
    parser.add_argument("--question-field")
    parser.add_argument("--answer-field")
    parser.add_argument("--instruction-field")
    parser.add_argument("--input-field")
    parser.add_argument("--output-field")
    parser.add_argument("--include-suffix", default=".jsonl,.json,.csv,.tsv,.txt,.md,.markdown,.html,.htm")
    parser.add_argument("--task-type", default="auto")
    args = parser.parse_args()
    started = perf_counter()
    include = _suffixes(args.include_suffix)
    files = [item for item in DatasetBridge().scan(args.source, args.recursive) if item.suffix in include and item.ingestible and item.size_bytes <= int(args.max_file_mb * 1024 * 1024)]
    files = files[: args.max_files]
    warnings: list[str] = []
    schemas = []
    articles = []
    rows_seen = 0
    for item in files:
        if item.suffix in {".txt", ".md", ".markdown", ".html", ".htm"}:
            reader = LocalFolderSourceReader(item.path.parent, recursive=False, max_files=1, max_articles=args.max_rows, max_file_mb=args.max_file_mb, include_suffixes={item.suffix})
            local_articles = [article for article in reader.iter_articles() if article.path and article.path.resolve() == item.path]
            articles.extend(local_articles)
            rows_seen += len(local_articles)
            warnings.extend(reader.warnings)
            continue
        schema = detect_schema(item.path, max_file_mb=args.max_file_mb)
        schemas.append(schema.to_dict())
        config = DatasetReaderConfig(
            max_rows=max(0, args.max_rows - rows_seen),
            max_field_chars=args.max_field_chars,
            title_field=args.title_field,
            text_field=args.text_field,
            question_field=args.question_field,
            answer_field=args.answer_field,
            instruction_field=args.instruction_field,
            input_field=args.input_field,
            output_field=args.output_field,
            task_type=None if args.task_type == "auto" else DatasetTaskType(args.task_type.upper()),
        )
        try:
            reader = reader_for_path(item.path, config, schema)
        except ValueError as exc:
            warnings.append(str(exc))
            continue
        rows = list(reader.iter_rows())
        rows_seen += len(rows)
        warnings.extend(reader.warnings)
        articles.extend(dataset_row_to_article(row) for row in rows)
        if rows_seen >= args.max_rows:
            warnings.append(f"stopped at max rows cap: {args.max_rows}")
            break
    stats = None
    if not args.dry_run and articles:
        stats = ingest_articles(articles, args.out)
        warnings.extend(stats.get("warnings", []))
    elif not articles:
        warnings.append("no usable dataset rows found; check field mappings or add local dataset files")
    payload = {
        "source": str(args.source),
        "out": str(args.out),
        "profile": args.profile,
        "dry_run": args.dry_run,
        "summary": summarize_dataset_files(files),
        "schemas": schemas,
        "selected_files": [item.to_dict() for item in files],
        "rows_seen": rows_seen,
        "articles": len(articles) if args.dry_run else (stats or {}).get("articles", 0),
        "concepts": 0 if args.dry_run else (stats or {}).get("concepts", 0),
        "warnings": sorted(set(warnings)),
        "elapsed_ms": round((perf_counter() - started) * 1000, 3),
    }
    return _print(payload, args.json_output)


def _suffixes(raw: str) -> set[str]:
    return {item.strip().lower() if item.strip().startswith(".") else f".{item.strip().lower()}" for item in raw.split(",") if item.strip()}


def _print(payload: dict, json_output: bool) -> int:
    if json_output:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0
    print("KRM-Core dataset ingest")
    print(f"source: {payload['source']}")
    print(f"out: {payload['out']}")
    print(f"profile: {payload['profile']}")
    print(f"dry run: {payload['dry_run']}")
    print(f"selected files: {len(payload['selected_files'])}")
    print(f"rows seen: {payload['rows_seen']}")
    print(f"articles: {payload['articles']}")
    print(f"concepts: {payload['concepts']}")
    for warning in payload["warnings"]:
        print(f"warning: {warning}")
    print(f"elapsed_ms: {payload['elapsed_ms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
