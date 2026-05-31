from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.pipeline import ingest_articles
from krm.source.dataset_discovery import DatasetDiscovery
from krm.source.zim_backend import ZimBackendError, ZimBackendUnavailableError, backend_from_name
from krm.source.zim_reader import ZimSourceReader


def _namespaces(raw: list[str] | None) -> list[str] | None:
    if not raw:
        return None
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely ingest capped local ZIM articles when an optional backend is available.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--profile", default="tiny")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--max-articles", type=int, default=500)
    parser.add_argument("--max-article-chars", type=int, default=20_000)
    parser.add_argument("--max-file-mb", type=float, default=2048.0)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--include-title-contains", default=None)
    parser.add_argument("--exclude-title-contains", default=None)
    parser.add_argument("--include-namespace", action="append", default=None)
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--no-recursive", action="store_false", dest="recursive")
    args = parser.parse_args()

    started = perf_counter()
    backend = backend_from_name(args.backend)
    discovered = [item for item in DatasetDiscovery().scan(args.source, recursive=args.recursive) if item.suffix == ".zim"]
    warnings: list[str] = []
    selected = []
    for item in discovered:
        if item.size_bytes > int(args.max_file_mb * 1024 * 1024):
            warnings.append(f"{item.path}: file exceeds max file warning/cap")
            continue
        selected.append(item)
        if len(selected) >= args.max_files:
            break

    payload = {
        "source": str(args.source),
        "out": str(args.out),
        "profile": args.profile,
        "dry_run": args.dry_run,
        "backend": getattr(backend, "selected_backend_name", backend.name),
        "backend_available": backend.is_available(),
        "backend_status": backend.explain_availability(),
        "zim_files_found": len(discovered),
        "zim_files_selected": len(selected),
        "files": [{"path": str(item.path), "size_bytes": item.size_bytes, "size_mb": round(item.size_bytes / (1024 * 1024), 3)} for item in selected],
        "articles_attempted": 0,
        "articles_ingested": 0,
        "skipped_articles": 0,
        "skipped_reasons": [],
        "estimated_artifact_size_bytes": 0,
        "estimated_ram_impact_bytes": 0,
        "concept_count": 0,
        "warning_count": 0,
        "warnings": warnings,
        "elapsed_ms": 0.0,
    }

    if not selected:
        warnings.append("no .zim file found or selected")
    if not backend.is_available():
        warnings.append(backend.explain_availability())
        payload["warnings"] = sorted(set(warnings))
        payload["warning_count"] = len(payload["warnings"])
        payload["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        return _print(payload, args.json_output, exit_code=0)
    if args.dry_run:
        payload["articles_attempted"] = min(args.max_articles, args.max_articles * len(selected))
        payload["estimated_artifact_size_bytes"] = max(4096, sum(item.size_bytes for item in selected) // 64)
        payload["estimated_ram_impact_bytes"] = min(512 * 1024 * 1024, args.max_articles * args.max_article_chars * 2)
        payload["warnings"] = sorted(set(warnings))
        payload["warning_count"] = len(payload["warnings"])
        payload["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        return _print(payload, args.json_output, exit_code=0)

    articles = []
    for item in selected:
        reader = ZimSourceReader(
            item.path,
            max_articles=args.max_articles,
            max_article_chars=args.max_article_chars,
            include_namespaces=_namespaces(args.include_namespace),
            backend=backend,
            include_title_contains=args.include_title_contains,
            exclude_title_contains=args.exclude_title_contains,
        )
        try:
            file_articles = list(reader.iter_articles())
            payload["articles_attempted"] += args.max_articles
            articles.extend(file_articles)
            warnings.extend(reader.warnings)
        except (ZimBackendUnavailableError, ZimBackendError) as exc:
            warnings.append(str(exc))
        except Exception as exc:
            warnings.append(f"{item.path}: ZIM ingest failed: {exc}")
    if articles:
        stats = ingest_articles(articles, args.out)
        payload["articles_ingested"] = stats["articles"]
        payload["concept_count"] = stats["concepts"]
        warnings.extend(stats.get("warnings", []))
    else:
        warnings.append("No sufficient ZIM-derived concepts found. Try increasing --max-articles or choose a richer ZIM file.")
    payload["skipped_articles"] = max(0, payload["articles_attempted"] - payload["articles_ingested"])
    payload["skipped_reasons"] = sorted(set(warnings))
    payload["estimated_artifact_size_bytes"] = max(4096, sum(len(article.text) for article in articles) // 4) if articles else 0
    payload["estimated_ram_impact_bytes"] = min(512 * 1024 * 1024, max(0, sum(len(article.text) for article in articles) * 2))
    payload["warnings"] = sorted(set(warnings))
    payload["warning_count"] = len(payload["warnings"])
    payload["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
    return _print(payload, args.json_output, exit_code=0)


def _print(payload: dict, json_output: bool, exit_code: int) -> int:
    if json_output:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return exit_code
    print("KRM-Core real ZIM ingest")
    print(f"source: {payload['source']}")
    print(f"out: {payload['out']}")
    print(f"profile: {payload['profile']}")
    print(f"dry run: {payload['dry_run']}")
    print(f"backend: {payload['backend']}")
    print(f"backend available: {payload['backend_available']}")
    print(f"backend status: {payload['backend_status']}")
    print(f"zim files found: {payload['zim_files_found']}")
    print(f"zim files selected: {payload['zim_files_selected']}")
    print(f"articles attempted: {payload['articles_attempted']}")
    print(f"articles ingested: {payload['articles_ingested']}")
    print(f"skipped articles: {payload['skipped_articles']}")
    print(f"estimated artifact size: {payload['estimated_artifact_size_bytes']} bytes")
    print(f"estimated RAM impact: {payload['estimated_ram_impact_bytes']} bytes")
    print(f"concept count: {payload['concept_count']}")
    print(f"warning count: {payload['warning_count']}")
    for warning in payload["warnings"]:
        print(f"warning: {warning}")
    print(f"elapsed_ms: {payload['elapsed_ms']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
