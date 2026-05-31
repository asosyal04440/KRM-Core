from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.source.dataset_discovery import DatasetDiscovery
from krm.source.zim_backend import ZimBackendError, backend_from_name


def inspect_zims(source: Path, backend_name: str, max_articles: int, show_titles: bool) -> dict:
    started = perf_counter()
    backend = backend_from_name(backend_name)
    zims = [item for item in DatasetDiscovery().scan(source) if item.suffix == ".zim"]
    results = []
    for item in zims:
        record = {
            "path": str(item.path),
            "size_bytes": item.size_bytes,
            "size_mb": round(item.size_bytes / (1024 * 1024), 3),
            "backend": getattr(backend, "selected_backend_name", backend.name),
            "backend_available": backend.is_available(),
            "backend_status": backend.explain_availability(),
            "metadata": {},
            "metadata_status": "not attempted",
            "article_listing_status": "not attempted",
            "articles": [],
            "recommendation": "",
        }
        if not backend.is_available():
            record["metadata_status"] = "backend unavailable"
            record["article_listing_status"] = "backend unavailable"
            record["recommendation"] = backend.explain_availability()
            results.append(record)
            continue
        try:
            opened = backend.open(item.path)
            record["metadata"] = opened.metadata()
            record["metadata_status"] = "ok"
            if show_titles:
                record["articles"] = [article.to_dict() for article in opened.iter_articles(limit=max_articles)]
            record["article_listing_status"] = "ok" if show_titles else "not requested"
            record["recommendation"] = "Use ingest_zim_real.py with conservative caps before building shards."
        except ZimBackendError as exc:
            record["metadata_status"] = "failed"
            record["article_listing_status"] = "failed"
            record["recommendation"] = str(exc)
        except Exception as exc:  # normal CLI mode should stay friendly
            record["metadata_status"] = "failed"
            record["article_listing_status"] = "failed"
            record["recommendation"] = f"ZIM inspection failed: {exc}"
        results.append(record)
    return {
        "source": str(source),
        "zim_file_count": len(zims),
        "backend": getattr(backend, "selected_backend_name", backend.name),
        "backend_available": backend.is_available(),
        "backend_status": backend.explain_availability(),
        "elapsed_ms": round((perf_counter() - started) * 1000, 3),
        "files": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect local ZIM files without extracting or ingesting them.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--max-articles", type=int, default=20)
    parser.add_argument("--show-titles", action="store_true")
    parser.add_argument("--backend", default="auto")
    args = parser.parse_args()
    result = inspect_zims(args.source, args.backend, args.max_articles, args.show_titles)
    if args.json_output:
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0
    print(f"source: {args.source}")
    print(f"detected ZIM files: {result['zim_file_count']}")
    print(f"backend: {result['backend']}")
    print(f"backend available: {result['backend_available']}")
    print(f"backend status: {result['backend_status']}")
    if not result["files"]:
        print("recommendation: no .zim files found; place local .zim files under the source folder.")
    for item in result["files"]:
        print(f"- {item['path']} size_mb={item['size_mb']}")
        print(f"  backend status: {item['backend_status']}")
        print(f"  metadata status: {item['metadata_status']}")
        print(f"  article listing status: {item['article_listing_status']}")
        if item["articles"]:
            print("  article titles:")
            for article in item["articles"]:
                print(f"  - {article['title']}")
        print(f"  recommendation: {item['recommendation']}")
    print(f"elapsed_ms: {result['elapsed_ms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
