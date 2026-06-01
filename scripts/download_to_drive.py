"""
KRM-Core: Drive'a Dataset İndirme Scripti
=========================================
Çalıştırmak için: Colab'da bu dosyayı çalıştırın.
Drive'a Türkçe + Matematik + Bilgisayar Bilimi datasetleri indirir.

Kullanım:
  1. Colab'da bu hücreyi çalıştırın
  2. Drive'ınızı mount edin
  3. Hangi datasetleri istediğinizi seçin
  4. İndirme başlar, Drive'a kaydedilir

Depolama: ~2TB Google AI Pro üyeliği ile rahatlıkla sığar.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ============================================================
# DATASET TANIMLARI
# ============================================================

DATASETS = {
    # -- TÜRKÇE --
    "turkce_wikipedia": {
        "path": "Wikipedia",
        "config": "tr",
        "split": "train",
        "field": "text",
        "category": "turkce",
        "desc": "Türkçe Vikipedi (~200MB)",
        "max_tokens": 500_000_000,
        "priority": 1,
    },
    "turkce_oscar": {
        "path": "oscar-corpus/OSCAR-2301",
        "config": "tr",
        "split": "train",
        "field": "text",
        "category": "turkce",
        "desc": "OSCAR Türkçe (~2GB)",
        "max_tokens": 2_000_000_000,
        "priority": 2,
    },
    "turkce_cc100": {
        "path": "cc100",
        "config": "tr",
        "split": "train",
        "field": "text",
        "category": "turkce",
        "desc": "CC100 Türkçe (~5GB)",
        "max_tokens": 5_000_000_000,
        "priority": 3,
    },
    "turkce_culturaX": {
        "path": "HPLMG/culturaX",
        "config": "tr",
        "split": "train",
        "field": "text",
        "category": "turkce",
        "desc": "CulturaX Türkçe (~6.3B token, ~12GB)",
        "max_tokens": 6_300_000_000,
        "priority": 4,
    },
    "turkce_flores": {
        "path": "facebook/flores",
        "config": "swh_Latn",
        "split": "dev",
        "field": "text_swh_Latn",
        "category": "turkce",
        "desc": "Türkçe-İngilizce çift dil verisi (~10MB)",
        "max_tokens": 10_000_000,
        "priority": 5,
    },

    # -- MATEMATİK --
    "math_openwebmath": {
        "path": "allenai/openwebmath",
        "config": None,
        "split": "train",
        "field": "text",
        "category": "matematik",
        "desc": "OpenWebMath (~14.7B token, ~30GB)",
        "max_tokens": 14_700_000_000,
        "priority": 1,
    },
    "math_gsm8k": {
        "path": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "field": "question",
        "category": "matematik",
        "desc": "GSM8K Matematik (~8K soru, ~10MB)",
        "max_tokens": 10_000_000,
        "priority": 2,
    },
    "math_metamathqa": {
        "path": "meta-math/MetaMathQA",
        "config": None,
        "split": "train",
        "field": "query",
        "category": "matematik",
        "desc": "MetaMathQA (~395K soru, ~200MB)",
        "max_tokens": 200_000_000,
        "priority": 3,
    },
    "math_numina": {
        "path": "AI-MO/NuminaMath-QA",
        "config": None,
        "split": "train",
        "field": "problem",
        "category": "matematik",
        "desc": "NuminaMath-QA (~860K soru, ~400MB)",
        "max_tokens": 400_000_000,
        "priority": 4,
    },
    "math_mmlu": {
        "path": "cais/mmlu",
        "config": "abstract_algebra",
        "split": "validation",
        "field": "question",
        "category": "matematik",
        "desc": "MMLU Matematik (~14K soru, ~20MB)",
        "max_tokens": 20_000_000,
        "priority": 5,
    },

    # -- BİLGİSAYAR BİLİMİ / KOD --
    "cs_the_stack_python": {
        "path": "bigcode/the-stack",
        "config": "Python",
        "split": "train",
        "field": "content",
        "category": "bilgisayar_bilimi",
        "desc": "The Stack Python (~15GB)",
        "max_tokens": 15_000_000_000,
        "priority": 1,
    },
    "cs_github_code": {
        "path": "codeparrot/github-code",
        "config": None,
        "split": "train",
        "field": "code",
        "category": "bilgisayar_bilimi",
        "desc": "GitHub Code (~20GB)",
        "max_tokens": 20_000_000_000,
        "priority": 2,
    },
    "cs_pileoflaw": {
        "path": "pile-of-law/pile-of-law",
        "config": None,
        "split": "train",
        "field": "text",
        "category": "bilgisayar_bilimi",
        "desc": "Pile of Law (Hukuk/Bilim, ~2GB)",
        "max_tokens": 2_000_000_000,
        "priority": 3,
    },
    "cs_arxiv": {
        "path": "CulturaX/arxiv",
        "config": None,
        "split": "train",
        "field": "text",
        "category": "bilgisayar_bilimi",
        "desc": "ArXiv Makaleleri (~5GB)",
        "max_tokens": 5_000_000_000,
        "priority": 4,
    },

    # -- GENEL BİLGİ --
    "genel_fineweb_edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "field": "text",
        "category": "genel",
        "desc": "FineWeb-Edu (~1.3T token, ~30GB)",
        "max_tokens": 1_300_000_000_000,
        "priority": 1,
    },
    "genel_c4": {
        "path": "allenai/c4",
        "config": "en",
        "split": "train",
        "field": "text",
        "category": "genel",
        "desc": "C4 İngilizce (~180GB)",
        "max_tokens": 180_000_000_000,
        "priority": 2,
    },
}


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def download_dataset(
    ds_name: str,
    ds_info: dict,
    output_dir: Path,
    max_tokens: int | None = None,
    max_examples: int | None = None,
    streaming: bool = True,
) -> dict:
    """Tek bir dataseti Drive'a indir."""
    try:
        from datasets import load_dataset
    except ImportError:
        return {"status": "error", "message": "datasets kütüphanesi yüklü değil"}

    limit = max_tokens or ds_info["max_tokens"]
    if max_examples:
        limit = max_examples

    out_file = output_dir / f"{ds_name}.jsonl"
    if out_file.exists():
        existing = out_file.read_text(encoding="utf-8").count("\n")
        print(f"    Zaten mevcut: {out_file.name} ({existing} satır)")
        return {"status": "exists", "lines": existing, "file": str(out_file)}

    print(f"  İndiriliyor: {ds_info['desc']}")
    print(f"  Kaynak: {ds_info['path']}")
    t0 = time.time()

    try:
        ds = load_dataset(
            ds_info["path"],
            ds_info.get("config"),
            split=ds_info["split"],
            streaming=streaming,
            trust_remote_code=True,
        )
    except Exception as e:
        return {"status": "error", "message": str(e)}

    lines_written = 0
    tokens_written = 0
    buffer = []

    with open(out_file, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            if max_examples and lines_written >= max_examples:
                break

            text = example.get(ds_info["field"], "")
            if not text or len(text) < 10:
                continue

            tokens = estimate_tokens(text)
            if tokens_written + tokens > limit:
                remaining = limit - tokens_written
                text = text[: remaining * 4]
                tokens = estimate_tokens(text)

            record = json.dumps({
                "text": text,
                "source": ds_name,
                "tokens_est": tokens,
            }, ensure_ascii=False)
            f.write(record + "\n")
            lines_written += 1
            tokens_written += tokens

            if lines_written % 5000 == 0:
                elapsed = time.time() - t0
                print(f"    {lines_written:,} satır | {tokens_written:,} token | {elapsed:.0f}s")

    elapsed = time.time() - t0
    size_mb = out_file.stat().st_size / 1e6
    print(f"    Tamamlandı: {lines_written:,} satır, {tokens_written:,} token, {size_mb:.1f}MB, {elapsed:.0f}s")

    return {
        "status": "ok",
        "lines": lines_written,
        "tokens": tokens_written,
        "size_mb": size_mb,
        "elapsed_s": elapsed,
        "file": str(out_file),
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Drive'a dataset indir")
    parser.add_argument("--out", type=Path, required=True, help="Drive çıkış dizini")
    parser.add_argument("--categories", default="all", help="turkce,matematik,bilgisayar_bilimi,genel veya all")
    parser.add_argument("--max-tokens", type=int, default=None, help="Toplam token limiti")
    parser.add_argument("--max-examples", type=int, default=None, help="Max örnek sayısı (her dataset için)")
    parser.add_argument("--datasets", default="all", help="virgülle ayrılmış dataset isimleri veya all")
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", action="store_false", dest="streaming")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "download_manifest.json"

    # Hangi datasetleri indireceğimizi belirle
    if args.datasets != "all":
        selected = [d.strip() for d in args.datasets.split(",")]
    elif args.categories != "all":
        cats = {c.strip() for c in args.categories.split(",")}
        selected = [name for name, info in DATASETS.items() if info["category"] in cats]
    else:
        selected = list(DATASETS.keys())

    # Öncelik sırasına göre sırala
    selected.sort(key=lambda name: DATASETS[name]["priority"])

    print("=" * 60)
    print("KRM-Core Dataset İndirici")
    print("=" * 60)
    print(f"Çıkış: {args.out}")
    print(f"Dataset sayısı: {len(selected)}")
    print(f"Streaming: {args.streaming}")
    print()

    for name in selected:
        info = DATASETS[name]
        print(f"[{info['category']}] {name}: {info['desc']}")

    print()

    results = {}
    total_tokens = 0
    total_lines = 0

    for name in selected:
        info = DATASETS[name]
        print(f"\n{'-' * 50}")
        print(f"Dataset: {name}")
        print(f"{'-' * 50}")

        if args.dry_run:
            results[name] = {"status": "dry_run", "desc": info["desc"]}
            continue

        result = download_dataset(
            name, info, args.out,
            max_tokens=args.max_tokens,
            max_examples=args.max_examples,
            streaming=args.streaming,
        )
        results[name] = result
        if result["status"] in ("ok", "exists"):
            total_tokens += result.get("tokens", 0)
            total_lines += result.get("lines", 0)

    # Manifest kaydet
    manifest = {
        "total_tokens": total_tokens,
        "total_lines": total_lines,
        "total_datasets": len(selected),
        "datasets": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"TAMAMLANDI!")
    print(f"{'=' * 60}")
    print(f"Toplam satır: {total_lines:,}")
    print(f"Toplam token: {total_tokens:,} (~{total_tokens/1e9:.1f}B)")
    print(f"Manifest: {manifest_path}")
    print(f"Çıkış: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
