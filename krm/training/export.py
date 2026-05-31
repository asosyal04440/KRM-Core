from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def export_tiny_core(model_dir: Path, out_dir: Path) -> dict[str, Any]:
    if not model_dir.exists():
        return {"ok": False, "error": f"model path not found: {model_dir}"}
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ["config.json", "run_metadata.json", "eval_report.json"]:
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            copied.append(name)
    for checkpoint in sorted(model_dir.glob("*.pt"))[:1]:
        shutil.copy2(checkpoint, out_dir / checkpoint.name)
        copied.append(checkpoint.name)
    tokenizer_src = model_dir / "tokenizer.json"
    if tokenizer_src.exists():
        shutil.copy2(tokenizer_src, out_dir / "tokenizer.json")
        copied.append("tokenizer.json")
    readme = out_dir / "README_MODEL.md"
    readme.write_text(
        "# KRM Native Tiny Export\n\n"
        "This is a local experimental KRM-native model artifact export.\n\n"
        "License: unknown/local experiment unless the user adds explicit licensing notes.\n\n"
        "Inference integration is experimental and may require the optional training backend.\n",
        encoding="utf-8",
    )
    manifest = {"created_at": datetime.now(UTC).isoformat(), "source_model": str(model_dir), "copied_files": copied, "license": "unknown/local experiment"}
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return {"ok": True, "out": str(out_dir), "copied_files": copied}
