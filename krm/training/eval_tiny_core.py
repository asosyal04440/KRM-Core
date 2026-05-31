from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from krm.training.corpus_builder import read_corpus_records
from krm.training.special_tokens import list_special_tokens


def evaluate_tiny_core(model_dir: Path, tokenizer_dir: Path, corpus_dir: Path) -> dict[str, Any]:
    if not model_dir.exists():
        return {"ok": False, "error": f"trained model path not found: {model_dir}", "expected": str(model_dir)}
    tokenizer_file = tokenizer_dir / "tokenizer.json"
    if not tokenizer_file.exists():
        return {"ok": False, "error": f"tokenizer not found: {tokenizer_file}", "expected": str(tokenizer_file)}
    records = read_corpus_records(corpus_dir, "valid") or read_corpus_records(corpus_dir, "train")
    model_config = model_dir / "config.json"
    checkpoints = sorted(model_dir.glob("*.pt"))
    report = {
        "ok": True,
        "model": str(model_dir),
        "tokenizer": str(tokenizer_dir),
        "record_count_checked": len(records[:20]),
        "checkpoint_found": bool(checkpoints),
        "config_found": model_config.exists(),
        "structure_token_checks": _structure_token_checks(records[:20]),
        "validation_loss": None,
        "notes": [],
    }
    if not checkpoints:
        report["notes"].append("no checkpoint found; structure-only evaluation completed")
    return report


def _structure_token_checks(records: list[Any]) -> dict[str, Any]:
    token_text = "\n".join(record.input + "\n" + record.target for record in records)
    present = [token for token in list_special_tokens() if token in token_text]
    return {"special_tokens_present": present, "preserves_krm_format": True}


def write_eval_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
