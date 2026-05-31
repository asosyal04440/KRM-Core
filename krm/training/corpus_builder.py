from __future__ import annotations

import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from krm.concepts.concept_store import ConceptStore
from krm.concepts.domain import domain_name
from krm.training.curriculum import build_curriculum_mix, curriculum_statistics
from krm.training.record_format import TrainingRecord, TrainingTaskType, make_record
from krm.training.special_tokens import format_training_prompt


EXAMPLE_TYPE_TO_TASK: dict[str, TrainingTaskType] = {
    "router": TrainingTaskType.INTENT_ROUTING,
    "domain": TrainingTaskType.DOMAIN_CLASSIFICATION,
    "retrieval": TrainingTaskType.RETRIEVAL_SCORING,
    "planner": TrainingTaskType.ANSWER_PLANNING,
    "eval": TrainingTaskType.DO_NOT_CLAIM,
}


def build_training_corpus(
    mind_dir: Path | None,
    examples_dir: Path | None,
    out_dir: Path,
    profile: str = "tiny",
    tasks: set[str] | None = None,
    max_records: int | None = None,
    max_input_chars: int = 20000,
    max_target_chars: int = 20000,
    dry_run: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    records = collect_training_records(mind_dir, examples_dir, max_input_chars, max_target_chars)
    normalized_tasks = {task.upper() for task in tasks or set()}
    if normalized_tasks and "ALL" not in normalized_tasks:
        allowed = {TrainingTaskType(task) for task in normalized_tasks}
        records = [record for record in records if record.task in allowed]
    records = build_curriculum_mix(records, profile)
    records = _dedupe(records)
    rng = random.Random(seed)
    rng.shuffle(records)
    if max_records is not None:
        records = records[:max_records]
    splits = _split(records)
    manifest = _manifest(records, splits, mind_dir, examples_dir, profile, seed, max_input_chars, max_target_chars)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_records in splits.items():
            _write_jsonl(out_dir / f"{split_name}.jsonl", split_records)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return {"out": str(out_dir), "dry_run": dry_run, "record_count": len(records), "splits": {k: len(v) for k, v in splits.items()}, "manifest": manifest}


def collect_training_records(
    mind_dir: Path | None,
    examples_dir: Path | None,
    max_input_chars: int = 20000,
    max_target_chars: int = 20000,
) -> list[TrainingRecord]:
    records: list[TrainingRecord] = []
    if examples_dir is not None:
        records.extend(_records_from_examples(examples_dir, max_input_chars, max_target_chars))
    if mind_dir is not None:
        records.extend(_records_from_mind(mind_dir, max_input_chars, max_target_chars))
    return records


def _records_from_examples(examples_dir: Path, max_input_chars: int, max_target_chars: int) -> list[TrainingRecord]:
    path = examples_dir / "examples.jsonl"
    if not path.exists():
        return []
    records: list[TrainingRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            task = EXAMPLE_TYPE_TO_TASK.get(str(item.get("type")), TrainingTaskType.LM_CONTINUATION)
            target = json.dumps(item.get("target") or {}, ensure_ascii=True, sort_keys=True)
            metadata = {
                "source_refs": item.get("source_refs") or [],
                "domains": (item.get("target") or {}).get("domains", []),
                "intent": (item.get("target") or {}).get("intent", ""),
                "confidence": float(item.get("confidence", 0.75)),
                "created_by": "krm.training.example_forge",
                "notes": item.get("notes", ""),
            }
            records.append(make_record(task, format_input(str(item.get("input") or ""), task), target, metadata, max_input_chars, max_target_chars))
    return records


def _records_from_mind(mind_dir: Path, max_input_chars: int, max_target_chars: int) -> list[TrainingRecord]:
    concepts = ConceptStore.load_jsonl(mind_dir / "mind.skel" / "concepts.jsonl").all()
    records: list[TrainingRecord] = []
    for card in concepts:
        metadata = {
            "source_refs": card.source_refs[:3],
            "domains": [domain_name(card.domain_id)],
            "intent": "",
            "confidence": round(card.confidence / 255, 3),
            "created_by": "krm.training.corpus_builder",
            "notes": "derived from compact concept card",
        }
        records.append(
            make_record(
                TrainingTaskType.CONCEPT_EXTRACTION,
                format_input(card.canonical_name, TrainingTaskType.CONCEPT_EXTRACTION),
                json.dumps({"concept": card.canonical_name, "aliases": card.aliases[:4]}, ensure_ascii=True, sort_keys=True),
                metadata,
                max_input_chars,
                max_target_chars,
            )
        )
        records.append(
            make_record(
                TrainingTaskType.DOMAIN_CLASSIFICATION,
                format_input(card.canonical_name, TrainingTaskType.DOMAIN_CLASSIFICATION),
                json.dumps({"domains": [domain_name(card.domain_id)]}, ensure_ascii=True, sort_keys=True),
                metadata,
                max_input_chars,
                max_target_chars,
            )
        )
        if card.source_refs:
            records.append(
                make_record(
                    TrainingTaskType.RETRIEVAL_SCORING,
                    format_input(card.canonical_name, TrainingTaskType.RETRIEVAL_SCORING),
                    json.dumps({"relevant_concept_id": card.concept_id, "source_refs": card.source_refs[:3], "relevance": 1.0}, ensure_ascii=True, sort_keys=True),
                    metadata,
                    max_input_chars,
                    max_target_chars,
                )
            )
    return records


def format_input(text: str, task: TrainingTaskType) -> str:
    return f"task={task.value}\n{text.strip()}"


def _dedupe(records: list[TrainingRecord]) -> list[TrainingRecord]:
    seen: set[str] = set()
    unique: list[TrainingRecord] = []
    for record in records:
        if record.record_id in seen:
            continue
        seen.add(record.record_id)
        unique.append(record)
    return unique


def _split(records: list[TrainingRecord]) -> dict[str, list[TrainingRecord]]:
    if not records:
        return {"train": [], "valid": [], "test": []}
    train_end = max(1, int(len(records) * 0.8))
    valid_end = max(train_end, int(len(records) * 0.9))
    if len(records) >= 3:
        valid_end = max(train_end + 1, valid_end)
    return {"train": records[:train_end], "valid": records[train_end:valid_end], "test": records[valid_end:]}


def _write_jsonl(path: Path, records: list[TrainingRecord]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.to_jsonl_line() + "\n")


def _manifest(
    records: list[TrainingRecord],
    splits: dict[str, list[TrainingRecord]],
    mind_dir: Path | None,
    examples_dir: Path | None,
    profile: str,
    seed: int,
    max_input_chars: int,
    max_target_chars: int,
) -> dict[str, Any]:
    task_counts = Counter(record.task.value for record in records)
    domain_counts = Counter(domain for record in records for domain in record.metadata.get("domains", []))
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "record_counts": {"total": len(records), **{name: len(items) for name, items in splits.items()}},
        "task_distribution": dict(task_counts),
        "domain_distribution": dict(domain_counts),
        "curriculum": curriculum_statistics(records),
        "source_artifact_paths": {"mind": str(mind_dir) if mind_dir else None, "examples": str(examples_dir) if examples_dir else None},
        "limits_used": {"max_input_chars": max_input_chars, "max_target_chars": max_target_chars},
        "profile": profile,
        "deterministic_seed": seed,
        "warnings": [] if records else ["no records were generated"],
    }


def read_corpus_records(corpus_dir: Path, split: str = "train") -> list[TrainingRecord]:
    path = corpus_dir / f"{split}.jsonl"
    if not path.exists():
        return []
    return [TrainingRecord.from_jsonl_line(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def formatted_corpus_text(records: list[TrainingRecord]) -> str:
    return "\n\n".join(format_training_prompt(record) for record in records)
