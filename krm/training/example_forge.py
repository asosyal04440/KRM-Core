from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from krm.concepts.concept_store import ConceptStore
from krm.concepts.domain import domain_name
from krm.pipeline import load_source_pointers
from krm.reasoning.router import QueryRouter


DEFAULT_TYPES = {"router", "domain", "retrieval", "planner", "eval"}


def forge_examples(mind_dir: Path, out_dir: Path, max_examples: int = 1000, types: set[str] | None = None, dry_run: bool = False) -> dict[str, Any]:
    selected = types or DEFAULT_TYPES
    concepts = ConceptStore.load_jsonl(Path(mind_dir) / "mind.skel" / "concepts.jsonl").all()
    pointers = load_source_pointers(Path(mind_dir))
    examples: list[dict[str, Any]] = []
    router = QueryRouter()
    for card in concepts:
        if len(examples) >= max_examples:
            break
        source_refs = card.source_refs[:3]
        text = card.canonical_name
        intent = router.classify(text)
        if "router" in selected:
            examples.append(_record("router", text, {"intent": intent.intent_type.value, "domains": intent.needed_domains, "likely_shards": [domain_name(card.domain_id)]}, source_refs))
        if "domain" in selected:
            examples.append(_record("domain", text, {"domains": [domain_name(card.domain_id)]}, source_refs))
        if "retrieval" in selected:
            examples.append(_record("retrieval", text, {"relevant_concept_id": card.concept_id, "relevance": 1.0}, source_refs))
        if "planner" in selected:
            examples.append(_record("planner", text, {"sections": ["grounded explanation"], "key_concepts": [card.canonical_name]}, source_refs, 0.6))
        if "eval" in selected:
            examples.append(_record("eval", text, {"expected_concepts": [card.canonical_name], "expected_structure": ["grounded"]}, source_refs, 0.6))
    examples = examples[:max_examples]
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "examples.jsonl").open("w", encoding="utf-8") as fh:
            for item in examples:
                fh.write(json.dumps(item, ensure_ascii=True) + "\n")
    return {"mind": str(mind_dir), "out": str(out_dir), "dry_run": dry_run, "example_count": len(examples), "types": sorted(selected)}


def _record(example_type: str, input_text: str, target: dict[str, Any], source_refs: list[str], confidence: float = 0.75) -> dict[str, Any]:
    basis = f"{example_type}:{input_text}:{target}"
    import hashlib

    return {
        "example_id": hashlib.blake2b(basis.encode("utf-8"), digest_size=8).hexdigest(),
        "type": example_type,
        "input": input_text,
        "target": target,
        "source_refs": source_refs,
        "confidence": confidence,
        "notes": "deterministic lightweight KRM-Core V0.4 example",
    }
