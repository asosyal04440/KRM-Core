from __future__ import annotations

import json
from pathlib import Path

from krm.concepts.concept_card import ConceptCard
from krm.shards.shard_manifest import ShardManifest


class ShardStore:
    def __init__(self, mind_dir: Path) -> None:
        self.root = Path(mind_dir) / "mind.shards"

    def load_manifests(self) -> list[ShardManifest]:
        path = self.root / "manifest.json"
        if not path.exists():
            return []
        return [ShardManifest.from_dict(item) for item in json.loads(path.read_text(encoding="utf-8"))]

    def load_concepts(self, shard_id: str) -> list[ConceptCard]:
        path = self.root / f"{shard_id}.jsonl"
        concepts: list[ConceptCard] = []
        if not path.exists():
            return concepts
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    concepts.append(ConceptCard.from_compact_dict(json.loads(line)))
        return concepts
