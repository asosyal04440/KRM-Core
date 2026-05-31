from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from krm.concepts.concept_card import ConceptCard
from krm.concepts.domain import domain_name
from krm.runtime.memory_budget import MemoryBudget
from krm.shards.shard_manifest import ShardManifest


class ShardBuilder:
    def build(self, concepts: list[ConceptCard], out_dir: Path, budget: MemoryBudget) -> list[ShardManifest]:
        shard_root = out_dir / "mind.shards"
        shard_root.mkdir(parents=True, exist_ok=True)
        by_domain: dict[int, list[ConceptCard]] = {}
        for card in concepts:
            by_domain.setdefault(card.domain_id, []).append(card)
        manifests: list[ShardManifest] = []
        for domain_id, cards in sorted(by_domain.items()):
            shard_id = domain_name(domain_id)
            manifests.append(self._write_shard(shard_root, shard_id, [domain_id], cards, budget))
        if concepts:
            manifests.insert(0, self._write_shard(shard_root, "general", [], concepts, budget))
        manifest_path = shard_root / "manifest.json"
        manifest_path.write_text(
            json.dumps([m.to_dict() for m in manifests], indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return manifests

    def _write_shard(
        self,
        shard_root: Path,
        shard_id: str,
        domain_ids: list[int],
        cards: list[ConceptCard],
        budget: MemoryBudget,
    ) -> ShardManifest:
        path = shard_root / f"{shard_id}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for card in cards:
                fh.write(json.dumps(card.to_compact_dict(), ensure_ascii=True) + "\n")
        disk_size = path.stat().st_size
        sources = {src for card in cards for src in card.source_refs}
        return ShardManifest(
            shard_id=shard_id,
            name=shard_id.replace("_", " ").title(),
            domain_ids=domain_ids,
            concept_count=len(cards),
            source_count=len(sources),
            disk_size_bytes=disk_size,
            estimated_ram_bytes=budget.estimate_shard_memory(len(cards)),
            index_types=["lexical", "fingerprint"],
            created_at=datetime.now(timezone.utc).isoformat(),
            version="0.1",
        )
