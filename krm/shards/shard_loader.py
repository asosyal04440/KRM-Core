from __future__ import annotations

from krm.reasoning.router import QueryIntent
from krm.runtime.memory_budget import MemoryBudget
from krm.shards.shard_manifest import ShardManifest


class ShardLoader:
    def select(self, manifests: list[ShardManifest], intent: QueryIntent, budget: MemoryBudget) -> list[ShardManifest]:
        if not manifests:
            return []
        wanted = set(intent.needed_domains)
        matching = [manifest for manifest in manifests if wanted.intersection(manifest.domain_ids)]
        candidates = matching or manifests
        ranked = sorted(
            candidates,
            key=lambda m: (0 if wanted.intersection(m.domain_ids) else 1, m.estimated_ram_bytes, m.shard_id),
        )
        max_count = budget.clamp_loaded_shards(len(ranked))
        return ranked[:max_count]
