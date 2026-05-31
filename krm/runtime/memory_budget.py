from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceededError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    name: str
    max_ram_bytes: int
    max_vram_bytes: int
    max_disk_bytes: int
    default_candidates: int
    default_ghost_edges: int
    default_rounds: int
    source_snippet_chars: int


PROFILES = {
    "ultra_lite": ResourceProfile("ultra_lite", 4 * 1024**3, 4 * 1024**3, 32 * 1024**3, 120, 500, 3, 320),
    "local_core": ResourceProfile("local_core", 8 * 1024**3, 5 * 1024**3, 128 * 1024**3, 300, 2000, 4, 500),
    "colossus_lite": ResourceProfile("colossus_lite", 12 * 1024**3, 6 * 1024**3, 512 * 1024**3, 1000, 10000, 5, 700),
}


@dataclass
class MemoryBudget:
    profile: ResourceProfile
    degradation_decisions: list[str] = field(default_factory=list)

    @classmethod
    def for_profile(cls, name: str) -> "MemoryBudget":
        return cls(PROFILES.get(name, PROFILES["local_core"]))

    def estimate_query_memory(self, candidate_count: int, edge_count: int, loaded_shards: int) -> int:
        return 8 * 1024 * 1024 + candidate_count * 2048 + edge_count * 192 + loaded_shards * 2 * 1024 * 1024

    def estimate_shard_memory(self, concept_count: int, index_count: int = 2) -> int:
        return concept_count * 2048 + index_count * 512 * 1024

    def estimate_cache_memory(self, entries: int, avg_entry_bytes: int = 512) -> int:
        return entries * avg_entry_bytes

    def clamp_rounds(self, requested: int) -> int:
        if self.profile.max_ram_bytes < 128 * 1024 * 1024:
            reduced = max(1, min(requested, 1))
        else:
            reduced = min(requested, self.profile.default_rounds)
        if reduced < requested:
            self.degradation_decisions.append(f"reduced resonance rounds from {requested} to {reduced}")
        return reduced

    def clamp_edges(self, requested: int) -> int:
        reduced = min(requested, self.profile.default_ghost_edges)
        if self.profile.max_ram_bytes < 128 * 1024 * 1024:
            reduced = min(reduced, 64)
        if reduced < requested:
            self.degradation_decisions.append(f"reduced max ghost edges from {requested} to {reduced}")
        return max(0, reduced)

    def clamp_candidates(self, requested: int) -> int:
        reduced = min(requested, self.profile.default_candidates)
        if self.profile.max_ram_bytes < 128 * 1024 * 1024:
            reduced = min(reduced, 16)
        if reduced < requested:
            self.degradation_decisions.append(f"reduced candidate concepts from {requested} to {reduced}")
        return max(1, reduced)

    def clamp_loaded_shards(self, requested: int) -> int:
        reduced = requested
        if self.profile.max_ram_bytes < 128 * 1024 * 1024:
            reduced = min(requested, 1)
        if reduced < requested:
            self.degradation_decisions.append(f"reduced loaded shards from {requested} to {reduced}")
        return max(1, reduced)

    def snippet_chars(self) -> int:
        chars = self.profile.source_snippet_chars
        if self.profile.max_ram_bytes < 128 * 1024 * 1024:
            chars = min(chars, 180)
            self.degradation_decisions.append("shortened source snippets")
        return chars

    def ensure_query_fits(self, candidate_count: int, edge_count: int, loaded_shards: int) -> int:
        estimated = self.estimate_query_memory(candidate_count, edge_count, loaded_shards)
        if estimated > self.profile.max_ram_bytes:
            raise BudgetExceededError(f"estimated query memory {estimated} exceeds {self.profile.max_ram_bytes}")
        return estimated


def estimate_query_memory(candidate_count: int, edge_count: int, loaded_shards: int) -> int:
    return MemoryBudget.for_profile("local_core").estimate_query_memory(candidate_count, edge_count, loaded_shards)


def estimate_shard_memory(concept_count: int, index_count: int = 2) -> int:
    return MemoryBudget.for_profile("local_core").estimate_shard_memory(concept_count, index_count)


def estimate_cache_memory(entries: int, avg_entry_bytes: int = 512) -> int:
    return MemoryBudget.for_profile("local_core").estimate_cache_memory(entries, avg_entry_bytes)
