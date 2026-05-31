from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ShardManifest:
    shard_id: str
    name: str
    domain_ids: list[int]
    concept_count: int
    source_count: int
    disk_size_bytes: int
    estimated_ram_bytes: int
    index_types: list[str]
    created_at: str
    version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "name": self.name,
            "domain_ids": self.domain_ids,
            "concept_count": self.concept_count,
            "source_count": self.source_count,
            "disk_size_bytes": self.disk_size_bytes,
            "estimated_ram_bytes": self.estimated_ram_bytes,
            "index_types": self.index_types,
            "created_at": self.created_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShardManifest":
        return cls(
            shard_id=data["shard_id"],
            name=data["name"],
            domain_ids=[int(x) for x in data.get("domain_ids", [])],
            concept_count=int(data.get("concept_count", 0)),
            source_count=int(data.get("source_count", 0)),
            disk_size_bytes=int(data.get("disk_size_bytes", 0)),
            estimated_ram_bytes=int(data.get("estimated_ram_bytes", 0)),
            index_types=list(data.get("index_types", [])),
            created_at=data.get("created_at", ""),
            version=data.get("version", "0"),
        )
