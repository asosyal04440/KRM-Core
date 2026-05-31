from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from krm.concepts.concept_types import ConceptType


class ConceptBudgetError(ValueError):
    pass


@dataclass(slots=True)
class ConceptCard:
    concept_id: int
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    concept_type: ConceptType = ConceptType.UNKNOWN
    domain_id: int = 0
    importance: int = 128
    confidence: int = 128
    source_refs: list[str] = field(default_factory=list)
    parent_ids: list[int] = field(default_factory=list)
    neighbor_hint_ids: list[int] = field(default_factory=list)
    flags: int = 0
    short_fingerprint: int = 0
    tiny_embedding_id: int | None = None

    def to_compact_dict(self) -> dict[str, Any]:
        return {
            "id": self.concept_id,
            "name": self.canonical_name,
            "alias": self.aliases[:8],
            "type": self.concept_type.value,
            "dom": self.domain_id,
            "imp": self.importance,
            "conf": self.confidence,
            "src": self.source_refs[:16],
            "par": self.parent_ids[:16],
            "nbr": self.neighbor_hint_ids[:32],
            "flg": self.flags,
            "fp": self.short_fingerprint,
            "emb": self.tiny_embedding_id,
        }

    @classmethod
    def from_compact_dict(cls, data: dict[str, Any]) -> "ConceptCard":
        return cls(
            concept_id=int(data["id"]),
            canonical_name=str(data["name"]),
            aliases=list(data.get("alias") or []),
            concept_type=ConceptType(data.get("type", ConceptType.UNKNOWN.value)),
            domain_id=int(data.get("dom", 0)),
            importance=int(data.get("imp", 128)),
            confidence=int(data.get("conf", 128)),
            source_refs=list(data.get("src") or []),
            parent_ids=[int(x) for x in data.get("par") or []],
            neighbor_hint_ids=[int(x) for x in data.get("nbr") or []],
            flags=int(data.get("flg", 0)),
            short_fingerprint=int(data.get("fp", 0)),
            tiny_embedding_id=data.get("emb"),
        )

    def estimate_size_bytes(self) -> int:
        string_bytes = len(self.canonical_name.encode("utf-8"))
        string_bytes += sum(len(a.encode("utf-8")) for a in self.aliases)
        pointer_bytes = sum(len(s.encode("utf-8")) for s in self.source_refs)
        int_bytes = 72 + 8 * (len(self.parent_ids) + len(self.neighbor_hint_ids))
        return int_bytes + string_bytes + pointer_bytes

    def validate_budget(self, max_bytes: int = 2048) -> None:
        if not 0 <= self.importance <= 255:
            raise ConceptBudgetError("importance must fit in uint8 range")
        if not 0 <= self.confidence <= 255:
            raise ConceptBudgetError("confidence must fit in uint8 range")
        if self.estimate_size_bytes() > max_bytes:
            raise ConceptBudgetError(f"concept card exceeds {max_bytes} bytes")
