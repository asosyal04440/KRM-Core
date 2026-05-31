from __future__ import annotations

import json
from pathlib import Path

from krm.concepts.concept_card import ConceptCard


class ConceptStore:
    def __init__(self) -> None:
        self._cards: dict[int, ConceptCard] = {}
        self._by_name: dict[str, int] = {}
        self._next_id = 1

    def add(self, card: ConceptCard) -> ConceptCard:
        key = card.canonical_name.lower()
        if key in self._by_name:
            existing = self._cards[self._by_name[key]]
            existing.aliases = sorted(set(existing.aliases + card.aliases))[:8]
            existing.source_refs = sorted(set(existing.source_refs + card.source_refs))[:16]
            existing.importance = max(existing.importance, card.importance)
            existing.confidence = max(existing.confidence, card.confidence)
            return existing
        card.concept_id = self._next_id
        self._next_id += 1
        self._cards[card.concept_id] = card
        self._by_name[key] = card.concept_id
        return card

    def all(self) -> list[ConceptCard]:
        return [self._cards[i] for i in sorted(self._cards)]

    def get(self, concept_id: int) -> ConceptCard | None:
        return self._cards.get(concept_id)

    def save_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for card in self.all():
                fh.write(json.dumps(card.to_compact_dict(), ensure_ascii=True) + "\n")

    @classmethod
    def load_jsonl(cls, path: Path) -> "ConceptStore":
        store = cls()
        if not path.exists():
            return store
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                card = ConceptCard.from_compact_dict(json.loads(line))
                store._cards[card.concept_id] = card
                store._by_name[card.canonical_name.lower()] = card.concept_id
                store._next_id = max(store._next_id, card.concept_id + 1)
        return store
