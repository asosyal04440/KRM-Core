from krm.concepts.concept_card import ConceptCard
from krm.concepts.concept_types import ConceptType
from krm.index.fingerprint_index import fingerprint_text


def test_concept_card_serialization_size_and_budget() -> None:
    card = ConceptCard(
        concept_id=7,
        canonical_name="steam engine",
        aliases=["engine"],
        concept_type=ConceptType.TOOL,
        domain_id=1,
        importance=200,
        confidence=190,
        source_refs=["sample::steam"],
        short_fingerprint=fingerprint_text("steam engine"),
    )
    restored = ConceptCard.from_compact_dict(card.to_compact_dict())
    assert restored.canonical_name == "steam engine"
    assert restored.concept_type == ConceptType.TOOL
    assert restored.estimate_size_bytes() > 0
    restored.validate_budget(max_bytes=2048)
