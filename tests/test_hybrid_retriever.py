from krm.concepts.concept_card import ConceptCard
from krm.index.fingerprint_index import fingerprint_text
from krm.index.hybrid_retriever import HybridRetriever
from krm.runtime.memory_budget import MemoryBudget


def test_hybrid_retriever_lexical_fingerprint_and_explanations() -> None:
    concepts = [
        ConceptCard(1, "steam engine", source_refs=["s1"], short_fingerprint=fingerprint_text("steam engine coal")),
        ConceptCard(2, "chloroplast", source_refs=["s2"], short_fingerprint=fingerprint_text("chloroplast plant light")),
    ]
    result = HybridRetriever(concepts).search("coal steam engine", MemoryBudget.for_profile("local_core"), limit=10)
    assert result.hits[0].concept.canonical_name == "steam engine"
    assert result.hits[0].reasons
    assert any("match" in reason for reason in result.hits[0].reasons)
