from krm.concepts.domain import DOMAIN_IDS
from krm.concepts.extractor import ConceptExtractor
from krm.source.source_pointer import SourceArticle


def test_extractor_gets_title_repeated_terms_and_filters_garbage() -> None:
    article = SourceArticle(
        source_id="t",
        source_type="plain_text",
        article_id="industrial",
        title="Industrial Revolution in Britain",
        text="The Industrial Revolution in Britain used coal. Coal enabled the steam engine and textile industry.",
    )
    result = ConceptExtractor().extract(article)
    names = {card.canonical_name for card in result.concepts}
    assert "Industrial Revolution in Britain" in names
    assert "coal" in names
    assert "steam engine" in names
    assert "The" not in names
    assert result.concepts[0].domain_id == DOMAIN_IDS["history"]
