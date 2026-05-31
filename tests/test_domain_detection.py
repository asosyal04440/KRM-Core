from krm.concepts.domain import DOMAIN_IDS, detect_domains


def test_domain_detection_for_core_domains() -> None:
    assert detect_domains("photosynthesis glucose chloroplast")[0].domain_id == DOMAIN_IDS["biology"]
    assert detect_domains("Industrial Revolution Britain coal trade")[0].domain_id == DOMAIN_IDS["history"]
    assert any(score.domain_id == DOMAIN_IDS["technology"] for score in detect_domains("steam engine machine transport"))
    assert any(score.domain_id == DOMAIN_IDS["economics"] for score in detect_domains("capital labor market production"))
