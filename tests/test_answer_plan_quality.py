from krm.pipeline import build_shards, ingest_source, run_query


def test_answer_plan_quality_shapes(tmp_path):
    source = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "sample_docs"
    mind = tmp_path / "mind"
    ingest_source(source, mind)
    build_shards(mind, "local_core")
    industrial = run_query(mind, "Why did the Industrial Revolution start in Britain?")["answer_plan"]
    assert "reinforcing factors" in industrial["thesis"]
    assert any("coal" in section["concepts"] for section in industrial["sections"])
    biology = run_query(mind, "Compare photosynthesis and cellular respiration.")["answer_plan"]
    assert biology["intent"] == "COMPARISON"
    assert biology["comparisons"] == ["inputs", "outputs", "cell location", "energy direction"]
    ottoman = run_query(mind, "What might have changed if the Ottoman Empire had adopted the printing press widely much earlier?")["answer_plan"]
    assert ottoman["speculation_label_required"] is True
    assert len(ottoman["sections"]) >= 3
