from krm.pipeline import build_shards, ingest_source, run_query


def test_resonance_trace_contains_rounds_edges_and_pruning(tmp_path):
    source = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "sample_docs"
    mind = tmp_path / "mind"
    ingest_source(source, mind)
    build_shards(mind, "local_core")
    result = run_query(mind, "Why did the Industrial Revolution start in Britain?", max_concepts=5, max_edges=5, rounds=2)
    trace = result["resonance_trace"]
    assert len(trace["rounds"]) == 2
    assert trace["rounds"][0]["top_concepts"]
    assert trace["top_edges_used"]
    assert trace["pruning_decisions"]
