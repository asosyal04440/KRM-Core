from pathlib import Path

from krm.pipeline import build_shards, ingest_source, inspect_shard, run_query


def test_vertical_slice_runs_on_sample_docs(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    source = repo / "data" / "sample_docs"
    mind = tmp_path / "mind"
    ingest = ingest_source(source, mind)
    assert ingest["articles"] >= 3
    assert ingest["concepts"] >= 20
    built = build_shards(mind, "local_core")
    assert built["shards"]
    result = run_query(mind, "Why did the Industrial Revolution start in Britain?")
    assert result["intent"] == "CAUSE_EFFECT"
    assert result["ghost_edge_count"] > 0
    assert "Britain" in result["final_answer"]
    assert "coal" in result["final_answer"]
    inspected = inspect_shard(mind, result["selected_shards"][0])
    assert "manifest" in inspected
