from krm.evals.benchmark import DEMO_QUERIES
from krm.pipeline import build_shards, ingest_source, run_query


def test_demo_queries_pass_rubric(tmp_path):
    source = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "sample_docs"
    mind = tmp_path / "mind"
    ingest_source(source, mind)
    build_shards(mind, "local_core")
    for query in DEMO_QUERIES:
        result = run_query(mind, query)
        assert result["quality_rubric"]["passed"], result["quality_rubric"]
