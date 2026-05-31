import json

from krm.training.corpus_builder import build_training_corpus


def _examples(path):
    path.mkdir(parents=True)
    rows = [
        {"example_id": "1", "type": "router", "input": "Why coal?", "target": {"intent": "EXPLANATION", "domains": ["history"]}, "source_refs": ["s1"], "confidence": 0.8},
        {"example_id": "2", "type": "planner", "input": "hot graph", "target": {"sections": ["answer"]}, "source_refs": ["s2"], "confidence": 0.7},
    ]
    with (path / "examples.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_corpus_builder_splits_manifest_and_dry_run(tmp_path) -> None:
    examples = tmp_path / "examples"
    out = tmp_path / "corpus"
    _examples(examples)
    dry = build_training_corpus(None, examples, out, dry_run=True)
    assert dry["record_count"] == 2
    assert not out.exists()
    real = build_training_corpus(None, examples, out, dry_run=False)
    assert real["splits"]["train"] >= 1
    assert (out / "train.jsonl").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task_distribution"]["INTENT_ROUTING"] == 1
