import json
import subprocess
import sys

from krm.training.corpus_builder import build_training_corpus
from krm.training.tokenizer import CharByteTokenizer


def _examples(path):
    path.mkdir(parents=True)
    (path / "examples.jsonl").write_text(
        json.dumps({"example_id": "1", "type": "router", "input": "query", "target": {"intent": "FACTUAL"}, "source_refs": [], "confidence": 0.9}) + "\n",
        encoding="utf-8",
    )


def test_training_cli_dry_run_and_tokenizer(tmp_path) -> None:
    examples = tmp_path / "examples"
    corpus = tmp_path / "corpus"
    tok = tmp_path / "tok"
    _examples(examples)
    build_training_corpus(None, examples, corpus)
    result = subprocess.run([sys.executable, "scripts/train_tokenizer.py", "--corpus", str(corpus), "--out", str(tok)], capture_output=True, text=True, check=True)
    assert "vocab_size" in result.stdout
    assert CharByteTokenizer.load(tok).vocab_size > 256
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_tiny_core.py",
            "--corpus",
            str(corpus),
            "--tokenizer",
            str(tok),
            "--out",
            str(tmp_path / "model"),
            "--config",
            "10m",
            "--max-steps",
            "1",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "KRM-Core tiny training plan" in result.stdout
    result = subprocess.run([sys.executable, "scripts/estimate_training_resources.py", "--config", "10m", "--batch-size", "1", "--seq-len", "64"], capture_output=True, text=True, check=True)
    assert "parameters:" in result.stdout
