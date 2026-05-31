# KRM From-Scratch Training - Kaggle Template

This template assumes you upload KRM-Core and local artifacts manually. It does not use Kaggle API credentials, dataset downloaders, Hugging Face APIs, or remote fetches.

## 1. Add Files Manually

Attach/upload:

- KRM-Core repository
- `data/training_corpus`
- `data/tokenizers` if already built

## 2. Optional Training Install

```bash
pip install -e .[train]
```

## 3. Corpus And Tokenizer

```bash
python scripts/build_training_corpus.py --mind data/mind_dataset --examples data/training_examples --out data/training_corpus --profile tiny --dry-run
python scripts/train_tokenizer.py --corpus data/training_corpus --out data/tokenizers/krm_byte_tokenizer
```

## 4. Safe Training Plan

```bash
python scripts/estimate_training_resources.py --config 10m --batch-size 2 --seq-len 128
python scripts/train_tiny_core.py --corpus data/training_corpus --tokenizer data/tokenizers/krm_byte_tokenizer --out data/models/krm_native_10m --config 10m --max-steps 100 --dry-run
```

## 5. Optional Tiny Training

```bash
python scripts/train_tiny_core.py --corpus data/training_corpus --tokenizer data/tokenizers/krm_byte_tokenizer --out data/models/krm_native_10m --config 10m --max-steps 100 --batch-size 2 --seq-len 128
```

## 6. Export

```bash
python scripts/export_tiny_core.py --model data/models/krm_native_10m --out data/models/exports/krm_native_10m_export
```

Save checkpoints and exports manually before the session ends.
