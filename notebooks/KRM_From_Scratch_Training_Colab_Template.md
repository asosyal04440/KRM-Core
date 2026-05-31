# KRM From-Scratch Training - Colab Template

This template is intentionally offline-first. It does not download datasets, call Hugging Face, call Kaggle, or use API tokens.

## 1. Upload Local Artifacts Manually

Upload the repository and any already-built local folders:

- `data/training_corpus`
- `data/tokenizers`
- optionally `data/models`

## 2. Install Optional Training Dependencies

```bash
pip install -e .[train]
```

## 3. Build Or Validate Corpus

```bash
python scripts/build_training_corpus.py --mind data/mind_dataset --examples data/training_examples --out data/training_corpus --profile tiny --dry-run
python scripts/build_training_corpus.py --mind data/mind_dataset --examples data/training_examples --out data/training_corpus --profile tiny
```

## 4. Train Tokenizer

```bash
python scripts/train_tokenizer.py --corpus data/training_corpus --out data/tokenizers/krm_byte_tokenizer
```

## 5. Dry Run Tiny Training

```bash
python scripts/train_tiny_core.py --corpus data/training_corpus --tokenizer data/tokenizers/krm_byte_tokenizer --out data/models/krm_native_10m --config 10m --max-steps 100 --dry-run
```

## 6. Optional Tiny Smoke Training

```bash
python scripts/train_tiny_core.py --corpus data/training_corpus --tokenizer data/tokenizers/krm_byte_tokenizer --out data/models/krm_native_10m --config 10m --max-steps 100 --batch-size 2 --seq-len 128
```

## 7. Evaluate And Export

```bash
python scripts/eval_tiny_core.py --model data/models/krm_native_10m --tokenizer data/tokenizers/krm_byte_tokenizer --corpus data/training_corpus
python scripts/export_tiny_core.py --model data/models/krm_native_10m --out data/models/exports/krm_native_10m_export
```

Download the exported artifacts manually from the notebook environment.
