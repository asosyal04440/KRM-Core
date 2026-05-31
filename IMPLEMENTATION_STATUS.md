# KRM-Core V0.5 Implementation Status

Date: 2026-05-24

## 1. What Works

- V0 vertical slice still runs end to end.
- V0.1 intelligence quality scripts still pass demo rubric checks.
- V0.2 local data bridge remains intact.
- V0.3 ZIM bridge still detects local `.zim` files and reports missing optional backend clearly.
- V0.4 dataset bridge still ingests local JSONL/JSON/CSV/TSV/text fixtures and builds `data\mind_dataset`.
- V0.5 adds a standard KRM training record JSONL format with validation and deterministic record ids.
- V0.5 adds KRM special structure tokens such as `<|SOURCE|>`, `<|QUERY|>`, `<|ANSWER_PLAN|>`, `<|UNCERTAIN|>`, and `<|DO_NOT_CLAIM|>`.
- V0.5 adds a corpus builder that reads KRM artifacts and V0.4 training examples, deduplicates records, applies curriculum ordering, and writes train/valid/test JSONL plus a manifest.
- V0.5 adds curriculum profiles: `tiny`, `small`, `router_only`, `planner_only`, and `composer_only`.
- V0.5 adds a stdlib byte tokenizer that preserves KRM special tokens as single ids and round-trips English/Turkish text.
- V0.5 adds tiny model configs for `KRM_NATIVE_10M`, `KRM_NATIVE_30M`, and config-only `KRM_NATIVE_100M`.
- V0.5 adds rough training resource estimation for parameters, model memory, optimizer memory, activations, and checkpoints.
- V0.5 adds optional PyTorch backend code that imports torch only inside guarded functions.
- V0.5 adds `train_tiny_core.py --dry-run`, which works without PyTorch.
- V0.5 adds evaluation and export helpers that fail gracefully when model artifacts are missing.
- V0.5 adds offline-first Colab and Kaggle markdown notebook templates.
- No downloads, network code, API tokens, cloud APIs, large-scale training, GPU requirement, or heavy core dependencies were added.

## 2. What Fails

- No current V0.5 test failure remains.
- Actual tiny training was not run because PyTorch is not installed in the current core environment. This is expected; `train_tiny_core.py --dry-run` explains how to install `.[train]` manually.
- `eval_tiny_core.py` returns a clear non-zero graceful error when no trained model directory exists.

## 3. Commands Tested

```powershell
cd D:\yeni_ai_hiyerarsisi\krm_core
.venv\Scripts\python -m pytest
.venv\Scripts\python scripts\build_training_corpus.py --mind data\mind_dataset --examples data\training_examples --out data\training_corpus --profile tiny --dry-run
.venv\Scripts\python scripts\build_training_corpus.py --mind data\mind_dataset --examples data\training_examples --out data\training_corpus --profile tiny
.venv\Scripts\python scripts\train_tokenizer.py --corpus data\training_corpus --out data\tokenizers\krm_byte_tokenizer
.venv\Scripts\python scripts\estimate_training_resources.py --config 10m --batch-size 4 --seq-len 512
.venv\Scripts\python scripts\train_tiny_core.py --corpus data\training_corpus --tokenizer data\tokenizers\krm_byte_tokenizer --out data\models\krm_native_10m --config 10m --max-steps 100 --dry-run
.venv\Scripts\python scripts\eval_tiny_core.py --model data\models\missing_model --tokenizer data\tokenizers\krm_byte_tokenizer --corpus data\training_corpus
```

## 4. Test Results

- `pytest`: 53 passed.
- Training corpus dry-run: 320 records, train/valid/test split 256/32/32, no artifacts written.
- Training corpus real build: 320 records, `train.jsonl`, `valid.jsonl`, `test.jsonl`, and `manifest.json` created.
- Task distribution: `ANSWER_PLANNING` 40, `INTENT_ROUTING` 40, `DOMAIN_CLASSIFICATION` 80, `DO_NOT_CLAIM` 40, `CONCEPT_EXTRACTION` 40, `RETRIEVAL_SCORING` 80.
- Tokenizer build: vocab size 274, KRM special tokens preserved, 320 records scanned.
- Resource estimate for 10M config at batch 4 / seq 512: 8,477,184 parameters, about 115 MB rough total training memory.
- Training dry-run: works without torch, reports optional PyTorch install guidance.
- Eval missing-model path: graceful clear error, no stack trace.

## 5. Corpus Builder Status

- Reads `data\training_examples\examples.jsonl` if present.
- Reads compact concept cards from `mind.skel\concepts.jsonl` if `--mind` is provided.
- Produces deterministic `TrainingRecord` rows.
- Deduplicates by deterministic record id.
- Writes `train.jsonl`, `valid.jsonl`, `test.jsonl`, and `manifest.json`.
- Does not duplicate full raw source documents.

## 6. Tokenizer Status

- `CharByteTokenizer` / `ByteLevelTokenizer` is stdlib-only.
- KRM special tokens encode as single ids.
- Normal UTF-8 text round-trips, including Turkish characters.
- Tokenizer saves and loads from `tokenizer.json`.

## 7. Optional Torch Backend Status

- Core project imports without torch.
- Missing torch reports: `Optional PyTorch backend is not installed. Install manually with pip install -e .[train] if you want tiny smoke training.`
- If torch is installed manually, tests can run a tiny forward pass and `train_tiny_core.py` can execute a tiny local smoke loop.
- No distributed training, automatic mixed precision, downloader, or GPU requirement is included.

## 8. Notebook Template Status

- `notebooks\KRM_From_Scratch_Training_Colab_Template.md` exists.
- `notebooks\KRM_From_Scratch_Training_Kaggle_Template.md` exists.
- Both templates avoid credentials, dataset downloads, Kaggle APIs, Hugging Face APIs, and remote fetches.

## 9. Known Limitations

- V0.5 is a training forge, not a trained model release.
- Actual tiny training was not smoke-tested in this environment because optional torch is absent.
- The tokenizer is byte-level and intentionally simple; it is not BPE.
- `KRM_NATIVE_100M` is config-only and should not be trained by default.
- Eval is structure-oriented unless a real checkpoint exists.
- Export can package metadata and checkpoints when model artifacts exist, but KRM-native inference integration remains future work.
- Training examples are generated from small local artifacts and are not enough for a useful model yet.

## 10. V0.5 Acceptance Criteria

V0.5 acceptance criteria are met:

- Tests pass.
- Existing V0/V0.1/V0.2/V0.3/V0.4 commands remain covered by the regression suite and prior smoke commands.
- `build_training_corpus.py --dry-run` works.
- `build_training_corpus.py` creates train/valid/test JSONL and manifest.
- `train_tokenizer.py` creates a tokenizer.
- Tokenizer preserves KRM special tokens.
- `estimate_training_resources.py` works.
- `train_tiny_core.py --dry-run` works without torch.
- Project imports and tests pass without torch installed.
- `eval_tiny_core.py` fails gracefully if no trained model exists.
- Export helper is tested with metadata artifacts.
- Notebook templates exist.
- No downloads occur.
- No network code was added.
- No large-scale training is performed by default.
- README was updated.
- `IMPLEMENTATION_STATUS.md` was updated.

## 11. Next Recommended Phase

Recommended next phase: V0.6 source-grounding and training-data hardening before any serious training.

Good next work:

- Add paragraph-level source spans into training records.
- Add stronger `DO_NOT_CLAIM` and unsupported-claim negative examples.
- Add held-out quality sets for router/domain/planner tasks.
- Add corpus manifest checksums and provenance.
- Manually install optional train extra only when ready for a tiny 10M smoke training run.
