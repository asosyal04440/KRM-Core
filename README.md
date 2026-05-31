# KRM-Core

KRM-Core is an experimental local AI architecture that separates language generation from structured knowledge activation. It uses compact concept skeletons, shard-based retrieval, procedural ghost edges, and a concept resonance engine to build a small active thought graph for each query. A small local language model then turns this reasoning context into an answer.

V0 uses original sample text, rebuildable JSONL artifacts, and a deterministic mock model. It does not use cloud APIs, a GPU, a real LLM, or real Kiwix/ZIM parsing.

## Why This Is Not Normal RAG

Normal RAG often embeds many chunks and lets a language model do most of the reasoning. KRM-Core keeps raw source documents as the source of truth, stores compact concept cards, loads small shards, generates query-local ghost edges, and runs symbolic activation before the mock language layer writes the answer.

## Core Ideas

- Source pointers: artifacts point back to source files instead of duplicating full text.
- Folded shards: concepts are grouped into small rebuildable shard files.
- Tiny indexes: lexical and fingerprint indexes are used before any vector layer.
- Ghost edges: query-time relationships are generated only inside the active candidate set.
- Concept resonance: activation propagates over a small hot subgraph.
- Policy seeds: compact answer behavior, not hidden model weights.

## Windows Quickstart

```powershell
cd D:\yeni_ai_hiyerarsisi\krm_core
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
python scripts\ingest_zim.py --source data\sample_docs --out data\mind
python scripts\build_shards.py --mind data\mind --profile local_core
python scripts\run_query.py --mind data\mind --query "Why did the Industrial Revolution start in Britain?"
pytest
```

Other demo queries:

```powershell
python scripts\run_query.py --mind data\mind --query "Compare photosynthesis and cellular respiration."
python scripts\run_query.py --mind data\mind --query "What might have changed if the Ottoman Empire had adopted the printing press widely much earlier?"
python scripts\run_query.py --mind data\mind --query "Why did the Industrial Revolution start in Britain?" --json
python scripts\run_query.py --mind data\mind --query "Why did the Industrial Revolution start in Britain?" --trace
python scripts\benchmark_memory.py --mind data\mind
python scripts\inspect_shard.py --mind data\mind --shard general
python scripts\compare_baselines.py --mind data\mind
python scripts\evaluate_demo_quality.py --mind data\mind
```

## V0.2 Local Data Bridge

V0.2 lets KRM-Core inspect and ingest small local files from disk without downloading anything. Put local files in:

```powershell
D:\yeni_ai_hiyerarsisi\krm_core\veriler
```

Supported ingestible formats:

- `.txt`
- `.md`
- `.markdown`
- `.html`
- `.htm`
- `.jsonl`
- `.csv`

ZIM files are detected and reported, but not parsed yet. Real ZIM parsing is planned for a later phase and V0.2 does not add `pyzim`, `libzim`, downloads, or extraction code.

Safety defaults:

- `--max-file-mb 25`
- `--max-files 100`
- `--max-articles 10000`
- `--profile tiny`
- `--dry-run` available before writing artifacts

Local data commands:

```powershell
python scripts\inspect_data.py --source veriler
python scripts\inspect_data.py --source veriler --json
python scripts\ingest_local.py --source veriler --out data\mind_local --profile tiny --dry-run
python scripts\ingest_local.py --source veriler --out data\mind_local --profile tiny
python scripts\build_shards.py --mind data\mind_local --profile tiny
python scripts\run_query.py --mind data\mind_local --query "your query here" --trace
```

JSONL defaults:

- title field: `title`
- text field: `text`

CSV defaults:

- title field: `title`
- text field: `text`

Override fields when needed:

```powershell
python scripts\ingest_local.py --source veriler --out data\mind_local --jsonl-title-field headline --jsonl-text-field body
python scripts\ingest_local.py --source veriler --out data\mind_local --csv-title-field name --csv-text-field content
```

## V0.3 Real ZIM Reader Bridge

V0.3 adds an optional backend bridge for real local `.zim` files that the user places manually in:

```powershell
D:\yeni_ai_hiyerarsisi\krm_core\veriler
```

V0.3 does not download ZIM files, does not extract a full ZIM to disk, and does not load a full ZIM into RAM. It uses strict caps by default and still works when no optional ZIM backend is installed.

Inspect local ZIM files:

```powershell
python scripts\inspect_zim.py --source veriler
python scripts\inspect_zim.py --source veriler --json --show-titles --max-articles 20
```

Dry run before ingest:

```powershell
python scripts\ingest_zim_real.py --source veriler --out data\mind_zim --profile tiny --max-articles 500 --dry-run
```

Real ingest if a supported optional backend is available:

```powershell
python scripts\ingest_zim_real.py --source veriler --out data\mind_zim --profile tiny --max-articles 500
python scripts\build_shards.py --mind data\mind_zim --profile tiny
python scripts\run_query.py --mind data\mind_zim --query "your query here" --trace
```

If no backend is available, the command reports:

```text
ZIM file detected, but no real ZIM parsing backend is installed. V0.3 backend is optional. Install/enable a supported backend later.
```

Supported backend selection names:

- `auto`
- `stub`
- `libzim.reader`
- `pyzim`

No optional ZIM dependency is installed automatically. If a backend is enabled later, install it manually in the local environment and rerun `inspect_zim.py`.

## V0.4 Dataset Bridge + Training Example Forge

KRM-Core is not limited to ZIM. V0.4 adds a lightweight bridge for manually downloaded local datasets such as Kaggle-style CSV/TSV files or Hugging Face-style JSON/JSONL files. It never downloads data, never calls Kaggle or Hugging Face, never requires API tokens, and never trains a model.

Place local dataset files here:

```powershell
D:\yeni_ai_hiyerarsisi\krm_core\veriler\datasets
```

Supported ingestible formats:

- `.jsonl`
- `.json`
- `.csv`
- `.tsv`
- `.txt`
- `.md`
- `.markdown`
- `.html`
- `.htm`

Detected but not supported in V0.4:

- `.parquet`
- `.arrow`
- `.sqlite`
- `.db`
- `.zip`
- `.gz`

V0.4 inspects schemas, guesses fields such as `question`, `answer`, `title`, `text`, `instruction`, `input`, and `output`, and allows explicit field mapping overrides during ingestion. Dataset rows are converted into the existing `SourceArticle` and `SourcePointer` pipeline, then the normal concept extraction, shard building, retrieval, resonance, and mock answer flow can run.

Dataset commands:

```powershell
python scripts\inspect_dataset.py --source veriler\datasets
python scripts\inspect_dataset.py --source veriler\datasets --json
python scripts\dataset_quality_report.py --source veriler\datasets
python scripts\ingest_dataset.py --source veriler\datasets --out data\mind_dataset --profile tiny --dry-run
python scripts\ingest_dataset.py --source veriler\datasets --out data\mind_dataset --profile tiny
python scripts\build_shards.py --mind data\mind_dataset --profile tiny
python scripts\run_query.py --mind data\mind_dataset --query "your query here" --trace
python scripts\forge_training_examples.py --mind data\mind_dataset --out data\training_examples --max-examples 1000
```

Field override examples:

```powershell
python scripts\ingest_dataset.py --source veriler\datasets --out data\mind_dataset --question-field prompt --answer-field completion --profile tiny
python scripts\ingest_dataset.py --source veriler\datasets --out data\mind_dataset --title-field heading --text-field body --profile tiny
```

The training example forge exports small deterministic JSONL examples for router, domain detector, retrieval, planner, and evaluation workflows. It does not train models and does not add ML dependencies.

## V0.5 From-Scratch Training Forge

V0.5 adds the infrastructure for a small KRM-native model to learn KRM behavior from birth. This is not large model training. The first target is a tiny 10M to 30M parameter decoder experiment that can learn source-grounded behavior, concept extraction, domain and intent routing, ghost-edge style reasoning, answer planning, uncertainty handling, speculation labeling, and "do not claim without support" rules.

Core install still does not require training libraries. Optional PyTorch support is isolated behind the `train` extra:

```powershell
pip install -e .[train]
```

V0.5 does not download datasets, does not call cloud APIs, does not use Kaggle or Hugging Face tokens, and does not train a large model by default. Notebook templates are offline-first guides only; they assume you upload local artifacts manually.

Training forge commands:

```powershell
python scripts\build_training_corpus.py --mind data\mind_dataset --examples data\training_examples --out data\training_corpus --profile tiny --dry-run
python scripts\build_training_corpus.py --mind data\mind_dataset --examples data\training_examples --out data\training_corpus --profile tiny
python scripts\train_tokenizer.py --corpus data\training_corpus --out data\tokenizers\krm_byte_tokenizer
python scripts\estimate_training_resources.py --config 10m --batch-size 4 --seq-len 512
python scripts\train_tiny_core.py --corpus data\training_corpus --tokenizer data\tokenizers\krm_byte_tokenizer --out data\models\krm_native_10m --config 10m --max-steps 100 --dry-run
```

If the optional train extra is installed:

```powershell
python scripts\train_tiny_core.py --corpus data\training_corpus --tokenizer data\tokenizers\krm_byte_tokenizer --out data\models\krm_native_10m --config 10m --max-steps 100
python scripts\eval_tiny_core.py --model data\models\krm_native_10m --tokenizer data\tokenizers\krm_byte_tokenizer --corpus data\training_corpus
python scripts\export_tiny_core.py --model data\models\krm_native_10m --out data\models\exports\krm_native_10m_export
```

Notebook templates:

- `notebooks\KRM_From_Scratch_Training_Colab_Template.md`
- `notebooks\KRM_From_Scratch_Training_Kaggle_Template.md`

Generated training artifact families:

- `data\training_corpus`
- `data\tokenizers`
- `data\models`
- `data\checkpoints`
- `data\training_runs`
- `data\eval_reports`

## V0.1 Intelligence Quality Pass

V0.1 improves the working vertical slice without adding heavyweight infrastructure. Concept extraction now uses stronger deterministic heuristics for titles, headings, technical phrases, repeated terms, cause/comparison cues, and domain keywords. Domain-aware retrieval adds exact title matching, lexical overlap, source/title proximity, fingerprint similarity, domain priors, query intent boosts, and explained matches.

Ghost edges now include stronger pattern rules such as `CAUSES`, `ENABLES`, `INPUT_OUTPUT`, `CONTRASTS`, `TEMPORAL_NEAR`, and `COUNTERFACTUAL_LINK`. The resonance engine uses intent-sensitive edge weights and records a compact trace with initial activations, per-round top concepts, top edge transfers, pruning decisions, degradation decisions, and final attractor clusters.

Answer plans are now structured into sections with purpose, concepts, source refs, bullet claims, uncertainty, confidence, speculation requirements, supporting paths, and expected missing information. The mock composer uses these plans to produce deterministic but more readable grounded answers.

Quality tools:

```powershell
python scripts\run_query.py --mind data\mind --query "Why did the Industrial Revolution start in Britain?" --trace
python scripts\compare_baselines.py --mind data\mind
python scripts\evaluate_demo_quality.py --mind data\mind
```

Still not included in V0.1:

- Real ZIM parsing.
- Real local LLM backend.
- Rust acceleration.
- Large-scale vector indexing.
- GPU inference.

## Resource Profiles

- Ultra Lite: lowest RAM and disk target, fewer candidates, fewer resonance rounds.
- Local Core: default V0 target for the RTX 4050 laptop class machine.
- Colossus Lite: roadmap profile only in V0 configs.

When a query is too large, KRM-Core degrades by reducing resonance rounds, ghost edges, candidates, loaded shards, verifier usage, and source snippet length before returning a lower-confidence answer.

## Current Limitations

- Real ZIM parsing is optional and depends on a manually installed backend; without one, ZIM commands report the limitation gracefully.
- `LocalModel` is a stub; `MockModel` is the only V0 composer backend.
- Extraction is heuristic and intentionally small.
- Storage is JSONL for clarity and rebuildability.
- Dataset support is stdlib-only; parquet, arrow, sqlite, compressed archives, and large-scale dataset processing are future work.
- V0.5 training scripts are experimental and default to dry-run or tiny local smoke settings.

## Roadmap

- Real Kiwix/ZIM parser.
- GGUF or ONNX local model backend.
- Parquet/Arrow support behind optional lightweight adapters.
- Actual model training after the JSONL training examples are reviewed.
- Better KRM-native model evaluation and inference integration.
- Int8 or binary tiny vector index.
- Memory-mapped shard files.
- Rust resonance engine.
- Better extraction and entity normalization.
- Source-grounded verifier.
- Benchmark suite against small LLM and RAG baselines.
- Adaptive cache and dynamic shard loading.
- Multilingual and Turkish-first concept cards.
- GUI or local web interface.
