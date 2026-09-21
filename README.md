# pipeline-v2

A multi-page multimodal document dataset preparation and preprocessing pipeline for Document Visual Question Answering (DocVQA) and Document Analysis.

---

## Repository Structure

```text
pipeline-v2/
├── pyproject.toml              # Project metadata, dependencies, and build config
├── uv.lock                     # Deterministic dependency lockfile
├── README.md                   # Pipeline documentation & architecture guidelines
├── ci/                         # Code quality & CI automation scripts
│   ├── run_checks.sh           # Main verification runner (lint, format, test)
│   ├── lint.sh                 # Type checking (mypy) & linter (ruff)
│   ├── format.sh               # Code formatter (ruff)
│   ├── test.sh                 # Unit tests & coverage runner (pytest)
│   └── bump.sh                 # Version bumping script
├── src/
│   └── pipeline_v2/            # Core package: reusable loaders, parsers, and transforms
│       ├── __init__.py
│       ├── datasets/           # Dataset loaders and adapters (registered with atria_core)
│       │   ├── __init__.py     # Re-exports MMLongBenchDoc, MPDocVQA, SlideVQA
│       │   ├── mmlongbench_doc.py
│       │   ├── mpdocvqa.py
│       │   ├── slidevqa.py
│       │   └── utils.py        # Manual download guards & exceptions
│       ├── parsers/            # Document parsing engines
│       │   └── docling.py      # Docling layout analysis & OCR transform
│       └── processors/         # Downstream transforms & model integrations (e.g. LLMs)
└── usage/                      # Sequential, stateless stage CLI drivers
    ├── 00_prepare_dataset.py   # Stage 0: Dataset ingestion, caching, & visualization
    ├── 01_preprocess.py        # Stage 1: Batch Docling OCR & layout parsing
    └── 02_...                  # Stage 2+: Downstream components (indexing, LLM extraction)
```

---

## Setup & Authentication

1. **Install dependencies and sync environment**:
   ```bash
   uv sync
   ```

2. **Authenticate with Hugging Face**:
   Datasets such as `slidevqa` require Hugging Face authentication. Log in once with:
   ```bash
   uv run hf auth login
   ```
   *(Paste your Hugging Face access token when prompted. Credentials will be stored locally at `~/.cache/huggingface/token`.)*

3. **Activate the virtual environment**:
5. **MP-DocVQA Manual Dataset Setup (Cluster Setup)**:
   While metadata/IMDBs (`mpdocvqa_imdbs.zip`) can be downloaded automatically, the MP-DocVQA page images (21.2 GB `images.tar.gz`) must be downloaded manually or directly fetched on the cluster into `<data_dir>/mpdocvqa/images`.

   To download directly on your cluster into your data directory (e.g., `/netscratch/$USER/data`):
   ```bash
   cd /netscratch/akhtar/data/mpdocvqa

   # Download images archive directly on cluster (supports resuming):
   wget --no-check-certificate -c https://datasets.cvc.uab.es/rrc/DocVQA/Task4/images.tar.gz

   # Extract into images directory so .jpg files are directly inside:
   mkdir -p images
   tar -xzf images.tar.gz -C images/
   ```

---

## Usage scripts

`pipeline_v2.datasets` registers three custom datasets: `mmlongbench_doc`, `mpdocvqa`, and `slidevqa`. Both usage scripts work with any of the three — just swap the dataset name.

### Step 1 — `usage/00_prepare_dataset.py`

Loads a registered dataset, optionally caches it, and visualizes the first sample of each split. Run this first as a sanity check of what a dataset actually contains before running anything heavier on it.

```bash
python usage/00_prepare_dataset.py mmlongbench_doc
python usage/00_prepare_dataset.py mpdocvqa
python usage/00_prepare_dataset.py slidevqa
```

Output: per-split sample visualizations under `./dataset_visualizations/<name>/<split>/`.

Pass `--split {train,validation,test}` to build and load only that split's files, instead of all splits.

Pass `--max-samples N` to restrict loading and caching to `N` samples/decks (ideal for fast pilot runs on a laptop before cluster deployment).

Pass `--enable-caching` (on by default; use `--no-enable-caching` to disable) to write a fast-reload cache of the dataset via `atria_core`'s `Cacher` (`--storage-type msgpack`, the default). Images are always kept as on-disk files referenced by path rather than embedded in the cache, so reload stays cheap. With and without caching the datasets get prepared already but caching additionally allows faster loading of the data instead of reading heavy huggingface files over and over on reach run. Mspgack caching is very fast and optimied for loading.

### Step 2 — `usage/01_preprocess.py`

Runs `DoclingTransform` (layout analysis + OCR via the external Docling API service) over every page of every sample in a dataset split, writing each page's parsed `DoclingDocument` to disk.

> [!IMPORTANT]
> The Docling API server address (e.g. `http://serv-3334:10001`) is **not constant**; the DFKI cluster node hostname and port change dynamically with each job allocation.
> You must pass the active API URL using the `--api-url` parameter, or set the `DOCLING_API_URL` environment variable.

```bash
# Pass the cluster endpoint directly:
python usage/01_preprocess.py mmlongbench_doc --api-url http://serv-3334:10001 --num-workers 4
python usage/01_preprocess.py mpdocvqa --api-url http://serv-3334:10001 --num-workers 4
python usage/01_preprocess.py slidevqa --api-url http://serv-3334:10001 --num-workers 4

# Or set it once in your environment:
export DOCLING_API_URL="http://serv-3334:10001"
python usage/01_preprocess.py mmlongbench_doc --num-workers 4
```

- First argument — the registered dataset name (`mmlongbench_doc`, `mpdocvqa`, or `slidevqa`).
- `--api-url` — Docling API service URL (e.g. `http://serv-3334:10001`). Required unless `DOCLING_API_URL` is set in the environment.
- Outputs are written inside the dataset's own `data_dir` (wherever that dataset was downloaded/cached to) at `<data_dir>/docling/<split>/<sample.key>/<page.key>.json` — no separate output path needed.
- `--split {train,validation,test}` — build and process only that split, instead of all of them.
- `--max-samples N` — preprocess only up to `N` samples/decks.
- `--num-workers` — number of concurrent worker processes making API requests. Defaults to `1`.
- `--no-ocr` — disable OCR in API requests (enabled by default).

Already-parsed pages are skipped on re-run, so the script can be safely re-invoked to resume an interrupted run.

### Pipeline Flow & Next Steps

The end-to-end pipeline follows this decoupled stage flow:

```text
load -> MultiPageDocumentInstance -> preprocess transform -> ParsedInstance
     -> tree indexing -> ParsedWithTree -> map samples to PydanticAI dataset
```

---

## Architectural & Structural Guidelines

When extending or adding stages to `pipeline-v2`, adhere to the following design principles:

### 1. Stateless Pipeline Stages
- Each stage in the pipeline corresponds to an arrow in the flow above.
- Every stage must remain completely stateless: **read inputs from disk, write outputs to disk, and touch nothing else**.
- Keep every stage runnable as an independent script in `usage/`, identical to `00_prepare_dataset.py` and `01_preprocess.py`. This ensures fast debugging and inspection of intermediate artifacts without rerunning upstream compute.

### 2. Idempotency & Resumability
- Long-running batch jobs (e.g., OCR, layout parsing, model inference) may be interrupted or preempted on compute clusters.
- Always check if the target output file already exists before processing:
  ```python
  if out_path.exists():
      continue
  ```
- Re-running any script should safely resume and only process unfinished samples.

### 3. Standardized Output Locations
- All stage outputs must be stored inside the dataset's own `data_dir` to keep dataset artifacts self-contained:
  - **Stage 0 (Ingestion & Cache)**: `<data_dir>/` (handled by `atria_core` cacher)
  - **Stage 1 (Docling Layout & OCR)**: `<data_dir>/docling/<split>/<sample.key>/<page.key>.json`
  - **Stage 2+ (Downstream Processing)**: `<data_dir>/<stage_name>/<split>/<sample.key>.json`

### 4. Separation of Concerns
- **`src/pipeline_v2/`**: Reusable modules, data models, parsing transforms, and API clients (e.g., `src/pipeline_v2/processors/gpt.py`). Code here should be importable as a clean library and never parse CLI arguments directly.
- **`usage/`**: Numbered CLI driver scripts (`00_prepare_dataset.py`, `01_preprocess.py`, `02_gpt_process.py`, etc.) that parse CLI flags via `argparse`, orchestrate multiprocessing/batching, and invoke transforms.

### 5. Adding New Components (Example: LLM / GPT API Processing)
To add a downstream component:
1. Create a processor class under `src/pipeline_v2/processors/<name>.py` that handles the transform/inference logic.
2. Create a driver script under `usage/02_<name>.py` that loads upstream artifacts from disk, calls the processor, and writes stage results back to `<data_dir>/<name>/<split>/`.

### 6. Downstream Agentic Pipeline Handoff
Once a dataset has been mapped to its final PydanticAI form, dump it to JSON. Those JSON files are the handoff point, loaded independently by the main Agentic Pipeline. This repository's responsibility ends there; it strictly prepares datasets.
