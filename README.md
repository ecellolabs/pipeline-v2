# agentic

## Usage scripts

`mmagentic.datasets` registers three custom datasets: `mmlongbench_doc`, `mpdocvqa`, and `slidevqa`. Both usage scripts work with any of the three — just swap the dataset name.

Activate the environment first, then run scripts as plain Python from the repo root:

```bash
source .venv/bin/activate
```

### Step 1 — `usage/00_prepare_dataset.py`

Loads a registered dataset, optionally caches it, and visualizes the first sample of each split. Run this first as a sanity check of what a dataset actually contains before running anything heavier on it.

```bash
python usage/00_prepare_dataset.py mmlongbench_doc
python usage/00_prepare_dataset.py mpdocvqa
python usage/00_prepare_dataset.py slidevqa
```

Output: per-split sample visualizations under `./test/<name>/<split>/`.

Pass `--split {train,validation,test}` to build and load only that split's files, instead of all splits.

Pass `--enable-caching` to write a fast-reload cache of the dataset via `atria_core`'s `Cacher` (`--storage-type {msgpack,deltalake}`, default `deltalake`). Images are always kept as on-disk files referenced by path rather than embedded in the cache, so reload stays cheap regardless of storage type.

### Step 2 — `usage/01_preprocess.py`

Runs `DoclingTransform` (layout analysis + OCR via docling) over every page of every sample in a dataset split, writing each page's parsed `DoclingDocument` to disk.

```bash
python usage/01_preprocess.py mmlongbench_doc --num-workers 4
python usage/01_preprocess.py mpdocvqa --num-workers 4
python usage/01_preprocess.py slidevqa --num-workers 4
```

- First argument — the registered dataset name (`mmlongbench_doc`, `mpdocvqa`, or `slidevqa`).
- Outputs are written inside the dataset's own `data_dir` (wherever that dataset was downloaded/cached to) at `<data_dir>/docling/<split>/<sample.key>/<page.key>.json` — no separate output path needed.
- `--split {train,validation,test}` — build and process only that split, instead of all of them.
- `--num-workers` — number of worker processes; each builds its own `DoclingTransform` (docling's converter isn't cheaply shareable across processes). Defaults to `1` (no multiprocessing).

Already-parsed pages are skipped on re-run, so the script can be safely re-invoked to resume an interrupted run.
