# agentic

## Usage scripts

Both scripts currently work with `slidevqa` (an `atria_datasets`-registered dataset). Activate the environment first, then run scripts as plain Python from the repo root:

```bash
source .venv/bin/activate
```

### `usage/prepare_dataset.py`

Loads a registered dataset, optionally caches it, and visualizes the first sample of each split. Useful for a quick sanity check of what a dataset actually contains before running anything heavier on it.

```bash
python usage/prepare_dataset.py slidevqa
```

Output: per-split sample visualizations under `./dataset_visualizations/slidevqa/<split>/`.

### `usage/preprocess.py`

Runs `DoclingTransform` (layout analysis + OCR via docling) over every page of every sample in a dataset split, writing each page's parsed `DoclingDocument` to disk.

```bash
python usage/preprocess.py slidevqa --num-workers 4
```

- `slidevqa` — the registered dataset name.
- Outputs are written inside the dataset's own `data_dir` (wherever that dataset was downloaded/cached to) at `<data_dir>/<split>/docling/<sample.key>/<page.key>.json` — no separate output path needed.
- `--num-workers` — number of worker processes; each builds its own `DoclingTransform` (docling's converter isn't cheaply shareable across processes). Defaults to `1` (no multiprocessing).

Already-parsed pages are skipped on re-run, so the script can be safely re-invoked to resume an interrupted run.
