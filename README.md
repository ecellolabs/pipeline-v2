# agentic

## Usage scripts

`agentic.datasets` registers three custom datasets: `mmlongbench_doc`, `mp_docvqa`, and `slidevqa`. Both usage scripts work with any of the three — just swap the dataset name.

Activate the environment first, then run scripts as plain Python from the repo root:

```bash
source .venv/bin/activate
```

### Step 1 — `usage/00_prepare_dataset.py`

Loads a registered dataset, optionally caches it, and visualizes the first sample of each split. Run this first as a sanity check of what a dataset actually contains before running anything heavier on it.

```bash
python usage/00_prepare_dataset.py mmlongbench_doc
python usage/00_prepare_dataset.py mp_docvqa
python usage/00_prepare_dataset.py slidevqa
```

Output: per-split sample visualizations under `./test/<name>/<split>/`.

### Step 2 — `usage/01_preprocess.py`

Runs `DoclingTransform` (layout analysis + OCR via docling) over every page of every sample in a dataset split, writing each page's parsed `DoclingDocument` to disk.

```bash
python usage/01_preprocess.py mmlongbench_doc --num-workers 4
python usage/01_preprocess.py mp_docvqa --num-workers 4
python usage/01_preprocess.py slidevqa --num-workers 4
```

- First argument — the registered dataset name (`mmlongbench_doc`, `mp_docvqa`, or `slidevqa`).
- Outputs are written inside the dataset's own `data_dir` (wherever that dataset was downloaded/cached to) at `<data_dir>/docling/<split>/<sample.key>/<page.key>.json` — no separate output path needed.
- `--num-workers` — number of worker processes; each builds its own `DoclingTransform` (docling's converter isn't cheaply shareable across processes). Defaults to `1` (no multiprocessing).

Already-parsed pages are skipped on re-run, so the script can be safely re-invoked to resume an interrupted run.
