#!/bin/bash
export PYTHONUNBUFFERED=1

# Execute via srun on compute node with live logging in terminal
srun -c 8 --mem=32G uv run usage/01_preprocess.py slidevqa \
  --data-dir /netscratch/akhtar/data/slidevqa \
  --api-url http://serv-3331:10001 \
  --num-workers 8 "$@"