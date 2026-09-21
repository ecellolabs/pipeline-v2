#!/bin/bash
export PYTHONUNBUFFERED=1

# Execute via srun on compute node with minimal resources (-c 2, 4G RAM)
srun -c 2 --mem=4G uv run usage/01_preprocess.py slidevqa \
  --data-dir /netscratch/akhtar/data/slidevqa \
  --api-url http://serv-3331:10001 \
  --num-workers 4 "$@"
