#!/usr/bin/env bash

uv run coverage run --source=atria_datasets -m pytest $@
uv run coverage report --show-missing
