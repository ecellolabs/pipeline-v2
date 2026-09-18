#!/usr/bin/env bash

uv run coverage run --source=pipeline_v2 -m pytest $@
uv run coverage report --show-missing
