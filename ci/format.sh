#!/usr/bin/env bash

uv run ruff check src --fix      # linter
uv run ruff format src --check # formatter
