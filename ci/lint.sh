#!/usr/bin/env bash

uv run mypy src  # type check
uv run ruff check src        # linter
uv run ruff format src --check # formatter
