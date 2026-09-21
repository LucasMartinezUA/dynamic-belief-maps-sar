#!/usr/bin/env bash
# Anonymous release flow: install the locked environment, fetch the pinned
# datasets, verify the frozen artifact manifest and run the test suite.
set -euo pipefail
cd "$(dirname "$0")/.."

pixi install
pixi run python scripts/download_datasets.py
pixi run python scripts/verify_public_artifacts.py
pixi run pytest tests/ -q
