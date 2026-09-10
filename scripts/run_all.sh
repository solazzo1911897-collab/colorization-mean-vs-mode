#!/usr/bin/env bash
# Everything, from a fresh clone to the figures. Took about three hours on my M2.
set -euo pipefail
cd "$(dirname "$0")/.."

for seed in 0 1 2; do
  uv run python scripts/train.py config/l2.yaml --seed "$seed"
  uv run python scripts/train.py config/classification.yaml --seed "$seed"
done

uv run python scripts/evaluate.py
uv run python scripts/mechanism.py
uv run python scripts/bins_check.py
uv run python scripts/figures.py
uv run python scripts/entropy.py
uv run python scripts/report_numbers.py
