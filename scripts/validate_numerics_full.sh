#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTHONDONTWRITEBYTECODE=1
python experiments/validate_highres_targets.py \
  --resolutions 16 32 \
  --samples 4 \
  --out results/generated/target_solver_validation_full.json
