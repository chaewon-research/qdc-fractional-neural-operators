#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
mkdir -p results/core_scalar
for seed in 42 43 44; do
  python experiments/train_scalar.py --config configs/core_scalar.yaml --seed "$seed" --out "results/core_scalar/seed_${seed}.json"
done
python experiments/aggregate_core_scalar.py results/core_scalar/seed_42.json results/core_scalar/seed_43.json results/core_scalar/seed_44.json --out results/core_scalar/aggregate.json
