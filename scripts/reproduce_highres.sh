#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
for N in 64 128; do
  files=()
  for seed in 101 102 103 104 105; do
    out="results/highres${N}_seed${seed}.json"
    python experiments/train_higher_resolution.py --config configs/higher_resolution.yaml --resolution "$N" --seed "$seed" --out "$out"
    files+=("$out")
  done
  python experiments/aggregate_highres.py "${files[@]}" --out "results/highres${N}_aggregate.json"
done
