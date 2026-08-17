#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
mkdir -p results/fresh_scalar_qdc
# The paper's independently frozen residual-margin selector chooses K=82 for each of these three fresh scalar runs.
# This workflow reproduces the held-out QDC application at that frozen rank. The selector implementation and
# strong/weak feasibility rules are in qdc_fno/qdc.py; supporting mixed/tie searches are not rerun by this focused artifact.
for seed in 45 46 47; do
  python experiments/run_fresh_scalar_qdc.py --config configs/fresh_scalar_qdc.yaml --seed "$seed" --k 82 --out "results/fresh_scalar_qdc/seed_${seed}.json"
done
python experiments/aggregate_fresh_scalar_qdc.py results/fresh_scalar_qdc/seed_45.json results/fresh_scalar_qdc/seed_46.json results/fresh_scalar_qdc/seed_47.json --out results/fresh_scalar_qdc/aggregate.json
