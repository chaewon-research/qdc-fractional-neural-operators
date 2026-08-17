#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the RealPDEBench data root}"
LOCK="${ASSET_LOCK:-external/realpdebench_assets.lock.json}"
mkdir -p results/realpde
for seed in 0 1 2 3 4; do
  ckpt="results/realpdebench/checkpoints/seed_${seed}_best.pth"
  test -f "$ckpt" || { echo "Missing $ckpt. Run scripts/run_realpdebench_finetune.sh first." >&2; exit 2; }
  python experiments/realpdebench_cylinder.py --dataset-root "$DATASET_ROOT" --checkpoint "$ckpt" --seed "$seed" --mode activation --asset-lock "$LOCK" --out "results/realpde/act_${seed}.json"
  python experiments/realpdebench_cylinder.py --dataset-root "$DATASET_ROOT" --checkpoint "$ckpt" --seed "$seed" --mode weight --asset-lock "$LOCK" --out "results/realpde/wt_${seed}.json"
done
python experiments/aggregate_realpdebench.py \
  --activation results/realpde/act_0.json results/realpde/act_1.json results/realpde/act_2.json results/realpde/act_3.json results/realpde/act_4.json \
  --weight results/realpde/wt_0.json results/realpde/wt_1.json results/realpde/wt_2.json results/realpde/wt_3.json results/realpde/wt_4.json \
  --out results/realpde/aggregate.json
