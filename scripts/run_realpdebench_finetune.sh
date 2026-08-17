#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-external/RealPDEBench}"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to a writable RealPDEBench data root}"
LOCK="${ASSET_LOCK:-external/realpdebench_assets.lock.json}"
DATA_REV="$(python - "$LOCK" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['dataset_revision'])
PY
)"
BASE="$(python scripts/fetch_realpdebench_assets.py --asset-lock "$LOCK")"
python experiments/prepare_realpdebench_seed_configs.py --upstream-root "$UPSTREAM_ROOT" --dataset-root "$DATASET_ROOT" --asset-lock "$LOCK" --base-checkpoint "$BASE"
mkdir -p results/realpdebench/checkpoints
for seed in 0 1 2 3 4; do
  seed_root="results/realpdebench_training/seed_${seed}"
  rm -rf "$seed_root"
  cfg="work/realpdebench_configs/fno_cylinder_seed${seed}.yaml"
  python -m realpdebench.train --config "$cfg" --train_data_type real --is_finetune \
    --use_hf_dataset --hf_auto_download --hf_revision "$DATA_REV"
  python scripts/select_realpdebench_best_checkpoint.py \
    --seed-root "results/realpdebench_training/seed_${seed}" \
    --out "results/realpdebench/checkpoints/seed_${seed}_best.pth"
done
printf '%s\n' 'Five independent fine-tuning runs finished and validation-RMSE-best checkpoints were copied to results/realpdebench/checkpoints/.'
