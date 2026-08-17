#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
RUN_TRAIN="${RUN_TRAIN:-0}"
mkdir -p results/lobpcg_selection/validation results/lobpcg_selection/holdout

# Reproduce the paper's actual protocol:
#   (1) sweep all candidate K on validation artifacts;
#   (2) choose the smallest K reaching >=95% of the best validation high-accuracy
#       same-rank-reference Rel-L2 recovery;
#   (3) freeze K;
#   (4) evaluate only that K on the disjoint high-contrast holdout.
for spec in "32 82" "64 150" "128 200"; do
  read -r N EXPECTED_K <<<"$spec"
  candidate_aggs=()

  # Ensure disjoint validation and holdout prediction artifacts exist for every matched seed.
  for seed in 101 102 103 104 105; do
    base="results/highres${N}_seed${seed}"
    val_art="${base}.validation_qdc.pt"
    hold_art="${base}.qdc.pt"
    if [[ "$RUN_TRAIN" == "1" && ( ! -f "$val_art" || ! -f "$hold_art" ) ]]; then
      python experiments/train_higher_resolution.py \
        --config configs/higher_resolution.yaml --resolution "$N" --seed "$seed" --out "${base}.json"
    fi
    test -f "$val_art" || { echo "Missing $val_art. Re-run with RUN_TRAIN=1 using the updated high-resolution trainer." >&2; exit 2; }
    test -f "$hold_art" || { echo "Missing $hold_art. Re-run with RUN_TRAIN=1 using the updated high-resolution trainer." >&2; exit 2; }
  done

  # Validation-side candidate sweep.
  for K in 41 50 82 100 150 200 250; do
    files=()
    for seed in 101 102 103 104 105; do
      artifact="results/highres${N}_seed${seed}.validation_qdc.pt"
      out="results/lobpcg_selection/validation/N${N}_K${K}_seed${seed}.json"
      python experiments/run_lobpcg_qdc.py --artifact "$artifact" --k "$K" --out "$out"
      files+=("$out")
    done
    agg="results/lobpcg_selection/validation/N${N}_K${K}_aggregate.json"
    python experiments/aggregate_lobpcg.py "${files[@]}" --out "$agg"
    candidate_aggs+=("$agg")
  done

  selection="results/lobpcg_selection/N${N}_selection.json"
  python experiments/select_lobpcg_rank.py "${candidate_aggs[@]}" \
    --fraction 0.95 --expected-k "$EXPECTED_K" --out "$selection"
  SELECTED_K=$(python - "$selection" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['selected_k'])
PY
)

  # Disjoint holdout evaluation at the frozen selected K only.
  hold_files=()
  for seed in 101 102 103 104 105; do
    artifact="results/highres${N}_seed${seed}.qdc.pt"
    out="results/lobpcg_selection/holdout/N${N}_K${SELECTED_K}_seed${seed}.json"
    python experiments/run_lobpcg_qdc.py --artifact "$artifact" --k "$SELECTED_K" --out "$out"
    hold_files+=("$out")
  done
  python experiments/aggregate_lobpcg.py "${hold_files[@]}" \
    --out "results/lobpcg_selection/holdout/N${N}_K${SELECTED_K}_aggregate.json"
done

printf '%s\n' 'LOBPCG validation-selection reproduction complete: candidate ranks swept on validation, K frozen by the 95% rule, and only the selected K evaluated on disjoint holdout.'
