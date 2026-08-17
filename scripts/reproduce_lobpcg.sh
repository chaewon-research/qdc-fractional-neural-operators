#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
RUN_TRAIN="${RUN_TRAIN:-0}"
mkdir -p results/lobpcg results/lobpcg32
# Reproduce the reported 32x32 same-rank QDC gain-retention result, then the selected 64/128 ranks.
files32=()
for seed in 101 102 103 104 105; do
  base="results/highres32_seed${seed}"
  if [[ "$RUN_TRAIN" == "1" ]]; then
    python experiments/train_higher_resolution.py --config configs/higher_resolution.yaml --resolution 32 --seed "$seed" --out "${base}.json"
  fi
  artifact="${base}.qdc.pt"
  test -f "$artifact" || { echo "Missing $artifact. Re-run with RUN_TRAIN=1 or generate the N32 artifact first." >&2; exit 2; }
  out="results/lobpcg/N32_seed${seed}.json"
  python experiments/run_lobpcg_qdc.py --artifact "$artifact" --k 82 --out "$out"
  files32+=("$out")
done
python experiments/aggregate_lobpcg.py "${files32[@]}" --out "results/lobpcg/N32_aggregate.json"

# Saved high-resolution FNO predictions are required for the selected-rank QDC recovery calculation at 64/128.
for spec in "64 150" "128 200"; do
  read -r N K <<<"$spec"
  files=()
  for seed in 101 102 103 104 105; do
    base="results/highres${N}_seed${seed}"
    if [[ "$RUN_TRAIN" == "1" ]]; then
      python experiments/train_higher_resolution.py --config configs/higher_resolution.yaml --resolution "$N" --seed "$seed" --out "${base}.json"
    fi
    artifact="${base}.qdc.pt"
    test -f "$artifact" || { echo "Missing $artifact. Re-run with RUN_TRAIN=1 or generate the high-resolution artifacts first." >&2; exit 2; }
    out="results/lobpcg/N${N}_seed${seed}.json"
    python experiments/run_lobpcg_qdc.py --artifact "$artifact" --k "$K" --out "$out"
    files+=("$out")
  done
  python experiments/aggregate_lobpcg.py "${files[@]}" --out "results/lobpcg/N${N}_aggregate.json"
done
printf '%s\n' 'LOBPCG diagnostics complete: N32 same-rank gain retention and frozen selected ranks at N64/N128. Candidate-rank search is documented and implemented separately from this focused rerun.'
