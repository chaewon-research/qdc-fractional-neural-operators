#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-external/RealPDEBench}"
LOCK="${ASSET_LOCK:-external/realpdebench_assets.lock.json}"
COMMIT="$(python - "$LOCK" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['source_git_commit'])
PY
)"
if [[ ! -d "$ROOT/.git" ]]; then git clone https://github.com/AI4Science-WestlakeU/RealPDEBench.git "$ROOT"; fi
git -C "$ROOT" fetch --all --tags
git -C "$ROOT" checkout --detach "$COMMIT"
python -m pip install -e "$ROOT" --no-deps
python scripts/fetch_realpdebench_assets.py --asset-lock "$LOCK" >/tmp/qdc_realpde_base_checkpoint.txt
printf 'Pinned RealPDEBench source: %s\n' "$COMMIT"
printf 'Pinned numerical initialization checkpoint: %s\n' "$(cat /tmp/qdc_realpde_base_checkpoint.txt)"
