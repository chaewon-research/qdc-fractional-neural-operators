#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTHONDONTWRITEBYTECODE=1

# Closed-world release check. Local/generated output paths are intentionally ignored so
# this command is idempotent; a fresh release archive itself must contain only manifest
# payload files plus MANIFEST.txt and SHA256SUMS.
python - <<'PY'
from pathlib import Path
root=Path('.')
manifest={line.strip() for line in Path('MANIFEST.txt').read_text().splitlines() if line.strip()}
allowed=manifest | {'MANIFEST.txt','SHA256SUMS'}

def generated_or_cache(rel: str) -> bool:
    parts=rel.split('/')
    return (
        rel.startswith('results/generated/') or
        rel.startswith('work/') or
        '.pytest_cache' in parts or
        '__pycache__' in parts or
        rel.endswith(('.pyc','.pyo'))
    )

actual=set()
for p in root.rglob('*'):
    if p.is_file():
        rel=p.relative_to(root).as_posix()
        if generated_or_cache(rel):
            continue
        actual.add(rel)
extra=sorted(actual-allowed)
missing=sorted(manifest-actual)
if extra or missing:
    if extra:
        print('Unexpected unmanifested release files:')
        for x in extra: print('  +',x)
    if missing:
        print('Missing manifest files:')
        for x in missing: print('  -',x)
    raise SystemExit(1)
print(f'Closed-world manifest check passed: {len(manifest)} payload files, no unexpected release files.')
PY

if [[ -f SHA256SUMS ]]; then sha256sum -c SHA256SUMS; fi
python -m pytest -q -p no:cacheprovider
# Quick one-sample exact-dense numerical sanity check. The complete 16/32 audit is
# intentionally separate because it can take several minutes on CPU-only machines.
python experiments/validate_highres_targets.py --resolutions 16 --samples 1 --out results/generated/target_solver_validation_quick.json
python scripts/make_core_diagnosis_figure.py --out results/generated/core_diagnosis.pdf
python experiments/run_smoke_scalar.py --out results/generated/smoke_execution_only.json
printf '%s\n' 'Quick package validation complete. For the full exact-dense 16/32 numerical audit run: bash scripts/validate_numerics_full.sh'
printf '%s\n' 'Central workflows: reproduce_core.sh, reproduce_qdc.sh, reproduce_highres.sh, reproduce_lobpcg.sh, reproduce_lobpcg_sweep.sh, and reproduce_realpdebench.sh.'
