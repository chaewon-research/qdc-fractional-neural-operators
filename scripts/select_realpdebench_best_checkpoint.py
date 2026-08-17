"""Select the upstream validation-RMSE-best checkpoint and copy it to a stable seed path."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
import torch

p=argparse.ArgumentParser()
p.add_argument('--seed-root',required=True)
p.add_argument('--out',required=True)
a=p.parse_args()
root=Path(a.seed_root)
runs=[d for d in root.rglob('*') if d.is_dir() and list(d.glob('model_*.pth'))]
if not runs: raise SystemExit(f'No RealPDEBench training run with model_*.pth under {root}')
if len(runs) != 1:
    raise SystemExit(f'Expected exactly one RealPDEBench run under {root}; found {len(runs)}. Clear the seed directory or pass a unique seed root.')
run=runs[0]
files=sorted(run.glob('model_*.pth'))
last=torch.load(files[-1],map_location='cpu',weights_only=False)
best_it=int(last['best_iteration']); best_loss=float(last['best_val_loss'])
best=run/f'model_{best_it:04d}.pth'
if not best.exists():
    by_it={int(torch.load(f,map_location='cpu',weights_only=False).get('iteration',-1)):f for f in files}
    if best_it not in by_it: raise SystemExit(f'Best iteration {best_it} not found in saved checkpoints under {run}')
    best=by_it[best_it]
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(best,out)
def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def portable_id(path: Path, anchor: Path) -> str:
    try:
        return str(path.relative_to(anchor))
    except ValueError:
        return path.name

# Persist portable provenance only: no /home/<user>/... or cluster mount paths.
meta={
    'source_run_id': portable_id(run, root),
    'source_checkpoint_basename': best.name,
    'source_checkpoint_sha256': sha256_file(best),
    'best_iteration': best_it,
    'best_val_rmse': best_loss,
    'copied_checkpoint_basename': out.name,
    'copied_checkpoint_sha256': sha256_file(out),
}
out.with_suffix(out.suffix+'.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
