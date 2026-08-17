"""Create five seed-specific configs from the pinned upstream Cylinder/FNO recipe."""
from __future__ import annotations
import argparse, json, yaml
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--upstream-root',default='external/RealPDEBench')
p.add_argument('--dataset-root',required=True)
p.add_argument('--asset-lock',default='external/realpdebench_assets.lock.json')
p.add_argument('--base-checkpoint',default=None)
p.add_argument('--out-dir',default='work/realpdebench_configs',help='Local runtime configs; may contain absolute machine paths and should not be committed')
a=p.parse_args()
lock=json.loads(Path(a.asset_lock).read_text())
base=Path(a.upstream_root)/'realpdebench/configs/cylinder/fno.yaml'
if not base.exists(): raise SystemExit(f'Missing {base}; run scripts/prepare_realpdebench.sh first.')
if a.base_checkpoint:
    ckpt=Path(a.base_checkpoint)
else:
    ckpt=Path('external/realpdebench_checkpoints')/lock['base_finetune_checkpoint']
if not ckpt.exists(): raise SystemExit(f'Missing numerical initialization checkpoint {ckpt}; run scripts/prepare_realpdebench.sh first.')
out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
(out/'README_LOCAL_ONLY.txt').write_text('Generated RealPDEBench runtime configs may contain machine-specific absolute paths. Do not commit or include this directory in release archives.\n')
for seed in range(5):
    cfg=yaml.safe_load(base.read_text())
    cfg['seed']=seed
    cfg['dataset_root']=str(Path(a.dataset_root).resolve())
    cfg['results_path']=str((Path('results/realpdebench_training')/f'seed_{seed}').resolve())
    cfg['checkpoint_path']=str(ckpt.resolve())
    cfg['is_use_tb']=False
    dest=out/f'fno_cylinder_seed{seed}.yaml'
    dest.write_text(yaml.safe_dump(cfg,sort_keys=False))
    print(dest)
