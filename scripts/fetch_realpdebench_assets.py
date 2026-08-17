"""Download the immutable public RealPDEBench checkpoint needed for five-seed fine-tuning."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from huggingface_hub import hf_hub_download

p=argparse.ArgumentParser()
p.add_argument('--asset-lock',default='external/realpdebench_assets.lock.json')
p.add_argument('--out-dir',default='external/realpdebench_checkpoints')
a=p.parse_args()
lock=json.loads(Path(a.asset_lock).read_text())
out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
path=hf_hub_download(repo_id=lock['model_repository'], filename=lock['base_finetune_checkpoint'],
                     revision=lock['model_revision'], local_dir=out)
print(Path(path).resolve())
