"""Aggregate three independently trained scalar runs into paper-style mean and sample SD."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('files',nargs='+'); p.add_argument('--out',default='results/core_scalar/aggregate.json'); a=p.parse_args()
rows=[json.loads(Path(f).read_text()) for f in a.files]
variants=['fp32','q6_weight','q6_activation']; metrics=['rel_l2','event_rel_l2','hf_log','residual']
out={'seeds':[r['seed'] for r in rows],'variants':{}}
for v in variants:
    out['variants'][v]={}
    for m in metrics:
        vals=[float(r[v][m]) for r in rows]
        out['variants'][v][m]={'values':vals,'mean':statistics.fmean(vals),'sample_sd':statistics.stdev(vals) if len(vals)>1 else 0.0}
Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
