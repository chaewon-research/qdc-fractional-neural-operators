"""Aggregate fresh-scalar QDC seed outputs."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('files',nargs='+'); p.add_argument('--out',default='results/fresh_scalar_qdc/aggregate.json'); a=p.parse_args()
rows=[json.loads(Path(f).read_text()) for f in a.files]
out={'seeds':[r['seed'] for r in rows],'selected_k':[r['selected_k'] for r in rows],'mode_selection':rows[0]['mode_selection'],'metrics':{}}
for m in ['rel_l2','event_rel_l2','hf_log','residual','objective']:
    vals=[float(r['qdc'][m]) for r in rows]
    out['metrics'][m]={'values':vals,'mean':statistics.fmean(vals),'sample_sd':statistics.stdev(vals) if len(vals)>1 else 0.0}
vals=[float(r['ptq_activation_rel_l2']) for r in rows]; out['ptq_activation_rel_l2']={'values':vals,'mean':statistics.fmean(vals),'sample_sd':statistics.stdev(vals)}
Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
