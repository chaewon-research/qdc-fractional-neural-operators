"""Aggregate high-resolution activation-minus-weight gaps and paired Student-t intervals."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
from qdc_fno.stats import paired_t_interval
p=argparse.ArgumentParser(); p.add_argument('files',nargs=5); p.add_argument('--out',required=True); a=p.parse_args()
rows=[json.loads(Path(f).read_text()) for f in a.files]
def one(metric):
    d=[float(r['q6_activation'][metric])-float(r['q6_weight'][metric]) for r in rows]
    mean,lo,hi=paired_t_interval(d,.95)
    return {'differences':d,'mean':mean,'ci95':[lo,hi],'ordering_activation_worse':sum(x>0 for x in d)}
def absolute(variant,metric):
    vals=[float(r[variant][metric]) for r in rows]
    return {'values':vals,'mean':statistics.fmean(vals),'sample_sd':statistics.stdev(vals) if len(vals)>1 else 0.0}
absolute_summary={v:{m:absolute(v,m) for m in ['rel_l2','residual']} for v in ['fp32','q6_weight','q6_activation']}
out={'resolution':rows[0]['resolution'],'seeds':[r['seed'] for r in rows],
     'absolute_summary':absolute_summary,
     'paired_gap_summary':{'rel_l2':one('rel_l2'),'residual':one('residual')},
     # Backward-compatible aliases retained for existing consumers.
     'rel_l2_gap':one('rel_l2'),'residual_gap':one('residual'),
     'interval':'paired Student-t across matched seed differences'}
Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
