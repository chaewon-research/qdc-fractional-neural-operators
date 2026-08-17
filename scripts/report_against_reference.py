"""Report fresh aggregate deviations from the paper's authoritative aggregate reference values.

This tool is descriptive: it never rewrites or silently substitutes reported paper numbers.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--core',default=None); p.add_argument('--qdc',default=None)
p.add_argument('--highres64',default=None); p.add_argument('--highres128',default=None); p.add_argument('--realpdebench',default=None)
p.add_argument('--reference',default='results/reference/reported_results.json')
a=p.parse_args(); ref=json.loads(Path(a.reference).read_text()); rows={}
def load(x): return json.loads(Path(x).read_text()) if x else None
def delta(got,want): return {'fresh':float(got),'reported_reference':float(want),'difference':float(got)-float(want)}
if a.core:
    d=load(a.core); out={}
    for v in ('fp32','q6_weight','q6_activation'):
        out[v]={m:delta(d['variants'][v][m]['mean'],ref['core_scalar_high_contrast']['metric_summary'][v][m][0]) for m in ('rel_l2','event_rel_l2','hf_log','residual')}
    rows['core']=out
if a.qdc:
    d=load(a.qdc); rows['qdc']={m:delta(d['metrics'][m]['mean'],w) for m,w in {'rel_l2':.0156,'residual':.2694,'objective':.0679}.items()}
for label,path,n in [('highres64',a.highres64,'64'),('highres128',a.highres128,'128')]:
    if path:
        d=load(path); rr=ref['higher_resolution'][n]
        rows[label]={'rel_gap':delta(d['rel_l2_gap']['mean'],rr['rel_gap'][0]),'residual_gap':delta(d['residual_gap']['mean'],rr['residual_gap'][0])}
if a.realpdebench:
    d=load(a.realpdebench); rr=ref['realpdebench_cylinder']; rows['realpdebench']={'rel_gap':delta(d['rel_l2']['mean'],rr['rel_gap'][0]),'frmse_gap':delta(d['frmse']['mean'],rr['frmse_gap'][0])}
print(json.dumps(rows,indent=2))
