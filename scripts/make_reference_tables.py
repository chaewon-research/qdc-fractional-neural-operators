"""Export the machine-readable paper reference results to compact CSV tables."""
import csv,json
from pathlib import Path
root=Path('results/reference'); d=json.loads((root/'reported_results.json').read_text()); out=root/'csv'; out.mkdir(exist_ok=True)
with (out/'core_scalar.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['variant','rel_mean','rel_std','event_mean','event_std','hf_mean','hf_std','residual_mean','residual_std'])
    for name,m in d['core_scalar_high_contrast']['metric_summary'].items(): w.writerow([name,*m['rel_l2'],*m['event_rel_l2'],*m['hf_log'],*m['residual']])
with (out/'generalization.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['setting','solution_gap','lo','hi','physical_gap','physical_lo','physical_hi','order'])
    for n,v in d['higher_resolution'].items(): w.writerow([f'fractional_{n}',*v['rel_gap'],*v['residual_gap'],v['order']])
    r=d['realpdebench_cylinder']; w.writerow(['realpdebench_cylinder',*r['rel_gap'],*r['frmse_gap'],r['order']])
print(out)
