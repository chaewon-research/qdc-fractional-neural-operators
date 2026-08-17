import argparse,json
from pathlib import Path
from qdc_fno.stats import paired_t_interval
p=argparse.ArgumentParser(); p.add_argument('--activation',nargs=5,required=True); p.add_argument('--weight',nargs=5,required=True); p.add_argument('--out',default='results/realpdebench_summary.json'); a=p.parse_args()
A=[json.loads(Path(x).read_text()) for x in a.activation]; W=[json.loads(Path(x).read_text()) for x in a.weight]
for aa,ww in zip(A,W):
    if aa['seed']!=ww['seed']: raise SystemExit('Activation and weight files must be matched by seed.')
def summary(key):
    dif=[aa['metrics'][key]-ww['metrics'][key] for aa,ww in zip(A,W)]; mean,lo,hi=paired_t_interval(dif,.95); return {'differences':dif,'mean':mean,'ci95':[lo,hi],'ordering_activation_worse':sum(d>0 for d in dif)}
out={'seeds':[x['seed'] for x in A],'rel_l2':summary('rel_l2'),'frmse':summary('frmse'),'interval':'paired Student-t over five independently fine-tuned FNO seeds'}
Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
