"""Regenerate the matched core-diagnosis figure from the authoritative aggregate table."""
import json,argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
p=argparse.ArgumentParser(); p.add_argument('--reference',default='results/reference/reported_results.json'); p.add_argument('--out',default='results/generated/core_diagnosis.pdf'); a=p.parse_args()
d=json.loads(Path(a.reference).read_text())
core=d['core_scalar_high_contrast']['metric_summary']
keys=[('FP32','fp32'),('Q6 weight','q6_weight'),('Q6 activation','q6_activation')]
vals={label:(core[key]['rel_l2'][0],core[key]['rel_l2'][1],core[key]['residual'][0],core[key]['residual'][1]) for label,key in keys}
labels=list(vals); rel=np.array([vals[k][0] for k in labels]); relsd=np.array([vals[k][1] for k in labels]); res=np.array([vals[k][2] for k in labels]); ressd=np.array([vals[k][3] for k in labels])
fig,axs=plt.subplots(1,2,figsize=(7.2,2.45)); x=np.arange(3)
for ax,y,e,title,yl in [(axs[0],rel,relsd,'Solution error','Rel-$\\ell_2$'),(axs[1],res,ressd,'Physical residual','Relative residual')]:
    bars=ax.bar(x,y,yerr=e,capsize=3,edgecolor='black',linewidth=.8); ax.set_xticks(x,labels,rotation=16,ha='right'); ax.set_title(title,fontsize=10); ax.set_ylabel(yl,fontsize=9); ax.tick_params(labelsize=8); ax.grid(axis='y',alpha=.22,linewidth=.5); ax.set_axisbelow(True); ymax=max(y+e)*1.62; ax.set_ylim(0,ymax)
    for b,v,err in zip(bars,y,e): ax.text(b.get_x()+b.get_width()/2,v+err+ymax*.065,f'{v:.4f}',ha='center',va='bottom',fontsize=7.5,bbox=dict(facecolor='white',edgecolor='none',pad=.5))
axs[0].text(.04,.985,'Q6 activation: +56.5% vs. FP32',transform=axs[0].transAxes,ha='left',va='top',fontsize=7.8); axs[1].text(.04,.985,'Q6 activation: +73.7% vs. FP32',transform=axs[1].transAxes,ha='left',va='top',fontsize=7.8)
fig.tight_layout(w_pad=1.8); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,bbox_inches='tight'); print(out)
