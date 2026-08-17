"""End-to-end fresh-scalar QDC evaluation using the verified frozen K=82 operating point.

The release also exposes the calibration selector in qdc_fno.qdc. This focused runner evaluates
its verified selected rank on disjoint high-contrast data and applies residual-energy top-K modes per sample.
"""
import argparse,json,yaml,torch
from pathlib import Path
from torch.utils.data import DataLoader
from qdc_fno.data import generate_scalar_dataset
from qdc_fno.models import FNO2dQ
from qdc_fno.training import train_fp32
from qdc_fno.operators import build_scalar_operator,eigh_nonnegative
from qdc_fno.metrics import rel_l2,event_mask,event_rel_l2,hf_log_error,residual_rel_single,scalar_qdc_objective
from qdc_fno.qdc import residual_defect,rank_residual_energy,qdc_apply
from qdc_fno.utils import set_seed

p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/fresh_scalar_qdc.yaml'); p.add_argument('--seed',type=int)
p.add_argument('--k',type=int,default=82); p.add_argument('--out',default='results/fresh_scalar_qdc.json'); p.add_argument('--device',default=None)
a=p.parse_args(); c=yaml.safe_load(Path(a.config).read_text()); seed=a.seed if a.seed is not None else c['seeds'][0]; set_seed(seed)
dev=torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu')); N=int(c['resolution']); lam=float(c['lambda']); d=c['data']; tr=c['training']; m=c['model']
train,_=generate_scalar_dataset(d['train'],N,lam,'train',seed,d); val,_=generate_scalar_dataset(d['val'],N,lam,'val',seed+10000,d); hold,_=generate_scalar_dataset(d['high_contrast'],N,lam,'high_contrast',seed+30000,d)
TL=DataLoader(train,batch_size=tr['batch_size'],shuffle=True); VL=DataLoader(val,batch_size=tr['batch_size']); HL=DataLoader(hold,batch_size=tr['batch_size'])
model=FNO2dQ(3,1,m['width'],m['modes'],m['layers'],m['hidden']).to(dev); train_fp32(model,TL,VL,tr['epochs'],tr['lr'],tr['weight_decay'],dev,tr['grad_clip'])
rels=[]; evs=[]; hfs=[]; residuals=[]; ptq_rels=[]
model.eval(); model.set_quant('act_q6',6)
with torch.no_grad():
  for x,y in HL:
    pred=model(x.to(dev)).cpu(); y=y.cpu()
    for i in range(x.shape[0]):
      logk=x[i,0].double(); f=x[i,1].double(); s=float(x[i,2,0,0]); vals,vecs=eigh_nonnegative(build_scalar_operator(torch.exp(logk)))
      uq=pred[i,0]; target=y[i,0]; dres=residual_defect(uq,f,vals,vecs,s,lam); idx=rank_residual_energy(dres,vecs,a.k); qc=qdc_apply(uq,f,vals,vecs,s,lam,idx)
      rels.append(float(rel_l2(qc[None,None],target[None,None])[0])); ptq_rels.append(float(rel_l2(uq[None,None],target[None,None])[0]))
      mask=event_mask(logk[None].float()); evs.append(float(event_rel_l2(qc[None],target[None],mask)[0])); hfs.append(float(hf_log_error(qc[None],target[None])[0])); residuals.append(residual_rel_single(qc,f,vals,vecs,s,lam))
mean=lambda z:sum(z)/len(z); rr,ee,hh,res=mean(rels),mean(evs),mean(hfs),mean(residuals)
out={'seed':seed,'selected_k':a.k,'cost_fraction':a.k/(N*N),'ptq_activation_rel_l2':mean(ptq_rels),'qdc':{'rel_l2':rr,'event_rel_l2':ee,'hf_log':hh,'residual':res,'objective':scalar_qdc_objective(rr,ee,res,hh)},'mode_selection':'per-sample residual-energy; calibration freezes rank K'}
Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
