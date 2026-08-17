import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from qdc_fno.data import generate_scalar_dataset
from qdc_fno.models import FNO2dQ
from qdc_fno.training import train_fp32
from qdc_fno.utils import device_from_arg, save_json, set_seed

p=argparse.ArgumentParser(); p.add_argument('--out',default='results/smoke_execution_only.json'); p.add_argument('--device',default=None); a=p.parse_args()
set_seed(7); dev=device_from_arg(a.device)
tr,_=generate_scalar_dataset(24,8,0.50,'train',7001); va,_=generate_scalar_dataset(8,8,0.50,'val',7002)
tr=DataLoader(tr,batch_size=4,shuffle=True); va=DataLoader(va,batch_size=4)
m=FNO2dQ(width=12,modes=3,layers=2,hidden=16).to(dev); train_fp32(m,tr,va,epochs=2,lr=2e-3,weight_decay=1e-5,device=dev)
x,y=next(iter(va)); x=x.to(dev); y=y.to(dev)
checks={}
for mode in ['fp32','act_q6','weight_q6']:
    m.set_quant(mode,6)
    out=m(x)
    checks[mode]={'shape_ok':list(out.shape)==list(y.shape),'finite':bool(torch.isfinite(out).all().item())}
# Exercise backward path once on FP32.
m.train(); m.set_quant('fp32'); m.zero_grad(set_to_none=True); pred=m(x); loss=torch.mean((pred-y)**2); loss.backward()
grad_finite=all(p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in m.parameters())
res={
  'scientific_result':False,
  'purpose':'API/shape/numerical-sanity execution smoke test only',
  'not_comparable_to_paper':True,
  'epochs':2,
  'resolution':8,
  'device':str(dev),
  'forward_paths':checks,
  'backward_pass':True,
  'gradients_finite':grad_finite,
  'no_nan_inf':all(v['finite'] for v in checks.values()) and grad_finite
}
save_json(res,a.out); print(json.dumps(res,indent=2))
