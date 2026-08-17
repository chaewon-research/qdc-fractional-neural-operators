"""Train/evaluate the corrected scalar-Darcy FNO for FP32, Q6-weight, and Q6-activation."""
import argparse,json,yaml,torch
from pathlib import Path
from torch.utils.data import DataLoader
from qdc_fno.data import generate_scalar_dataset
from qdc_fno.models import FNO2dQ
from qdc_fno.training import train_fp32
from qdc_fno.evaluate import evaluate_scalar
from qdc_fno.utils import set_seed
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/core_scalar.yaml'); p.add_argument('--seed',type=int); p.add_argument('--out',default='results/scalar_run.json'); p.add_argument('--device',default=None); a=p.parse_args()
c=yaml.safe_load(Path(a.config).read_text()); seed=a.seed if a.seed is not None else c['seeds'][0]; set_seed(seed); dev=torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
# Historical protocol: the base run seed controls both training/model randomness and the synthetic data realization.
data_seeds={'train':seed,'val':seed+10000,'holdout':seed+30000}
N=c['resolution']; lam=float(c['lambda']); d=c['data']; tr=c['training']; m=c['model']
train,_=generate_scalar_dataset(d['train'],N,lam,'train',seed,d); val,_=generate_scalar_dataset(d['val'],N,lam,'val',seed+10000,d); hold,_=generate_scalar_dataset(d['high_contrast'],N,lam,'high_contrast',seed+30000,d)
TL=DataLoader(train,batch_size=tr['batch_size'],shuffle=True); VL=DataLoader(val,batch_size=tr['batch_size']); HL=DataLoader(hold,batch_size=tr['batch_size'])
model=FNO2dQ(3,1,m['width'],m['modes'],m['layers'],m.get('hidden',96)).to(dev); train_fp32(model,TL,VL,tr['epochs'],tr['lr'],tr['weight_decay'],dev,tr.get('grad_clip',1.0))
res={}
for mode,name in [('fp32','fp32'),('weight_q6','q6_weight'),('act_q6','q6_activation')]: res[name]=evaluate_scalar(model,HL,lam,dev,mode).to_dict()
guards=c.get('reproduction_sanity_guards',{})
if guards:
    fp=res['fp32']
    checks={
        'rel_l2': float(fp['rel_l2']) < float(guards['rel_l2_max']),
        'residual': float(fp['residual']) < float(guards['residual_max']),
        'event_rel_l2': float(fp['event_rel_l2']) < float(guards['event_rel_l2_max']),
    }
    res['reproduction_sanity_guards']={
        'thresholds': guards,
        'checks': checks,
        'all_pass': all(checks.values()),
    }
res['seed']=seed; res['model_seed']=seed; res['data_seeds']=data_seeds; res['seed_semantics']='independent training/data-generation run; quantization conditions matched on the same holdout within run'; res['config']=a.config; Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
