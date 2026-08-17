"""Train/evaluate anisotropic fractional FNO controls."""
import argparse,json,yaml,torch
from pathlib import Path
from torch.utils.data import DataLoader
from qdc_fno.anisotropic_data import generate_anisotropic_dataset
from qdc_fno.models import FNO2dQ
from qdc_fno.training import train_fp32
from qdc_fno.evaluate import evaluate_anisotropic
from qdc_fno.utils import set_seed
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/anisotropic_n16.yaml'); p.add_argument('--seed',type=int); p.add_argument('--out',default='results/aniso_run.json'); p.add_argument('--device',default=None); a=p.parse_args()
c=yaml.safe_load(Path(a.config).read_text()); seed=a.seed if a.seed is not None else c['seeds'][0]; set_seed(seed); dev=torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
N=c['resolution']; lam=float(c['lambda']); d=c['data']; tr=c['training']; m=c['model']
train=generate_anisotropic_dataset(d['train'],N,lam,'train',seed); val=generate_anisotropic_dataset(d['val'],N,lam,'val',seed+10000); hold=generate_anisotropic_dataset(d['high_contrast'],N,lam,'high_contrast',seed+30000)
TL=DataLoader(train,batch_size=tr['batch_size'],shuffle=True); VL=DataLoader(val,batch_size=tr['batch_size']); HL=DataLoader(hold,batch_size=tr['batch_size'])
model=FNO2dQ(4,1,m['width'],m['modes'],m['layers'],m.get('hidden',96)).to(dev); train_fp32(model,TL,VL,tr['epochs'],tr['lr'],tr['weight_decay'],dev,tr.get('grad_clip',1.0))
res={}
for mode,name in [('fp32','fp32'),('weight_q6','q6_weight'),('act_q6','q6_activation')]: res[name]=evaluate_anisotropic(model,HL,lam,dev,mode).to_dict()
res['seed']=seed; res['config']=a.config; Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
