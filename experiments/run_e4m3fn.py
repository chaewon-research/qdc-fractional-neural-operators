"""E4M3FN activation-vs-weight evaluation on a trained FNO checkpoint."""
import argparse,json,torch,yaml
from pathlib import Path
from torch.utils.data import DataLoader
from qdc_fno.data import generate_scalar_dataset
from qdc_fno.models import FNO2dQ
from qdc_fno.evaluate import evaluate_scalar
p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--model-config',required=True); p.add_argument('--seed',type=int,default=101); p.add_argument('--out',default='results/e4m3fn.json'); a=p.parse_args()
c=yaml.safe_load(Path(a.model_config).read_text()); N=c['resolution']; lam=float(c.get('lambda',.15)); m=c['model']; d=c['data']; dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=FNO2dQ(3,1,m['width'],m['modes'],m['layers'],m.get('hidden',96)).to(dev); model.load_state_dict(torch.load(a.checkpoint,map_location=dev)); ds,_=generate_scalar_dataset(d.get('high_contrast',256),N,lam,'high_contrast',a.seed+30000); loader=DataLoader(ds,batch_size=d.get('eval_batch_size',8))
res={'seed':a.seed,'activation':evaluate_scalar(model,loader,lam,dev,'act_fp8').to_dict(),'weight':evaluate_scalar(model,loader,lam,dev,'weight_fp8').to_dict()}; Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
