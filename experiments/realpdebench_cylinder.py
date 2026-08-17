"""Pinned RealPDEBench Cylinder Q6 activation-vs-weight evaluation.

Use one independently fine-tuned FNO checkpoint per seed. The loader/model/evaluator come from the
pinned upstream RealPDEBench revision; only SpectralConv3d forward passes are patched for Q6.
"""
import argparse,json,hashlib,torch,yaml
from pathlib import Path
from qdc_fno.realpdebench_patch import patch_realpdebench_fno

p=argparse.ArgumentParser(); p.add_argument('--dataset-root',required=True); p.add_argument('--checkpoint',required=True)
p.add_argument('--mode',choices=['activation','weight'],required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--test-mode',default='all',choices=['seen','in_dist','out_dist','unseen','all']); p.add_argument('--asset-lock',default='external/realpdebench_assets.lock.json'); p.add_argument('--upstream-root',default='external/RealPDEBench'); p.add_argument('--out',required=True); a=p.parse_args()
try:
    from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
    from realpdebench.data.data_normalizer import GaussianNormalizer
    from realpdebench.model.load_model import load_model
    from realpdebench.utils.metrics import eval_metrics
except Exception as e:
    raise SystemExit('Install the pinned upstream RealPDEBench revision first: '+repr(e))

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(a.seed)
lock_path=Path(a.asset_lock)
if not lock_path.exists(): raise SystemExit('Missing immutable asset lock: external/realpdebench_assets.lock.json')
lock=json.loads(lock_path.read_text())
common=dict(hf_auto_download=True,hf_repo_id=lock['dataset_repository'],hf_endpoint=None,hf_revision=lock['dataset_revision'])
test=CylinderHFDataset(dataset_name='cylinder',dataset_root=a.dataset_root,mode='test',N_autoregressive=10,dataset_type='real',test_mode=a.test_mode,**common)
train=CylinderHFDataset(dataset_name='cylinder',dataset_root=a.dataset_root,mode='train',dataset_type='real',mask_prob=0.1,**common)
normds=CylinderHFDataset(dataset_name='cylinder',dataset_root=a.dataset_root,mode='train',dataset_type='numerical',**common)
normalizer=GaussianNormalizer(normds,device=device)
cfg_path=Path(a.upstream_root)/'realpdebench/configs/cylinder/fno.yaml'
if not cfg_path.exists(): raise SystemExit(f'Missing pinned upstream FNO config: {cfg_path}')
cfg=yaml.safe_load(cfg_path.read_text())
kwargs=dict(model_name=cfg['model_name'],modes1=int(cfg['modes1']),modes2=int(cfg['modes2']),modes3=int(cfg['modes3']),n_layers=int(cfg['n_layers']),width=int(cfg['width']),dataset_root=a.dataset_root,dataset_name='cylinder',N_autoregressive=int(cfg['N_autoregressive']))
model=load_model(train,device=device,**kwargs); model.load_checkpoint(a.checkpoint,device); patch_realpdebench_fno(model,a.mode,bits=6); model.eval()
loader=torch.utils.data.DataLoader(test,batch_size=64,shuffle=False,num_workers=0)
preds=[]; targets=[]
with torch.no_grad():
    for inp,tgt in loader:
        inp,tgt=inp.to(device),tgt.to(device); inp_n,tgt_n=normalizer.preprocess(inp,tgt); cur=inp_n; rollout=[]
        for _ in range(10):
            pr_n=model(cur); _,pr=normalizer.postprocess(cur,pr_n); pr_n,_=normalizer.preprocess(pr,tgt); rollout.append(pr); cur=pr_n
        pred=torch.cat(rollout,dim=1); preds.append(pred.cpu()); targets.append(tgt.cpu())
pred=torch.cat(preds); tgt=torch.cat(targets)
# Pinned upstream API: eval_metrics(pred, target, c, batch_size) -> 13-tuple.
metrics=eval_metrics(pred,tgt,2,batch_size=64)
def scalar(v):
    if torch.is_tensor(v): return float(v.detach().cpu().mean())
    try: return float(v)
    except Exception: return str(v)
ckpt=Path(a.checkpoint)
def sha256_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
res={'mode':a.mode,'seed':a.seed,'test_mode':a.test_mode,'checkpoint':ckpt.name,'checkpoint_sha256':sha256_file(ckpt),
     'metrics':{'rel_l2':scalar(metrics[2]),'frmse':scalar(metrics[5])},
     'upstream_commit':lock['source_git_commit'],'dataset_revision':lock['dataset_revision'],'model_revision':lock['model_revision']}
Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
