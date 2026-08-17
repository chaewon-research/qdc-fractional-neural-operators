"""Run the 32/64/128 activation-vs-weight replication with adaptive matrix-free targets.

Besides the reported holdout metrics, this script saves disjoint validation and holdout
prediction artifacts.  The validation artifact is used by the LOBPCG rank-selection
reproduction; the selected rank is then frozen and evaluated on the holdout artifact.
"""
import argparse, json, yaml, torch
from pathlib import Path
from torch.utils.data import DataLoader
from qdc_fno.highres_data import generate_highres_scalar_dataset, highres_cache_fingerprint
from qdc_fno.models import FNO2dQ
from qdc_fno.training import train_fp32
from qdc_fno.evaluate import evaluate_scalar_highres
from qdc_fno.utils import set_seed

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/higher_resolution.yaml')
p.add_argument('--resolution', type=int, choices=[32, 64, 128], required=True)
p.add_argument('--seed', type=int)
p.add_argument('--out', required=True)
p.add_argument('--cache-dir', default='cache/highres')
p.add_argument('--device', default=None)
a = p.parse_args()

c = yaml.safe_load(Path(a.config).read_text())
st = next(x for x in c['settings'] if x['resolution'] == a.resolution)
seed = a.seed if a.seed is not None else c['seeds'][0]
set_seed(seed)
dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
N = a.resolution
lam = float(c['lambda'])
d = st['data']
tr = st['training']
m = st['model']
sol = c['target_solver']
dist = c['distribution']


def ds(split, count, offset):
    fp, _ = highres_cache_fingerprint(
        count, N, lam, split, seed + offset,
        start_dim=sol['start_dim'], max_dim=sol['max_dim'],
        rel_change_tol=sol['relative_change_tolerance'], data_config=dist,
    )
    cache = str(Path(a.cache_dir) / f'N{N}_{split}_seed{seed+offset}_{fp}.pt')
    return generate_highres_scalar_dataset(
        count, N, lam, split, seed + offset,
        start_dim=sol['start_dim'], max_dim=sol['max_dim'],
        rel_change_tol=sol['relative_change_tolerance'], cache_path=cache,
        data_config=dist,
    )


train = ds('train', d['train'], 0)
val = ds('val', d['val'], 10000)
hold = ds('high_contrast', d['high_contrast'], 30000)
TL = DataLoader(train, batch_size=tr['batch_size'], shuffle=True)
VL = DataLoader(val, batch_size=tr['batch_size'])
HL = DataLoader(hold, batch_size=tr['batch_size'])

model = FNO2dQ(3, 1, m['width'], m['modes'], m['layers'], m['hidden']).to(dev)
train_fp32(model, TL, VL, tr['epochs'], tr['lr'], tr['weight_decay'], dev, 1.0)

res = {}
for mode, name in [('fp32', 'fp32'), ('weight_q6', 'q6_weight'), ('act_q6', 'q6_activation')]:
    res[name] = evaluate_scalar_highres(
        model, HL, lam, dev, mode,
        start_dim=sol['start_dim'], max_dim=sol['max_dim'],
        rel_change_tol=sol['relative_change_tolerance'],
    ).to_dict()
res.update(seed=seed, resolution=N, target_solver=sol)

# Save the trained checkpoint and disjoint prediction artifacts used by the matrix-free QDC diagnostic.
base = Path(a.out)
base.parent.mkdir(parents=True, exist_ok=True)
ckpt = base.with_suffix('.model.pt')
torch.save({'state_dict': model.state_dict(), 'resolution': N, 'seed': seed, 'config': c}, ckpt)


def infer(loader, mode):
    model.eval()
    model.set_quant(mode, 6)
    xs, ys, ps = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            ps.append(model(xb.to(dev)).cpu())
            xs.append(xb.cpu())
            ys.append(yb.cpu())
    return torch.cat(xs), torch.cat(ys), torch.cat(ps)


def save_qdc_artifact(loader, split_name, path):
    x, y, pact = infer(loader, 'act_q6')
    _, _, pfp = infer(loader, 'fp32')
    payload = {
        'x': x,
        'y': y,
        'pred_act': pact,
        'pred_fp32': pfp,
        'resolution': N,
        'seed': seed,
        'lambda': lam,
        'split': split_name,
        'data_seed': seed + (10000 if split_name == 'validation' else 30000),
    }
    torch.save(payload, path)
    return str(path)

val_art = Path(str(base.with_suffix('')) + '.validation_qdc.pt')
hold_art = base.with_suffix('.qdc.pt')
res['checkpoint'] = str(ckpt)
res['validation_qdc_artifact'] = save_qdc_artifact(VL, 'validation', val_art)
res['qdc_artifact'] = save_qdc_artifact(HL, 'holdout_high_contrast', hold_art)

Path(a.out).write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
