from __future__ import annotations
from pathlib import Path
import hashlib, json
import torch
from torch.utils.data import TensorDataset
from .operators import smooth_random_field, make_source
from .matrixfree import solve_fractional_matrixfree


def highres_cache_fingerprint(num, N, lam, split, seed, start_dim=256, max_dim=512,
                              rel_change_tol=1e-7, data_config=None):
    spec = {
        'num': int(num), 'resolution': int(N), 'lambda': float(lam), 'split': str(split), 'seed': int(seed),
        'solver': {'start_dim': int(start_dim), 'max_dim': int(max_dim), 'rel_change_tol': float(rel_change_tol)},
        'data_config': data_config,
    }
    payload=json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()[:12], spec


def _params(split, cfg:dict|None=None):
    if cfg is not None:
        if split=='train': return tuple(cfg['train_alpha']),tuple(cfg['train_contrast']),list(cfg['train_s']),False
        if split in ('val','test_id'): return tuple(cfg['id_alpha']),tuple(cfg['id_contrast']),list(cfg['id_s']),False
        if split=='high_contrast': return tuple(cfg['high_contrast_alpha']),tuple(cfg['high_contrast_contrast']),list(cfg['high_contrast_s']),True
    if split == 'train': return (2.2, 5.0), (0.55, 1.30), [.45, .50, .65, .70, .85, .90, 1.0], False
    if split in ('val', 'test_id'): return (3.5, 4.8), (.65, .90), [.50, .70, .90, 1.0], False
    if split == 'high_contrast': return (2.1, 3.1), (1.0, 1.30), [.45, .65, .85, 1.0], True
    raise ValueError(split)


def generate_highres_scalar_dataset(num, N, lam, split, seed, start_dim=256, max_dim=512,
                                    rel_change_tol=1e-7, cache_path: str | None = None,
                                    data_config:dict|None=None):
    """Generate/cache high-resolution scalar data using the adaptive matrix-free target solver.

    Cached datasets carry a complete scientifically relevant configuration fingerprint. A cache is
    reused only when the requested split, seed, PDE parameter, distribution, and solver settings match.
    """
    _, cache_spec = highres_cache_fingerprint(num, N, lam, split, seed, start_dim, max_dim, rel_change_tol, data_config)
    if cache_path and Path(cache_path).exists():
        obj = torch.load(cache_path, map_location='cpu', weights_only=False)
        if obj.get('cache_spec') != cache_spec:
            raise RuntimeError(f"stale high-resolution cache at {cache_path}: cached configuration does not match requested experiment")
        return TensorDataset(obj['x'], obj['y'])
    ar, cr, svals, ood = _params(split,data_config); g = torch.Generator().manual_seed(seed); xs, ys = [], []
    solver_meta = []
    for _ in range(num):
        alpha = float(torch.empty(()).uniform_(*ar, generator=g)); c = float(torch.empty(()).uniform_(*cr, generator=g))
        sv = float(svals[int(torch.randint(0, len(svals), (1,), generator=g))])
        logk = c * smooth_random_field(N, alpha, g); k = torch.exp(logk); f = make_source(N, g, ood=ood, config=(data_config or {}).get('source'))
        u, meta = solve_fractional_matrixfree(k, f, sv, lam, start_dim=start_dim, max_dim=max_dim,
                                              rel_change_tol=rel_change_tol)
        if not meta['converged']:
            raise RuntimeError(f"target solve did not converge at N={N}; final metadata={meta['history'][-1]}")
        solver_meta.append(meta['history'][-1])
        xs.append(torch.stack([logk, f, torch.full((N, N), sv, dtype=torch.float64)]).float()); ys.append(u.unsqueeze(0).float())
    X, Y = torch.stack(xs), torch.stack(ys)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'x': X, 'y': Y, 'solver_meta': solver_meta, 'cache_spec': cache_spec,
                    'solver': {'start_dim': start_dim, 'max_dim': max_dim, 'rel_change_tol': rel_change_tol}}, cache_path)
    return TensorDataset(X, Y)
