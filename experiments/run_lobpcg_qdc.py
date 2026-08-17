"""Matrix-free lowest-eigenspace QDC diagnostic on saved FNO predictions.

The input artifact can represent either the validation split (for rank selection) or the
disjoint high-contrast holdout (for frozen-rank evaluation).  At 32x32 the same-rank
high-accuracy reference is exact dense; at 64/128 it is a tighter same-rank LOBPCG solve.
Operational runs use largest=False, tol=1e-4, maxiter=100, and no preconditioner.

For the documented 32x32 K=250 diagnostic, SciPy would use its dense fallback because
5K > n.  We record that case explicitly and use the exact dense subspace rather than
mislabeling it as an iterative LOBPCG result.
"""
import argparse, json, numpy as np, torch
from pathlib import Path
from qdc_fno.matrixfree import scalar_matvec_periodic, apply_fractional_matrixfree
from qdc_fno.lobpcg import torch_linear_operator, lowest_eigenspace, eigenpair_residuals
from qdc_fno.operators import build_scalar_operator, eigh_nonnegative

p = argparse.ArgumentParser()
p.add_argument('--artifact', required=True)
p.add_argument('--k', type=int, required=True)
p.add_argument('--max-samples', type=int, default=None)
p.add_argument('--out', required=True)
a = p.parse_args()

z = torch.load(a.artifact, map_location='cpu')
x, y, pa, pf = z['x'], z['y'], z['pred_act'], z['pred_fp32']
N = int(z['resolution'])
lam = float(z['lambda'])
split = str(z.get('split', 'unspecified'))
n = N * N
M = len(x) if a.max_samples is None else min(len(x), a.max_samples)
rows = []

for i in range(M):
    logk = x[i, 0].double()
    k = torch.exp(logk)
    f = x[i, 1].double()
    s = float(x[i, 2, 0, 0])
    target = y[i, 0].double()
    uact = pa[i, 0].double()
    ufp = pf[i, 0].double()
    op = torch_linear_operator(lambda v: scalar_matvec_periodic(k, v), n)

    # Exact eigensystem is affordable at N<=32 and doubles as the same-rank reference.
    exact_vals = exact_Q = None
    if N <= 32:
        exact_vals, exact_Q = eigh_nonnegative(build_scalar_operator(k))

    dense_fallback = (N <= 32 and 5 * a.k > n)
    if dense_fallback:
        vals_t = exact_vals[:a.k].double()
        Q_t = exact_Q[:, :a.k].double()
        hist = None
        operational_kind = 'dense_fallback_5k_gt_n'
        op_eigen_resid = 0.0
    else:
        vals, Q, hist = lowest_eigenspace(op, a.k, seed=1000 + i, tol=1e-4, maxiter=100, return_history=True)
        vals_t = torch.from_numpy(vals).double()
        Q_t = torch.from_numpy(Q).double()
        operational_kind = 'lobpcg_iterative'
        op_eigen_resid = float(eigenpair_residuals(op, vals, Q).max())

    Au, ameta = apply_fractional_matrixfree(k, uact, s, lam)
    defect = f.reshape(-1) - Au.reshape(-1).double()
    corr = Q_t @ ((Q_t.T @ defect) / (lam + vals_t.pow(s)))
    uq = (uact.reshape(-1) + corr).reshape(N, N)

    if N <= 32:
        rv = exact_vals[:a.k].double()
        rQ = exact_Q[:, :a.k].double()
        reference_kind = 'exact_dense_same_rank'
    else:
        rv_np, rQ_np, _ = lowest_eigenspace(op, a.k, seed=2000 + i, tol=1e-8, maxiter=300, return_history=True)
        rv = torch.from_numpy(rv_np).double()
        rQ = torch.from_numpy(rQ_np).double()
        reference_kind = 'tighter_lobpcg_same_rank_tol_1e-8_maxiter_300'

    rcorr = rQ @ ((rQ.T @ defect) / (lam + rv.pow(s)))
    uref = (uact.reshape(-1) + rcorr).reshape(N, N)

    def rel(u):
        return float(torch.linalg.vector_norm(u - target) / (torch.linalg.vector_norm(target) + 1e-14))

    def phys_res(u):
        Au2, _ = apply_fractional_matrixfree(k, u, s, lam)
        return float(torch.linalg.vector_norm(f.reshape(N, N) - Au2.double()) / (torch.linalg.vector_norm(f) + 1e-14))

    eact, efp, eop, eref = rel(uact), rel(ufp), rel(uq), rel(uref)
    ract, rfp, rop, rref = phys_res(uact), phys_res(ufp), phys_res(uq), phys_res(uref)
    ref_recovery = eact - eref
    op_recovery = eact - eop
    rows.append({
        'sample': i,
        'activation_rel_l2': eact,
        'fp32_rel_l2': efp,
        'lobpcg_qdc_rel_l2': eop,
        'reference_qdc_rel_l2': eref,
        'reference_rel_l2_recovery': ref_recovery,
        'operational_rel_l2_recovery': op_recovery,
        'activation_residual': ract,
        'fp32_residual': rfp,
        'lobpcg_qdc_residual': rop,
        'reference_qdc_residual': rref,
        'gain_retention': op_recovery / (ref_recovery + 1e-14),
        'fp32_gap_closure': op_recovery / (eact - efp + 1e-14),
        'residual_fp32_gap_closure': (ract - rop) / (ract - rfp + 1e-14),
        'max_eigenpair_residual': op_eigen_resid,
        'operational_kind': operational_kind,
        'reference_kind': reference_kind,
        'operator_action': ameta,
        'lobpcg_history': hist,
    })

mean = lambda key: float(np.mean([r[key] for r in rows]))
out = {
    'resolution': N,
    'split': split,
    'artifact': str(Path(a.artifact)),
    'k': a.k,
    'samples': M,
    'operational': {'largest': False, 'tol': 1e-4, 'maxiter': 100, 'preconditioner': None},
    'reference': 'exact dense at N<=32; tighter same-rank LOBPCG otherwise',
    'dense_fallback': bool(dense_fallback),
    'rows': rows,
    'mean_reference_rel_l2_recovery': mean('reference_rel_l2_recovery'),
    'mean_operational_rel_l2_recovery': mean('operational_rel_l2_recovery'),
    'mean_gain_retention': mean('gain_retention'),
    'mean_fp32_gap_closure': mean('fp32_gap_closure'),
    'mean_residual_fp32_gap_closure': mean('residual_fp32_gap_closure'),
}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(out, indent=2))
print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=2))
