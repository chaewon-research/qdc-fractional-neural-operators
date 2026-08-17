from __future__ import annotations
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, lobpcg


def torch_linear_operator(matvec, n:int):
    def mv(x):
        xt=torch.from_numpy(np.asarray(x)).double()
        if xt.ndim==1:
            return matvec(xt).detach().cpu().numpy()
        return torch.stack([matvec(xt[:,j]) for j in range(xt.shape[1])],dim=1).detach().cpu().numpy()
    return LinearOperator((n,n),matvec=mv,matmat=mv,dtype=np.float64)


def lowest_eigenspace(operator:LinearOperator,k:int,seed:int=0,tol:float=1e-4,maxiter:int=100,
                       return_history:bool=False):
    """Lowest LOBPCG eigenspace with the paper settings (no preconditioner)."""
    rng=np.random.default_rng(seed)
    X=rng.standard_normal((operator.shape[0],k))
    out=lobpcg(operator,X,largest=False,tol=tol,maxiter=maxiter,
               retLambdaHistory=return_history,retResidualNormsHistory=return_history)
    if return_history:
        vals,vecs,lambda_hist,resid_hist=out
    else:
        vals,vecs=out; lambda_hist=resid_hist=None
    order=np.argsort(vals)
    vals,vecs=vals[order],vecs[:,order]
    if return_history:
        return vals,vecs,{"lambda_history":_jsonable_history(lambda_hist),
                         "residual_history":_jsonable_history(resid_hist)}
    return vals,vecs


def _jsonable_history(hist):
    if hist is None: return None
    return [np.asarray(x,dtype=float).tolist() for x in hist]


def eigenpair_residuals(operator,vals,vecs):
    AV=operator.matmat(vecs)
    return np.linalg.norm(AV-vecs*vals[None,:],axis=0)


def select_smallest_rank_by_reference_recovery(candidates, fraction: float = 0.95):
    """Apply the paper's validation rank-budget rule.

    Parameters
    ----------
    candidates:
        Iterable of mappings with integer ``k`` and float
        ``mean_reference_rel_l2_recovery`` entries.
    fraction:
        Required fraction of the best validation high-accuracy same-rank-reference
        Rel-L2 recovery.  The paper uses 0.95.

    Returns
    -------
    dict with ``selected_k``, ``best_recovery``, ``threshold``, and sorted candidates.
    """
    rows = [dict(r) for r in candidates]
    if not rows:
        raise ValueError('at least one candidate is required')
    rows.sort(key=lambda r: int(r['k']))
    best = max(float(r['mean_reference_rel_l2_recovery']) for r in rows)
    if best <= 0:
        raise ValueError('best validation reference recovery must be positive')
    threshold = float(fraction) * best
    feasible = [r for r in rows if float(r['mean_reference_rel_l2_recovery']) >= threshold]
    if not feasible:
        raise ValueError('no candidate satisfies the recovery threshold')
    selected = min(feasible, key=lambda r: int(r['k']))
    return {
        'selected_k': int(selected['k']),
        'best_recovery': best,
        'threshold': threshold,
        'fraction': float(fraction),
        'candidates': rows,
    }
