from __future__ import annotations
import torch
from .operators import harmonic


def scalar_matvec_periodic(k: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Matrix-free periodic finite-volume action Lx=-div(k grad x). Axis 0=x, axis 1=y."""
    k = k.double(); n = k.shape[0]; u = x.reshape(n, n).double(); h2 = float(n * n)
    # axis 1 corresponds to y-neighbours, axis 0 to x-neighbours; coefficient is scalar here
    ke = harmonic(k, torch.roll(k, -1, 1)) * h2
    kw = torch.roll(ke, 1, 1)
    ks = harmonic(k, torch.roll(k, -1, 0)) * h2
    kn = torch.roll(ks, 1, 0)
    y = (ke + kw + ks + kn) * u - ke * torch.roll(u, -1, 1) - kw * torch.roll(u, 1, 1) \
        - ks * torch.roll(u, -1, 0) - kn * torch.roll(u, 1, 0)
    return y.reshape(-1)


def lanczos_function_action(matvec, b: torch.Tensor, func, m: int = 160, breakdown_tol: float = 1e-12):
    """Approximate func(A)b for symmetric A via fully reorthogonalized Lanczos projection.

    Returns (approximation, metadata). Negative Ritz values beyond roundoff are rejected.
    """
    b = b.reshape(-1).double(); beta0 = torch.linalg.vector_norm(b)
    if beta0 == 0:
        return b.clone(), {"iterations": 0, "min_ritz": 0.0}
    n = b.numel(); V = torch.empty((n, m), dtype=torch.float64); al, be = [], []
    v = b / beta0; vprev = torch.zeros_like(v); beta_prev = torch.tensor(0., dtype=torch.float64); actual = 0
    for j in range(m):
        V[:, j] = v
        w = matvec(v) - beta_prev * vprev
        a = torch.dot(v, w); w = w - a * v
        coeff = V[:, :j+1].T @ w
        w = w - V[:, :j+1] @ coeff
        beta = torch.linalg.vector_norm(w); al.append(float(a)); actual = j + 1
        if j == m - 1 or beta < breakdown_tol:
            break
        be.append(float(beta)); vprev, v = v, w / beta; beta_prev = beta
    T = torch.diag(torch.tensor(al[:actual], dtype=torch.float64))
    if actual > 1:
        off = torch.tensor(be[:actual-1], dtype=torch.float64); T += torch.diag(off, 1) + torch.diag(off, -1)
    th, S = torch.linalg.eigh(T)
    scale = max(1.0, float(th.abs().max())); neg_tol = 1e-10 * scale
    if float(th.min()) < -neg_tol:
        raise RuntimeError(f"Lanczos projection has materially negative Ritz value {float(th.min()):.3e}")
    th = torch.where(th < 0, torch.zeros_like(th), th)
    e1 = torch.zeros(actual, dtype=torch.float64); e1[0] = 1
    coeff = S @ (func(th) * (S.T @ e1))
    approx = beta0 * (V[:, :actual] @ coeff)
    return approx, {"iterations": actual, "min_ritz": float(th.min()), "max_ritz": float(th.max())}


def solve_fractional_matrixfree(k: torch.Tensor, f: torch.Tensor, s: float, lam: float,
                                start_dim: int = 256, max_dim: int = 512,
                                rel_change_tol: float = 1e-7, dim_step: int = 64,
                                return_double: bool = False):
    """Adaptive matrix-free approximation to (lam I + L^s)^-1 f.

    Krylov dimension starts at start_dim and grows until successive solution estimates change by
    less than rel_change_tol or max_dim is reached. Returns (solution, metadata).
    """
    mv = lambda x: scalar_matvec_periodic(k, x)
    fun = lambda theta: 1.0 / (float(lam) + theta.pow(float(s)))
    prev = None; history = []
    for m in range(start_dim, max_dim + 1, dim_step):
        cur, meta = lanczos_function_action(mv, f.reshape(-1).double(), fun, m=m)
        rel = None
        if prev is not None:
            rel = float(torch.linalg.vector_norm(cur - prev) / (torch.linalg.vector_norm(cur) + 1e-14))
        history.append({"dim": m, "relative_change": rel, **meta})
        if m >= f.numel():
            out=cur.reshape_as(f); return (out if return_double else out.float()), {"converged": True, "history": history, "criterion": "full_dimension_projection"}
        if rel is not None and rel < rel_change_tol:
            out=cur.reshape_as(f); return (out if return_double else out.float()), {"converged": True, "history": history}
        prev = cur
    out=prev.reshape_as(f); return (out if return_double else out.float()), {"converged": False, "history": history}


def apply_fractional_matrixfree(k: torch.Tensor, u: torch.Tensor, s: float, lam: float,
                                start_dim: int = 256, max_dim: int = 512,
                                rel_change_tol: float = 1e-7, dim_step: int = 64):
    """Adaptive matrix-free action (lam I + L^s)u, used for high-resolution residuals."""
    mv=lambda x: scalar_matvec_periodic(k,x)
    fun=lambda theta: float(lam)+theta.pow(float(s))
    prev=None; history=[]
    for m in range(start_dim,max_dim+1,dim_step):
        cur,meta=lanczos_function_action(mv,u.reshape(-1).double(),fun,m=m)
        rel=None if prev is None else float(torch.linalg.vector_norm(cur-prev)/(torch.linalg.vector_norm(cur)+1e-14))
        history.append({"dim":m,"relative_change":rel,**meta})
        if m >= u.numel():
            return cur.reshape_as(u).float(),{"converged":True,"history":history,"criterion":"full_dimension_projection"}
        if rel is not None and rel<rel_change_tol:
            return cur.reshape_as(u).float(),{"converged":True,"history":history}
        prev=cur
    return prev.reshape_as(u).float(),{"converged":False,"history":history}
