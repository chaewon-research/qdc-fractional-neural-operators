from __future__ import annotations
import math
import torch

EPS = 1e-8


def harmonic(a: torch.Tensor | float, b: torch.Tensor | float, eps: float = 1e-14):
    return 2.0 * a * b / (a + b + eps)


def smooth_random_field(n: int, alpha: float, generator: torch.Generator, dtype=torch.float64) -> torch.Tensor:
    noise = torch.randn((n, n), generator=generator, dtype=dtype)
    ft = torch.fft.rfft2(noise)
    kx = torch.fft.fftfreq(n, d=1.0 / n, dtype=dtype).view(n, 1)
    ky = torch.fft.rfftfreq(n, d=1.0 / n, dtype=dtype).view(1, n // 2 + 1)
    k2 = kx * kx + ky * ky
    filt = 1.0 / (1.0 + k2) ** (alpha / 2.0)
    field = torch.fft.irfft2(ft * filt, s=(n, n))
    field = (field - field.mean()) / (field.std() + EPS)
    return field


def make_source(n: int, generator: torch.Generator, ood: bool = False, config: dict | None = None) -> torch.Tensor:
    """Periodic forcing: smooth Gaussian sources plus low-frequency Fourier structure.

    The high-contrast/OOD split deliberately increases both coefficient contrast and forcing
    complexity. Exact source ranges can be supplied through the experiment configuration.
    """
    xs = torch.arange(n, dtype=torch.float64) / float(n)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    src = torch.zeros((n, n), dtype=torch.float64)
    default = {
        'train': {'blobs': [1, 4], 'amp': [0.60, 1.70], 'sigma': [0.055, 0.160], 'negative_prob': 0.30,
                  'lowfreq_terms': [1, 3], 'lowfreq_amp': [0.12, 0.32], 'lowfreq_kmax': 3},
        'ood':   {'blobs': [2, 6], 'amp': [0.70, 2.00], 'sigma': [0.035, 0.130], 'negative_prob': 0.30,
                  'lowfreq_terms': [1, 4], 'lowfreq_amp': [0.15, 0.42], 'lowfreq_kmax': 4},
    }
    key='ood' if ood else 'train'; cfg=(config or {}).get(key, default[key]) if config else default[key]
    blo,bhi=map(int,cfg['blobs']); num=int(torch.randint(blo,bhi+1,(1,),generator=generator).item())
    amin,amax=map(float,cfg['amp']); smin,smax=map(float,cfg['sigma']); pneg=float(cfg.get('negative_prob',0.30))
    for _ in range(num):
        cx=float(torch.rand((),generator=generator)); cy=float(torch.rand((),generator=generator))
        amp=amin+(amax-amin)*float(torch.rand((),generator=generator));
        if float(torch.rand((),generator=generator))<pneg: amp=-amp
        sig=smin+(smax-smin)*float(torch.rand((),generator=generator))
        dx=torch.minimum((xx-cx).abs(),1-(xx-cx).abs()); dy=torch.minimum((yy-cy).abs(),1-(yy-cy).abs())
        src += amp*torch.exp(-(dx*dx+dy*dy)/(2.0*sig*sig))
    llo,lhi=map(int,cfg.get('lowfreq_terms',[1,3])); nlf=int(torch.randint(llo,lhi+1,(1,),generator=generator).item())
    lfmin,lfmax=map(float,cfg.get('lowfreq_amp',[0.12,0.32])); kmax=int(cfg.get('lowfreq_kmax',3))
    for _ in range(nlf):
        kx=int(torch.randint(1,kmax+1,(1,),generator=generator)); ky=int(torch.randint(1,kmax+1,(1,),generator=generator))
        phase=2*math.pi*float(torch.rand((),generator=generator)); amp=lfmin+(lfmax-lfmin)*float(torch.rand((),generator=generator))
        src += amp*torch.sin(2*math.pi*(kx*xx+ky*yy)+phase)
    src -= src.mean()
    return src / (src.std() + EPS)


def build_scalar_operator(k: torch.Tensor) -> torch.Tensor:
    """Dense periodic finite-volume L=-div(k grad), harmonic edge means."""
    k = k.detach().cpu().double()
    n = k.shape[0]
    m = n*n
    L = torch.zeros((m, m), dtype=torch.float64)
    h2inv = float(n*n)
    def idx(i,j): return (i % n)*n + (j % n)
    for i in range(n):
        for j in range(n):
            p = idx(i,j)
            nbrs = [
                (i+1,j,harmonic(k[i,j], k[(i+1)%n,j])),
                (i-1,j,harmonic(k[i,j], k[(i-1)%n,j])),
                (i,j+1,harmonic(k[i,j], k[i,(j+1)%n])),
                (i,j-1,harmonic(k[i,j], k[i,(j-1)%n])),
            ]
            diag = 0.0
            for ni,nj,ke in nbrs:
                q = idx(ni,nj)
                val = float(ke)*h2inv
                L[p,q] -= val
                diag += val
            L[p,p] += diag
    return 0.5*(L + L.T)


def build_anisotropic_operator(kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
    kx, ky = kx.detach().cpu().double(), ky.detach().cpu().double()
    n = kx.shape[0]; m=n*n; L=torch.zeros((m,m),dtype=torch.float64); h2inv=float(n*n)
    def idx(i,j): return (i % n)*n + (j % n)
    for i in range(n):
        for j in range(n):
            p=idx(i,j); diag=0.0
            edges=[
                (i+1,j,harmonic(kx[i,j],kx[(i+1)%n,j])),
                (i-1,j,harmonic(kx[i,j],kx[(i-1)%n,j])),
                (i,j+1,harmonic(ky[i,j],ky[i,(j+1)%n])),
                (i,j-1,harmonic(ky[i,j],ky[i,(j-1)%n])),
            ]
            for ni,nj,ke in edges:
                q=idx(ni,nj); val=float(ke)*h2inv; L[p,q]-=val; diag+=val
            L[p,p]+=diag
    return 0.5*(L+L.T)


def eigh_nonnegative(L: torch.Tensor, rtol: float = 1e-10):
    """Symmetric eigendecomposition with clipping restricted to roundoff-scale negatives."""
    vals, vecs = torch.linalg.eigh(L.double())
    scale = max(1.0, float(vals.abs().max()))
    tol = rtol * scale
    vmin = float(vals.min())
    if vmin < -tol:
        raise RuntimeError(f"operator has a materially negative eigenvalue {vmin:.3e} < {-tol:.3e}")
    vals = torch.where(vals < 0, torch.zeros_like(vals), vals)
    return vals, vecs


def fractional_matrix_from_eigh(vals: torch.Tensor, vecs: torch.Tensor, s: float, lam: float) -> torch.Tensor:
    diag = lam + vals.pow(float(s))
    return (vecs * diag.unsqueeze(0)) @ vecs.T


def solve_fractional_from_eigh(vals: torch.Tensor, vecs: torch.Tensor, f: torch.Tensor, s: float, lam: float) -> torch.Tensor:
    coeff = vecs.T @ f.reshape(-1).double()
    u = vecs @ (coeff / (lam + vals.pow(float(s))))
    return u.reshape(f.shape).float()


def apply_fractional_from_eigh(vals: torch.Tensor, vecs: torch.Tensor, u: torch.Tensor, s: float, lam: float) -> torch.Tensor:
    flat = u.reshape(-1).double()
    coeff = vecs.T @ flat
    y = vecs @ ((lam + vals.pow(float(s))) * coeff)
    return y.reshape(u.shape).float()
