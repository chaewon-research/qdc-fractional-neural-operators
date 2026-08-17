from __future__ import annotations
import torch
from torch.utils.data import TensorDataset
from .operators import smooth_random_field,make_source,build_scalar_operator,eigh_nonnegative,solve_fractional_from_eigh


def _params(split:str, cfg:dict|None=None):
    if cfg is not None:
        if split=='train': return tuple(cfg['train_alpha']),tuple(cfg['train_contrast']),list(cfg['train_s']),False
        if split in ('val','test_id'): return tuple(cfg['id_alpha']),tuple(cfg['id_contrast']),list(cfg['id_s']),False
        if split=='high_contrast': return tuple(cfg['high_contrast_alpha']),tuple(cfg['high_contrast_contrast']),list(cfg['high_contrast_s']),True
        if split=='extreme': return tuple(cfg['extreme_alpha']),tuple(cfg['extreme_contrast']),list(cfg['extreme_s']),True
    if split=='train': return (2.2,5.0),(0.55,1.30),[0.45,0.50,0.65,0.70,0.85,0.90,1.00],False
    if split in ('val','test_id'): return (3.5,4.8),(0.65,0.90),[0.50,0.70,0.90,1.00],False
    if split=='high_contrast': return (2.1,3.1),(1.00,1.30),[0.45,0.65,0.85,1.00],True
    if split=='extreme': return (1.9,2.4),(1.35,1.55),[0.65,0.85,1.00],True
    raise ValueError(split)


def generate_scalar_dataset(n:int,N:int,lam:float,split:str,seed:int,data_config:dict|None=None):
    """Generate the scalar family; when provided, data_config is the source of split distributions."""
    ar,cr,svals,ood=_params(split,data_config); g=torch.Generator().manual_seed(seed); xs=[]; ys=[]; meta=[]
    for _ in range(n):
        alpha=float(torch.empty(()).uniform_(*ar,generator=g)); contrast=float(torch.empty(()).uniform_(*cr,generator=g)); s=float(svals[int(torch.randint(0,len(svals),(1,),generator=g))])
        base=smooth_random_field(N,alpha,g); logk=contrast*base; k=torch.exp(logk); f=make_source(N,g,ood=ood,config=(data_config or {}).get('source')); L=build_scalar_operator(k); vals,vecs=eigh_nonnegative(L); u=solve_fractional_from_eigh(vals,vecs,f,s,lam)
        sgrid=torch.full((N,N),s,dtype=torch.float64); x=torch.stack([logk,f,sgrid],dim=0).float(); xs.append(x); ys.append(u.unsqueeze(0)); meta.append(dict(alpha=alpha,contrast=contrast,s=s))
    return TensorDataset(torch.stack(xs),torch.stack(ys)),meta
