from __future__ import annotations
import torch
from .operators import apply_fractional_from_eigh
EPS=1e-8


def rel_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    p=pred.reshape(pred.shape[0],-1); t=target.reshape(target.shape[0],-1)
    return torch.linalg.vector_norm(p-t,dim=1)/(torch.linalg.vector_norm(t,dim=1)+eps)


def event_mask(logk: torch.Tensor, quantile: float = 0.80) -> torch.Tensor:
    gx=torch.roll(logk,-1,-1)-torch.roll(logk,1,-1)
    gy=torch.roll(logk,-1,-2)-torch.roll(logk,1,-2)
    g=torch.sqrt(gx*gx+gy*gy)
    flat=g.reshape(g.shape[:-2]+(-1,))
    thr=torch.quantile(flat,quantile,dim=-1,keepdim=True)
    return g>=thr.reshape(g.shape[:-2]+(1,1))


def event_rel_l2(pred: torch.Tensor,target: torch.Tensor,mask: torch.Tensor,eps:float=EPS)->torch.Tensor:
    m=mask.to(pred.dtype); p=(m*(pred-target)).reshape(pred.shape[0],-1); t=(m*target).reshape(target.shape[0],-1)
    return torch.linalg.vector_norm(p,dim=1)/(torch.linalg.vector_norm(t,dim=1)+eps)


def residual_rel_single(pred: torch.Tensor,f:torch.Tensor,vals:torch.Tensor,vecs:torch.Tensor,s:float,lam:float,eps:float=EPS)->float:
    r=f-apply_fractional_from_eigh(vals,vecs,pred,s,lam)
    return float(torch.linalg.vector_norm(r)/(torch.linalg.vector_norm(f)+eps))


def hf_log_error(pred:torch.Tensor,target:torch.Tensor,radial_quantile:float=0.65)->torch.Tensor:
    # batch-first, 2D spatial
    fp=torch.fft.fft2(pred,dim=(-2,-1)); ft=torch.fft.fft2(target,dim=(-2,-1))
    n1,n2=pred.shape[-2:]
    k1=torch.fft.fftfreq(n1,device=pred.device); k2=torch.fft.fftfreq(n2,device=pred.device)
    yy,xx=torch.meshgrid(k1,k2,indexing='ij'); rad=torch.sqrt(xx*xx+yy*yy)
    thr=torch.quantile(rad.flatten(),radial_quantile); mask=rad>=thr
    d=(torch.log1p(fp.abs())-torch.log1p(ft.abs())).abs()[...,mask]
    return d.mean(dim=-1)


def anisotropic_objective(rel,event,resid):
    return event + 0.50*rel + 0.25*resid


def scalar_qdc_objective(rel,event,resid,hf):
    return rel + event + 0.10*resid + 0.25*hf


def mask_calibration_objective(event,hf,rel,resid_proxy):
    return event + 0.50*hf + 0.50*rel + 0.20*resid_proxy
