from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import torch
from .metrics import rel_l2, event_mask, event_rel_l2, hf_log_error, residual_rel_single, anisotropic_objective, scalar_qdc_objective
from .operators import build_scalar_operator, build_anisotropic_operator, eigh_nonnegative

@dataclass
class EvalSummary:
    rel_l2: float
    event_rel_l2: float
    residual: float
    hf_log: float | None = None
    objective: float | None = None
    def to_dict(self): return asdict(self)

def _mean(xs): return float(np.mean(xs)) if xs else float('nan')

@torch.no_grad()
def evaluate_scalar(model, loader, lam: float, device: torch.device, quant_mode: str='fp32') -> EvalSummary:
    model.eval(); model.set_quant(quant_mode, bits=6)
    rels=[]; events=[]; residuals=[]; hfs=[]
    for x,y in loader:
        x=x.to(device); y=y.to(device); pred=model(x)
        rels.extend(rel_l2(pred,y).cpu().tolist())
        masks=event_mask(x[:,0].cpu())
        events.extend(event_rel_l2(pred.cpu()[:,0],y.cpu()[:,0],masks).tolist())
        hfs.extend(hf_log_error(pred.cpu()[:,0],y.cpu()[:,0]).tolist())
        for i in range(x.shape[0]):
            logk=x[i,0].cpu().double(); f=x[i,1].cpu().double(); s=float(x[i,2,0,0].item())
            vals,vecs=eigh_nonnegative(build_scalar_operator(torch.exp(logk)))
            residuals.append(residual_rel_single(pred[i,0].cpu(),f,vals,vecs,s,lam))
    r,e,res,h=_mean(rels),_mean(events),_mean(residuals),_mean(hfs)
    return EvalSummary(r,e,res,h,scalar_qdc_objective(r,e,res,h))

@torch.no_grad()
def evaluate_anisotropic(model, loader, lam: float, device: torch.device, quant_mode: str='fp32') -> EvalSummary:
    model.eval(); model.set_quant(quant_mode, bits=6)
    rels=[]; events=[]; residuals=[]
    for x,y in loader:
        x=x.to(device); y=y.to(device); pred=model(x)
        rels.extend(rel_l2(pred,y).cpu().tolist())
        avglog=0.5*(x[:,0].cpu()+x[:,1].cpu()); masks=event_mask(avglog)
        events.extend(event_rel_l2(pred.cpu()[:,0],y.cpu()[:,0],masks).tolist())
        for i in range(x.shape[0]):
            logkx=x[i,0].cpu().double(); logky=x[i,1].cpu().double(); f=x[i,2].cpu().double(); s=float(x[i,3,0,0].item())
            vals,vecs=eigh_nonnegative(build_anisotropic_operator(torch.exp(logkx),torch.exp(logky)))
            residuals.append(residual_rel_single(pred[i,0].cpu(),f,vals,vecs,s,lam))
    r,e,res=_mean(rels),_mean(events),_mean(residuals)
    return EvalSummary(r,e,res,None,anisotropic_objective(r,e,res))

@torch.no_grad()
def evaluate_scalar_highres(model, loader, lam: float, device: torch.device, quant_mode: str='fp32',
                            start_dim: int=256, max_dim: int=512, rel_change_tol: float=1e-7) -> EvalSummary:
    """High-resolution scalar evaluation without constructing a dense N^2-by-N^2 operator."""
    from .matrixfree import apply_fractional_matrixfree
    model.eval(); model.set_quant(quant_mode,bits=6)
    rels=[]; events=[]; residuals=[]; hfs=[]
    for x,y in loader:
        x=x.to(device); y=y.to(device); pred=model(x)
        rels.extend(rel_l2(pred,y).cpu().tolist())
        masks=event_mask(x[:,0].cpu()); events.extend(event_rel_l2(pred.cpu()[:,0],y.cpu()[:,0],masks).tolist())
        hfs.extend(hf_log_error(pred.cpu()[:,0],y.cpu()[:,0]).tolist())
        for i in range(x.shape[0]):
            k=torch.exp(x[i,0].cpu().double()); f=x[i,1].cpu().double(); s=float(x[i,2,0,0].item()); u=pred[i,0].cpu().double()
            Au,meta=apply_fractional_matrixfree(k,u,s,lam,start_dim=start_dim,max_dim=max_dim,rel_change_tol=rel_change_tol)
            if not meta['converged']:
                raise RuntimeError(f'high-resolution residual action failed to converge: {meta["history"][-1]}')
            r=f-Au.double(); residuals.append(float(torch.linalg.vector_norm(r)/(torch.linalg.vector_norm(f)+1e-8)))
    r,e,res,h=_mean(rels),_mean(events),_mean(residuals),_mean(hfs)
    return EvalSummary(r,e,res,h,scalar_qdc_objective(r,e,res,h))
