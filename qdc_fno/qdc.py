from __future__ import annotations
from dataclasses import dataclass
import torch


def residual_defect(uq:torch.Tensor,f:torch.Tensor,vals:torch.Tensor,vecs:torch.Tensor,s:float,lam:float)->torch.Tensor:
    flat=uq.reshape(-1).double(); coeff=vecs.T@flat; Au=vecs@((lam+vals.pow(float(s)))*coeff)
    return f.reshape(-1).double()-Au


def qdc_apply(uq:torch.Tensor,f:torch.Tensor,vals:torch.Tensor,vecs:torch.Tensor,s:float,lam:float,indices:torch.Tensor,alpha:float=1.0)->torch.Tensor:
    d=residual_defect(uq,f,vals,vecs,s,lam)
    Q=vecs[:,indices]; vk=vals[indices]
    corr=Q@((Q.T@d)/(lam+vk.pow(float(s))))
    return (uq.reshape(-1).double()+alpha*corr).reshape_as(uq).float()


def rank_residual_energy(d:torch.Tensor,vecs:torch.Tensor,k:int)->torch.Tensor:
    scores=(vecs.T@d).abs().pow(2); return torch.argsort(scores,descending=True)[:k]


def rank_defect_solution_energy(d:torch.Tensor,vals:torch.Tensor,vecs:torch.Tensor,s:float,lam:float,k:int)->torch.Tensor:
    coeff=(vecs.T@d)/(lam+vals.pow(float(s))); return torch.argsort(coeff.abs().pow(2),descending=True)[:k]


def rank_low_eigen(vals:torch.Tensor,k:int)->torch.Tensor:
    return torch.argsort(vals)[:k]


def rank_high_eigen(vals:torch.Tensor,k:int)->torch.Tensor:
    return torch.argsort(vals,descending=True)[:k]


@dataclass
class SelectorRow:
    k:int; objective:float; rel:float; residual:float


def select_residual_margin(rows:list[SelectorRow], best_control_objective:float,best_control_rel:float,best_control_residual:float,eta:float=0.92):
    """Residual-margin selector used by the paper.

    Strong feasible: objective and Rel-L2 beat the best declared mixed/tie control and residual
    is below eta times the best control residual. If none exists, find the smallest weak-feasible
    candidate that beats all three un-margined control guards and promote by one available rank.
    The protocol does not invent a fallback if no weak-feasible candidate exists.
    """
    rows=sorted(rows,key=lambda r:r.k)
    strong=[r for r in rows if r.objective<=best_control_objective and r.rel<=best_control_rel and r.residual<=eta*best_control_residual]
    if strong:
        return strong[0]
    weak=[r for r in rows if r.objective<=best_control_objective and r.rel<=best_control_rel and r.residual<=best_control_residual]
    if not weak:
        raise ValueError('no strong- or weak-feasible QDC rank under the declared calibration guards')
    target=weak[0].k
    larger=[r for r in rows if r.k>target]
    if not larger:
        raise ValueError('weak-feasible QDC rank exists but no larger budget is available for the required one-rank promotion')
    return larger[0]
