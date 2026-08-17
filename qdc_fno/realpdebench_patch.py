"""RealPDEBench FNO interoperability patch for object-level Q6 interventions.

This file follows the spectral-block structure of RealPDEBench ``SpectralConv3d`` at the pinned
upstream revision so that activation-only and weight-only QDQ can be inserted without vendoring
the benchmark. RealPDEBench is CC BY-NC 4.0; this adapted interoperability file is distributed
under those upstream terms. See ``THIRD_PARTY_NOTICES.md``. The benchmark loader, checkpoint,
normalizer, rollout, and evaluator remain upstream implementations.
"""
from __future__ import annotations
import types
import torch
from .quantization import symmetric_qdq_complex_independent


def _mul(x,w): return torch.einsum('bixyz,ioxyz->boxyz',x,w)


def _forward(self,x,mode,bits):
    B=x.shape[0]; xf=torch.fft.rfftn(x,dim=[-3,-2,-1]); out=torch.zeros(B,self.out_channels,x.size(-3),x.size(-2),x.size(-1)//2+1,dtype=torch.cfloat,device=x.device)
    blocks=[
      (slice(None),slice(None),slice(0,self.modes1),slice(0,self.modes2),slice(0,self.modes3),self.weights1),
      (slice(None),slice(None),slice(-self.modes1,None),slice(0,self.modes2),slice(0,self.modes3),self.weights2),
      (slice(None),slice(None),slice(0,self.modes1),slice(-self.modes2,None),slice(0,self.modes3),self.weights3),
      (slice(None),slice(None),slice(-self.modes1,None),slice(-self.modes2,None),slice(0,self.modes3),self.weights4),
    ]
    for bs,cs,xs,ys,zs,w in blocks:
        a=xf[bs,cs,xs,ys,zs]
        if mode=='activation': a=symmetric_qdq_complex_independent(a,bits)
        if mode=='weight': w=symmetric_qdq_complex_independent(w,bits)
        out[bs,:,xs,ys,zs]=_mul(a,w)
    return torch.fft.irfftn(out,s=(x.size(-3),x.size(-2),x.size(-1)))


def patch_realpdebench_fno(model,mode:str,bits:int=6):
    if mode not in ('activation','weight','fp32'): raise ValueError(mode)
    count=0
    for mod in model.modules():
        if mod.__class__.__name__=='SpectralConv3d':
            if mode=='fp32': continue
            mod.forward=types.MethodType(lambda self,x,_m=mode,_b=bits:_forward(self,x,_m,_b),mod); count+=1
    if mode!='fp32' and count==0: raise RuntimeError('No SpectralConv3d modules found; check RealPDEBench revision/model.')
    return model
