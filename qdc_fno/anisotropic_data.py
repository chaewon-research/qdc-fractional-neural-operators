from __future__ import annotations
import math, random
import torch
from torch.utils.data import TensorDataset
from .operators import smooth_random_field, build_anisotropic_operator, eigh_nonnegative, solve_fractional_from_eigh

EPS=1e-8

def _grid(n):
    a=torch.arange(n,dtype=torch.float64)/float(n); yy,xx=torch.meshgrid(a,a,indexing='ij'); return xx,yy

def _periodic_gaussian(xx,yy,cx,cy,sigma):
    dx=torch.minimum((xx-cx).abs(),1-(xx-cx).abs()); dy=torch.minimum((yy-cy).abs(),1-(yy-cy).abs())
    return torch.exp(-(dx*dx+dy*dy)/(2*sigma*sigma))

def _source(n,regime,rng):
    xx,yy=_grid(n); f=torch.zeros((n,n),dtype=torch.float64)
    if regime=='train': num=rng.randint(1,3); sr=(.070,.160); ar=(.7,1.25)
    elif regime=='high_contrast': num=rng.randint(2,4); sr=(.050,.130); ar=(.8,1.45)
    else: num=rng.randint(2,5); sr=(.035,.110); ar=(.9,1.65)
    for _ in range(num):
        amp=rng.uniform(*ar)*(-1 if rng.random()<.30 else 1); f += amp*_periodic_gaussian(xx,yy,rng.random(),rng.random(),rng.uniform(*sr))
    if rng.random()<.5:
        kmax=3 if regime=='train' else 4; kx=rng.randint(1,kmax); ky=rng.randint(1,kmax); ph=rng.random()*2*math.pi
        f += .20*torch.sin(2*math.pi*(kx*xx+ky*yy)+ph)
    return (f-f.mean())/(f.std()+EPS)

def _oriented(base,regime,axis,rng):
    n=base.shape[0]; xx,yy=_grid(n); out=base.clone()
    if regime=='train': nl=rng.randint(1,2); ar=(.45,.90); wr=(.030,.070)
    elif regime=='high_contrast': nl=rng.randint(2,4); ar=(.70,1.25); wr=(.020,.055)
    else: nl=rng.randint(3,5); ar=(.90,1.60); wr=(.014,.045)
    for _ in range(nl):
        theta=rng.gauss(0,.45) if axis=='x' else rng.gauss(math.pi/2,.45); cx,cy=rng.random(),rng.random()
        dist=((xx-cx)*math.cos(theta)+(yy-cy)*math.sin(theta)).abs(); dist=torch.minimum(dist,1-torch.clamp(dist,0,1))
        out += (1 if rng.random()<.75 else -1)*rng.uniform(*ar)*torch.exp(-(dist/rng.uniform(*wr))**2)
    return (out-out.mean())/(out.std()+EPS)

def _logk_pair(n,regime,gen,rng):
    if regime=='train': alpha=rng.uniform(3,5); contrast=rng.uniform(.70,1.35); amp=rng.uniform(.10,.35); clip=4
    elif regime=='high_contrast': alpha=rng.uniform(2.3,4.2); contrast=rng.uniform(1.20,1.75); amp=rng.uniform(.25,.55); clip=4
    else: alpha=rng.uniform(1.8,3.4); contrast=rng.uniform(1.60,2.25); amp=rng.uniform(.40,.80); clip=4.5
    shared=smooth_random_field(n,alpha,gen); fx=smooth_random_field(n,max(1.5,alpha-.3),gen); fy=smooth_random_field(n,max(1.5,alpha-.3),gen)
    lx=_oriented(shared+amp*fx,regime,'x',rng); ly=_oriented(shared-amp*fy,regime,'y',rng)
    lx=torch.clamp(contrast*(lx-lx.mean())/(lx.std()+EPS),-clip,clip); ly=torch.clamp(contrast*(ly-ly.mean())/(ly.std()+EPS),-clip,clip)
    return lx,ly

def generate_anisotropic_dataset(num:int,N:int,lam:float,split:str,seed:int,s_values=(.45,.60,.75,.90,1.0)):
    regime='train' if split in ('train','val','test_id') else ('high_contrast' if split in ('high_contrast','old_ood') else 'extreme')
    gen=torch.Generator().manual_seed(seed); rng=random.Random(seed); xs=[]; ys=[]
    sseq=(list(s_values)*((num+len(s_values)-1)//len(s_values)))[:num]; rng.shuffle(sseq)
    for sv in sseq:
        lx,ly=_logk_pair(N,regime,gen,rng); f=_source(N,regime,rng); L=build_anisotropic_operator(torch.exp(lx),torch.exp(ly)); vals,vecs=eigh_nonnegative(L); u=solve_fractional_from_eigh(vals,vecs,f,float(sv),lam)
        sf=torch.full((N,N),float(sv),dtype=torch.float64); xs.append(torch.stack([lx,ly,f,sf]).float()); ys.append(u.unsqueeze(0))
    return TensorDataset(torch.stack(xs),torch.stack(ys))
