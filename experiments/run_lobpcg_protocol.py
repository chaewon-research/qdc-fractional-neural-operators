"""Matrix-free lowest-eigenspace QDC diagnostic primitive using the paper LOBPCG protocol."""
import argparse,json,numpy as np,torch
from qdc_fno.lobpcg import torch_linear_operator,lowest_eigenspace,eigenpair_residuals
from qdc_fno.operators import smooth_random_field
from qdc_fno.matrixfree import scalar_matvec_periodic
from qdc_fno.utils import save_json
p=argparse.ArgumentParser(); p.add_argument('--n',type=int,default=32); p.add_argument('--k',type=int,default=82); p.add_argument('--seed',type=int,default=101); p.add_argument('--out',default='results/lobpcg.json'); a=p.parse_args()
g=torch.Generator().manual_seed(a.seed); logk=smooth_random_field(a.n,4.0,g); k=torch.exp(logk); op=torch_linear_operator(lambda x:scalar_matvec_periodic(k,x),a.n*a.n)
vals,vecs=lowest_eigenspace(op,a.k,seed=a.seed,tol=1e-4,maxiter=100); r=eigenpair_residuals(op,vals,vecs); res={'resolution':a.n,'k':a.k,'seed':a.seed,'max_eigenpair_residual':float(r.max()),'mean_eigenpair_residual':float(r.mean()),'protocol':{'largest':False,'tol':1e-4,'maxiter':100,'preconditioner':None}}; save_json(res,a.out); print(json.dumps(res,indent=2))
