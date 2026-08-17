"""Validate the adaptive matrix-free target solver against exact dense targets in float64."""
import argparse,json,torch
from pathlib import Path
from qdc_fno.operators import smooth_random_field,make_source,build_scalar_operator,eigh_nonnegative
from qdc_fno.matrixfree import solve_fractional_matrixfree

p=argparse.ArgumentParser(); p.add_argument('--resolutions',nargs='+',type=int,default=[16,32]); p.add_argument('--samples',type=int,default=4)
p.add_argument('--seed',type=int,default=913); p.add_argument('--lambda-value',type=float,default=.15); p.add_argument('--residual-guard',type=float,default=1e-6); p.add_argument('--out',default='results/target_solver_validation.json')
a=p.parse_args(); rows=[]
for N in a.resolutions:
    g=torch.Generator().manual_seed(a.seed+N)
    for j in range(a.samples):
        alpha=float(torch.empty(()).uniform_(2.1,3.1,generator=g)); c=float(torch.empty(()).uniform_(1.0,1.3,generator=g)); s=[.45,.65,.85,1.0][j%4]
        logk=c*smooth_random_field(N,alpha,g); k=torch.exp(logk); f=make_source(N,g,ood=True)
        vals,vecs=eigh_nonnegative(build_scalar_operator(k))
        coeff=vecs.T@f.reshape(-1).double(); exact=(vecs@(coeff/(a.lambda_value+vals.pow(float(s))))).reshape_as(f)
        approx,meta=solve_fractional_matrixfree(k,f,s,a.lambda_value,start_dim=min(256,N*N),max_dim=min(512,N*N),rel_change_tol=1e-7,dim_step=64,return_double=True)
        rel=float(torch.linalg.vector_norm(approx-exact)/(torch.linalg.vector_norm(exact)+1e-14))
        cc=vecs.T@approx.reshape(-1).double(); Au=(vecs@((a.lambda_value+vals.pow(float(s)))*cc)).reshape_as(f)
        pres=float(torch.linalg.vector_norm(f-Au)/(torch.linalg.vector_norm(f)+1e-14)); passed=pres<=a.residual_guard
        rows.append({'resolution':N,'sample':j,'relative_target_error':rel,'relative_physical_residual':pres,'residual_guard_pass':passed,'solver':meta})
res={'protocol':'float64 adaptive Lanczos function action validated against exact dense eigensolve','rows':rows,
     'residual_guard':a.residual_guard,'max_relative_target_error':max(r['relative_target_error'] for r in rows),'max_relative_physical_residual':max(r['relative_physical_residual'] for r in rows),'all_residual_guards_pass':all(r['residual_guard_pass'] for r in rows)}
Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(res,indent=2)); print(json.dumps({k:res[k] for k in ['max_relative_target_error','max_relative_physical_residual','all_residual_guards_pass']},indent=2))
if not res['all_residual_guards_pass']:
    raise SystemExit('high-resolution target validation failed the declared residual guard')
