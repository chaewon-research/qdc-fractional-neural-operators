import torch
from qdc_fno.quantization import symmetric_qdq_real,symmetric_qdq_complex_independent,calibrate_mixed_q4_q8,mixed_q4_q8_frozen
from qdc_fno.metrics import anisotropic_objective,scalar_qdc_objective
from qdc_fno.qdc import qdc_apply,SelectorRow,select_residual_margin


def test_q6_levels():
    x=torch.tensor([-10.,-1.,0.,1.,10.]); q=symmetric_qdq_real(x,6); assert torch.isfinite(q).all(); assert q.max()<=10 and q.min()>=-10


def test_complex_q6():
    z=torch.complex(torch.randn(4,4),torch.randn(4,4)); q=symmetric_qdq_complex_independent(z,6); assert q.dtype.is_complex


def test_objectives():
    assert abs(anisotropic_objective(.2,.3,.4)-.5)<1e-12
    assert abs(scalar_qdc_objective(.1,.2,.3,.04)-.34)<1e-12


def test_full_qdc_recovers_linear_solution():
    vals=torch.tensor([0.,1.,2.],dtype=torch.float64); vecs=torch.eye(3,dtype=torch.float64); f=torch.tensor([1.,2.,3.]); uq=torch.zeros(3); idx=torch.arange(3); u=qdc_apply(uq,f,vals,vecs,.5,.15,idx); exact=(f.double()/(.15+vals.sqrt())).float(); assert torch.allclose(u,exact,atol=1e-5)


def test_matrixfree_scalar_matches_dense():
    from qdc_fno.matrixfree import scalar_matvec_periodic
    from qdc_fno.operators import build_scalar_operator
    g=torch.Generator().manual_seed(7); n=5; k=torch.exp(0.1*torch.randn(n,n,generator=g,dtype=torch.float64)); x=torch.randn(n*n,generator=g,dtype=torch.float64)
    dense=build_scalar_operator(k)@x; mf=scalar_matvec_periodic(k,x); assert torch.allclose(dense,mf,atol=1e-10,rtol=1e-10)


def test_frozen_mixed_scales_are_test_input_independent():
    g=torch.Generator().manual_seed(4); samples=[torch.complex(torch.randn(2,3,4,4,generator=g),torch.randn(2,3,4,4,generator=g)) for _ in range(3)]
    st=calibrate_mixed_q4_q8(samples,(2,3),1,.25); q4=st.q4_scale.clone(); q8=st.q8_scale.clone(); z=100*torch.complex(torch.randn(2,3,4,4,generator=g),torch.randn(2,3,4,4,generator=g)); _=mixed_q4_q8_frozen(z,st,1)
    assert torch.equal(q4,st.q4_scale) and torch.equal(q8,st.q8_scale)


def test_residual_aware_loss_has_prediction_gradient():
    from qdc_fno.training import differentiable_scalar_residual_loss
    g=torch.Generator().manual_seed(3); n=4; pred=torch.randn(1,1,n,n,generator=g,requires_grad=True); logk=torch.zeros(n,n); f=torch.randn(n,n,generator=g); s=torch.full((n,n),.7); x=torch.stack([logk,f,s])[None]
    loss=differentiable_scalar_residual_loss(pred,x,.15); loss.backward(); assert pred.grad is not None and float(pred.grad.abs().sum())>0


def test_selector_prefers_smallest_strong_feasible():
    rows=[SelectorRow(41,.9,.8,.9),SelectorRow(82,.7,.7,.7),SelectorRow(100,.6,.6,.6)]
    got=select_residual_margin(rows,.8,.8,.8,.92); assert got.k==82


def test_anisotropic_axis_convention():
    from qdc_fno.operators import build_anisotropic_operator
    n=4; kx=torch.full((n,n),2.,dtype=torch.float64); ky=torch.ones((n,n),dtype=torch.float64); L=build_anisotropic_operator(kx,ky)
    # Field varying only along axis 0 should feel kx=2; field varying only along axis 1 should feel ky=1.
    a=torch.arange(n,dtype=torch.float64); u0=a[:,None].repeat(1,n).reshape(-1); u1=a[None,:].repeat(n,1).reshape(-1)
    assert float(torch.linalg.vector_norm(L@u0)) > 1.5*float(torch.linalg.vector_norm(L@u1))


def test_anisotropic_event_mask_uses_average_log_coefficient():
    from qdc_fno.metrics import event_mask
    n=6; x=torch.linspace(0,1,n).view(n,1).repeat(1,n); y=torch.linspace(0,1,n).view(1,n).repeat(n,1)
    lx=x; ly=y
    m1=event_mask((0.5*(lx+ly))[None],.8)
    m2=event_mask((0.5*(lx+ly))[None],.8)
    assert torch.equal(m1,m2)


def test_matrixfree_double_output_for_validation():
    from qdc_fno.matrixfree import solve_fractional_matrixfree
    g=torch.Generator().manual_seed(11); n=5; k=torch.exp(.1*torch.randn(n,n,generator=g,dtype=torch.float64)); f=torch.randn(n,n,generator=g,dtype=torch.float64)
    u,meta=solve_fractional_matrixfree(k,f,.7,.15,start_dim=n*n,max_dim=n*n,return_double=True)
    assert u.dtype==torch.float64 and meta['converged']

def test_highres_cache_fingerprint_changes_with_scientific_config():
    from qdc_fno.highres_data import highres_cache_fingerprint
    cfg={'train_alpha':[2.2,5.0]}
    a,_=highres_cache_fingerprint(10,64,.15,'train',1,data_config=cfg)
    b,_=highres_cache_fingerprint(10,64,.16,'train',1,data_config=cfg)
    c,_=highres_cache_fingerprint(10,64,.15,'train',1,data_config={'train_alpha':[2.3,5.0]})
    assert a != b and a != c


def test_selector_weak_requires_residual_guard_and_promotes_one_rank():
    rows=[SelectorRow(41,.70,.70,.85),SelectorRow(82,.75,.75,.79),SelectorRow(100,.60,.60,.60)]
    # best residual=.8: K41 fails residual; K82 is weak (but not strong because .79 > .736); promote to K100.
    got=select_residual_margin(rows,.8,.8,.8,.92)
    assert got.k==100


def test_selector_raises_when_no_feasible_candidate():
    rows=[SelectorRow(41,.9,.9,.9),SelectorRow(82,.85,.85,.85)]
    import pytest
    with pytest.raises(ValueError):
        select_residual_margin(rows,.8,.8,.8,.92)


def test_qat_fake_quantization_has_ste_gradient():
    from qdc_fno.quantization import symmetric_qdq_real_ste
    x=torch.randn(17,requires_grad=True); y=symmetric_qdq_real_ste(x,6).sum(); y.backward()
    assert x.grad is not None and torch.allclose(x.grad,torch.ones_like(x.grad))


def test_clean_fno_appends_coordinates_and_runs():
    from qdc_fno.models import FNO2dQ
    m=FNO2dQ(3,1,width=12,modes=3,layers=2,hidden=16,hidden2=8)
    x=torch.randn(2,3,8,8); y=m(x); assert y.shape==(2,1,8,8)


def test_selector_weak_at_max_rank_has_no_promotable_budget():
    import pytest
    rows=[SelectorRow(41,.9,.9,.9),SelectorRow(82,.7,.7,.79)]
    with pytest.raises(ValueError, match='no larger budget'):
        select_residual_margin(rows,.8,.8,.8,.92)

def test_spatial_coordinate_channels_use_nonduplicated_periodic_grid():
    from qdc_fno.models import FNO2dQ
    x=torch.zeros(1,3,5,7); c=FNO2dQ._coords(x)
    assert c.shape==(1,2,5,7)
    assert torch.isclose(c[:,0].min(),torch.tensor(0.)) and torch.isclose(c[:,0].max(),torch.tensor(6/7))
    assert torch.isclose(c[:,1].min(),torch.tensor(0.)) and torch.isclose(c[:,1].max(),torch.tensor(4/5))
