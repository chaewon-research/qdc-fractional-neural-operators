from __future__ import annotations
import copy
import torch
import torch.nn.functional as F
from .operators import build_scalar_operator, eigh_nonnegative


def train_fp32(model, train_loader, val_loader, epochs, lr, weight_decay, device, grad_clip=1.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best, best_loss = None, float('inf')
    for _ in range(epochs):
        model.train(); model.set_quant('fp32')
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); pred = model(x); loss = F.mse_loss(pred, y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip); opt.step()
        sched.step(); model.eval(); total = 0.; n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                total += F.mse_loss(model(x), y, reduction='sum').item(); n += y.numel()
        score = total / max(n, 1)
        if score < best_loss:
            best_loss = score; best = copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model


def _spectral_gradient_prediction_loss(pred, target):
    mse = F.mse_loss(pred, target)
    spectral = (torch.fft.rfft2(pred) - torch.fft.rfft2(target)).abs().mean()
    gx = (pred[..., 1:] - pred[..., :-1] - target[..., 1:] + target[..., :-1]).abs().mean()
    return mse + 0.004 * spectral + 0.004 * gx


def differentiable_scalar_residual_loss(pred: torch.Tensor, x: torch.Tensor, lam: float, eps: float = 1e-8):
    """Mean squared relative PDE residual computed from the current model prediction.

    x channels are [log k, f, s]. The physical operator is treated as fixed data; gradients
    flow through pred. This is the differentiable residual-aware QAT term used by the reproducibility implementation.
    """
    losses = []
    for i in range(pred.shape[0]):
        logk = x[i, 0].detach().cpu().double(); f = x[i, 1].to(pred.device, pred.dtype)
        s = float(x[i, 2, 0, 0].item())
        vals, vecs = eigh_nonnegative(build_scalar_operator(torch.exp(logk)))
        vals, vecs = vals.to(pred.device, pred.dtype), vecs.to(pred.device, pred.dtype)
        A = (vecs * (float(lam) + vals.pow(s)).unsqueeze(0)) @ vecs.T
        u = pred[i, 0].reshape(-1)
        r = f.reshape(-1) - A @ u
        losses.append((torch.linalg.vector_norm(r) / (torch.linalg.vector_norm(f.reshape(-1)) + eps)).pow(2))
    return torch.stack(losses).mean()


def qat_finetune_scalar(model, train_loader, epochs, base_lr, weight_decay, device, lam: float,
                        residual_aware: bool = False):
    """Activation-QAT or residual-aware activation-QAT for the scalar family."""
    model.set_quant('act_q6_qat', bits=6)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr * 0.35, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); pred = model(x)
            loss = _spectral_gradient_prediction_loss(pred, y)
            if residual_aware:
                loss = loss + 0.025 * differentiable_scalar_residual_loss(pred, x, lam)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
    return model
