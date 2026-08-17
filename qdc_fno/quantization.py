from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import torch

EPS = 1e-8


def _scale_from_absmax(x: torch.Tensor, qmax: int, dims=None, keepdim=True) -> torch.Tensor:
    if dims is None:
        amax = x.detach().abs().amax()
    else:
        amax = x.detach().abs().amax(dim=dims, keepdim=keepdim)
    scale = amax / float(qmax)
    return torch.where(scale > 0, scale, torch.ones_like(scale))


def symmetric_qdq_real(x: torch.Tensor, bits: int, scale: torch.Tensor | None = None) -> torch.Tensor:
    """Symmetric QDQ with levels [-qmax, qmax], zero-point 0 and round-to-even."""
    qmax = 2 ** (bits - 1) - 1
    if scale is None:
        scale = _scale_from_absmax(x, qmax, dims=None, keepdim=True)
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return q * scale




def symmetric_qdq_real_ste(x: torch.Tensor, bits: int, scale: torch.Tensor | None = None) -> torch.Tensor:
    """Straight-through fake quantization used during QAT."""
    q = symmetric_qdq_real(x, bits, scale)
    return x + (q - x).detach()


def symmetric_qdq_complex_independent_ste(z: torch.Tensor, bits: int) -> torch.Tensor:
    return torch.complex(symmetric_qdq_real_ste(z.real, bits), symmetric_qdq_real_ste(z.imag, bits))

def symmetric_qdq_complex_independent(z: torch.Tensor, bits: int) -> torch.Tensor:
    """Core-Q6 complex QDQ: real/imaginary components use independent absmax scales."""
    return torch.complex(symmetric_qdq_real(z.real, bits), symmetric_qdq_real(z.imag, bits))


def mode_magnitude_scores(z: torch.Tensor, mode_dims: tuple[int, ...]) -> torch.Tensor:
    """Mean complex magnitude over all non-mode axes."""
    keep = set(mode_dims)
    reduce = tuple(d for d in range(z.ndim) if d not in keep)
    return z.detach().abs().mean(dim=reduce)


def top_fraction_mask(scores: torch.Tensor, fraction: float = 0.25) -> torch.Tensor:
    """Deterministic top-fraction mask with lexicographic tie breaking."""
    flat = scores.flatten()
    k = int(round(fraction * flat.numel()))
    k = min(max(k, 0), flat.numel())
    order = sorted(range(flat.numel()), key=lambda idx: (-float(flat[idx]), idx))
    out = torch.zeros_like(flat, dtype=torch.bool)
    if k:
        out[torch.tensor(order[:k], device=flat.device)] = True
    return out.reshape_as(scores)


@dataclass
class FrozenMixedQ4Q8:
    """Frozen calibration artifact for the separate mixed Q4/Q8 control.

    mask is defined over retained mode axes and broadcast over batch/channel.
    q4_scale and q8_scale are one value per channel (broadcastable to activations).
    """
    mask: torch.Tensor
    q4_scale: torch.Tensor
    q8_scale: torch.Tensor

    def to(self, device):
        return FrozenMixedQ4Q8(self.mask.to(device), self.q4_scale.to(device), self.q8_scale.to(device))


def _subset_channel_scale(samples: list[torch.Tensor], masks: list[torch.Tensor], bits: int, channel_axis: int) -> torch.Tensor:
    qmax = 127 if bits == 8 else 7
    maxima = None
    for z, mask in zip(samples, masks):
        m = mask
        while m.ndim < z.ndim:
            m = m.unsqueeze(0)
        subset = m if bits == 8 else ~m
        # shared real/imag scale, reduce every axis except channel
        rr = torch.where(subset, z.real.detach(), torch.zeros_like(z.real))
        ii = torch.where(subset, z.imag.detach(), torch.zeros_like(z.imag))
        stacked = torch.stack([rr.abs(), ii.abs()], dim=0)
        reduce_dims = (0,) + tuple(d + 1 for d in range(z.ndim) if d != channel_axis)
        cur = stacked.amax(dim=reduce_dims, keepdim=False)
        maxima = cur if maxima is None else torch.maximum(maxima, cur)
    if maxima is None:
        raise ValueError("calibration samples cannot be empty")
    scale = maxima / float(qmax)
    return torch.where(scale > 0, scale, torch.ones_like(scale))


def calibrate_mixed_q4_q8(samples: list[torch.Tensor], mode_dims: tuple[int, ...], channel_axis: int = 1,
                           fraction_q8: float = 0.25) -> FrozenMixedQ4Q8:
    """Calibrate and freeze mixed-Q4/Q8 mask and numerical scales.

    The caller should pass activations from the fixed 500-example calibration set for a single layer.
    The mask is shared across real/imaginary components. No test-time scale recomputation is used.
    """
    if not samples:
        raise ValueError("calibration samples cannot be empty")
    scores = torch.stack([mode_magnitude_scores(z, mode_dims) for z in samples]).mean(0)
    mask = top_fraction_mask(scores, fraction_q8)
    masks = [mask] * len(samples)
    q4 = _subset_channel_scale(samples, masks, 4, channel_axis)
    q8 = _subset_channel_scale(samples, masks, 8, channel_axis)
    return FrozenMixedQ4Q8(mask.cpu(), q4.cpu(), q8.cpu())


def _broadcast_channel_scale(scale: torch.Tensor, z: torch.Tensor, channel_axis: int) -> torch.Tensor:
    shape = [1] * z.ndim
    shape[channel_axis] = scale.numel()
    return scale.reshape(shape).to(z.device, z.real.dtype)


def mixed_q4_q8_frozen(z: torch.Tensor, state: FrozenMixedQ4Q8, channel_axis: int = 1) -> torch.Tensor:
    """Apply the frozen mixed Q4/Q8 calibration artifact to a complex activation tensor."""
    state = state.to(z.device)
    mask = state.mask
    while mask.ndim < z.ndim:
        mask = mask.unsqueeze(0)
    out = torch.zeros_like(z)
    for bits, subset, scale_vec in ((8, mask, state.q8_scale), (4, ~mask, state.q4_scale)):
        qmin, qmax = (-128, 127) if bits == 8 else (-8, 7)
        scale = _broadcast_channel_scale(scale_vec, z, channel_axis)
        qr = torch.clamp(torch.round(z.real / scale), qmin, qmax) * scale
        qi = torch.clamp(torch.round(z.imag / scale), qmin, qmax) * scale
        out = torch.where(subset, torch.complex(qr, qi), out)
    return out


def e4m3fn_qdq(x: torch.Tensor) -> torch.Tensor:
    """Software E4M3FN conversion with saturation to the largest finite value."""
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("PyTorch build does not expose torch.float8_e4m3fn")
    finfo = torch.finfo(torch.float8_e4m3fn)
    x = torch.clamp(x, -finfo.max, finfo.max)
    return x.to(torch.float8_e4m3fn).to(x.dtype)


def e4m3fn_complex(z: torch.Tensor) -> torch.Tensor:
    return torch.complex(e4m3fn_qdq(z.real), e4m3fn_qdq(z.imag))
