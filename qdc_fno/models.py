from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantization import (
    symmetric_qdq_complex_independent,
    symmetric_qdq_complex_independent_ste,
    mixed_q4_q8_frozen,
    FrozenMixedQ4Q8,
    e4m3fn_complex,
)


def compl_mul2d(x, w):
    return torch.einsum('bixy,ioxy->boxy', x, w)


def _group_count(width: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, width), 0, -1):
        if width % g == 0:
            return g
    return 1


class SpectralConv2dQ(nn.Module):
    """Two-sided rFFT spectral convolution with independently quantizable activations/weights."""
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / max(1, in_channels * out_channels)
        self.w_pos = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat))
        self.w_neg = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat))
        self.quant_mode = 'fp32'
        self.bits = 6
        self.mixed_state = None

    def set_quant(self, mode='fp32', bits=6, mixed_state: FrozenMixedQ4Q8 | None = None):
        self.quant_mode = mode
        self.bits = bits
        self.mixed_state = mixed_state

    def _qact(self, z):
        if self.quant_mode == 'act_q6':
            return symmetric_qdq_complex_independent(z, self.bits)
        if self.quant_mode == 'act_q6_qat':
            return symmetric_qdq_complex_independent_ste(z, self.bits)
        if self.quant_mode == 'act_fp8':
            return e4m3fn_complex(z)
        if self.quant_mode == 'act_mixed':
            if self.mixed_state is None:
                raise RuntimeError('act_mixed requires frozen calibration state')
            return mixed_q4_q8_frozen(z, self.mixed_state, channel_axis=1)
        return z

    def _qwt(self, z):
        if self.quant_mode == 'weight_q6':
            return symmetric_qdq_complex_independent(z, self.bits)
        if self.quant_mode == 'weight_fp8':
            return e4m3fn_complex(z)
        return z

    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x, norm='ortho')
        WF = xf.shape[-1]
        m_h = min(self.modes, max(1, H // 2))
        m_w = min(self.modes, WF)
        out = torch.zeros(B, self.out_channels, H, WF, dtype=torch.cfloat, device=x.device)
        xp = self._qact(xf[:, :, :m_h, :m_w])
        xn = self._qact(xf[:, :, -m_h:, :m_w])
        wp = self._qwt(self.w_pos[:, :, :m_h, :m_w])
        wn = self._qwt(self.w_neg[:, :, :m_h, :m_w])
        out[:, :, :m_h, :m_w] = compl_mul2d(xp, wp)
        out[:, :, -m_h:, :m_w] = compl_mul2d(xn, wn)
        return torch.fft.irfft2(out, s=(H, W), norm='ortho')


class FNO2dQ(nn.Module):
    """Physics-aligned FNO used by the cleaned synthetic reproduction workflows.

    The model appends spatial x/y coordinates, combines spectral, pointwise, and local 3x3
    branches, uses GroupNorm and GELU, and retains a small residual connection in each block.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        width=60,
        modes=8,
        layers=5,
        hidden=128,
        hidden2=64,
        residual_scale=0.15,
        append_coords=True,
    ):
        super().__init__()
        self.width = width
        self.append_coords = append_coords
        self.residual_scale = float(residual_scale)
        lifted_channels = in_channels + (2 if append_coords else 0)
        self.fc0 = nn.Conv2d(lifted_channels, width, 1)
        self.spec = nn.ModuleList([SpectralConv2dQ(width, width, modes) for _ in range(layers)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.local = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, padding_mode='circular'),
                nn.GELU(),
                nn.Conv2d(width, width, 3, padding=1, padding_mode='circular'),
            ) for _ in range(layers)
        ])
        self.norm = nn.ModuleList([nn.GroupNorm(_group_count(width), width) for _ in range(layers)])
        self.fc1 = nn.Conv2d(width, hidden, 1)
        self.fc_mid = nn.Conv2d(hidden, hidden2, 1)
        self.fc2 = nn.Conv2d(hidden2, out_channels, 1)

    def set_quant(self, mode='fp32', bits=6, mixed_states=None):
        for i, s in enumerate(self.spec):
            s.set_quant(mode, bits, None if mixed_states is None else mixed_states[i])

    @staticmethod
    def _coords(x):
        B, _, H, W = x.shape
        yy = (torch.arange(H, device=x.device, dtype=x.dtype) / float(H)).view(1, 1, H, 1).expand(B, 1, H, W)
        xx = (torch.arange(W, device=x.device, dtype=x.dtype) / float(W)).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([xx, yy], dim=1)

    def forward(self, x):
        if self.append_coords:
            x = torch.cat([x, self._coords(x)], dim=1)
        x = self.fc0(x)
        for i, (s, p, l, n) in enumerate(zip(self.spec, self.pointwise, self.local, self.norm)):
            skip = x
            x = n(s(x) + p(x) + l(x)) + self.residual_scale * skip
            if i < len(self.spec) - 1:
                x = F.gelu(x)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc_mid(x))
        return self.fc2(x)
