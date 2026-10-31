"""Cross-detector polarization features for inclination / polarization / sky.

Each detector records one linear combination of the two polarizations,
d_I = F+_I h+ + Fx_I hx with h+ ~ (1 + cos^2 i)/2 and hx ~ cos i (90 deg out of
phase). Matched-filtering every detector with the SAME template gives one
complex number per detector, z_I = |z_I| e^{i phi_I}. Ratios between detectors
cancel distance and the unknown orbital phase:

    z_I / z_J = [F+_I (1+c^2)/2 - i Fx_I c] / [F+_J (1+c^2)/2 - i Fx_J c],  c = cos(i)

so they depend only on sky position (through F), polarization angle and
inclination — the quantities the encoder was measured not to carry
(analysis/inclination_info: probe R^2 ~ 0 on its features). This module forms
them explicitly, with sub-sample peak refinement: near merger the phase moves
~0.5-1 rad per ms, so reading it at the nearest ~1 ms baseband sample blurs the
relative phase by a few tenths of a radian.

Features per event ([B, 15]) for the template with the highest network SNR:
  per pair (HL, HV, LV): log |z_I/z_J|, cos/sin(phi_I - phi_J), t_I - t_J [ms]
  per detector:          asinh(SNR_I)
"""
from __future__ import annotations

import math

import torch

SR = 4096
PAIRS = ((0, 1), (0, 2), (1, 2))
N_FEATURES = 4 * len(PAIRS) + 3


def polarization_features(strain: torch.Tensor, bank_re: torch.Tensor, bank_im: torch.Tensor,
                          band_lo: int, window_s: float = 1.6) -> torch.Tensor:
    """strain [B, 3, T] whitened; bank [3, K, Nf] unit-norm whitened templates over
    rfft bins [band_lo, band_lo + Nf). Returns [B, N_FEATURES]."""

    B, D, T = strain.shape
    H = torch.complex(bank_re, bank_im)                               # [D, K, Nf]

    Nf = H.shape[-1]
    d = torch.fft.rfft(strain.float(), dim=-1)[..., band_lo: band_lo + Nf]   # [B, D, Nf]
    dt = (T / SR) / Nf                                                # baseband sample [s]

    ctr = Nf // 2
    w = min(int(window_s / dt), ctr - 2)

    scale = Nf / math.sqrt(T / 2.0)                                   # -> SNR units

    # pass 1 (chunked over templates, peaks only): the template with the
    # highest network SNR. A full [B, D, K, 2w] complex tensor is ~1.6 GB at
    # training batch size, so only each (detector, template) peak is kept.
    K = H.shape[1]
    chunk = max(1, int(6e7 / max(B * D * Nf * 8, 1)))
    net = []
    for k0 in range(0, K, chunk):
        zc_ = torch.fft.ifft(d.unsqueeze(2) * H[:, k0:k0 + chunk].conj().unsqueeze(0), dim=-1)
        net.append((zc_[..., ctr - w: ctr + w].abs().amax(-1) ** 2).sum(1))    # [B, kc]
    k_best = torch.cat(net, dim=1).argmax(-1)                          # [B]

    # pass 2: the full complex overlap series for that one template per event
    Hb = H[:, k_best].permute(1, 0, 2)                                # [B, D, Nf]
    zb = torch.fft.ifft(d * Hb.conj(), dim=-1)[..., ctr - w: ctr + w] * scale   # [B, D, 2w]
    ab = zb.abs()
    i = ab.argmax(-1).clamp(1, ab.shape[-1] - 2)                      # [B, D]

    # parabolic refinement of the |z| peak -> sub-sample offset in (-0.5, 0.5)
    g = lambda o: torch.take_along_dim(ab, (i + o).unsqueeze(-1), -1).squeeze(-1)
    y0, y1, y2 = g(-1), g(0), g(1)
    delta = (0.5 * (y0 - y2) / (y0 - 2 * y1 + y2).clamp(max=-1e-9)).clamp(-0.5, 0.5)

    # complex value at the refined peak (linear interpolation of the slowly
    # varying baseband series), then restore the heterodyne phase ramp
    zc = lambda o: torch.complex(torch.take_along_dim(zb.real, (i + o).unsqueeze(-1), -1).squeeze(-1),
                                 torch.take_along_dim(zb.imag, (i + o).unsqueeze(-1), -1).squeeze(-1))
    nb = torch.where(delta >= 0, zc(1), zc(-1))
    zpk = zc(0) + delta.abs() * (nb - zc(0))
    n_bin = i.float() + delta + (ctr - w)
    phase = torch.angle(zpk) + 2 * math.pi * band_lo * n_bin / Nf
    t_pk = (n_bin - ctr) * dt                                         # [B, D] s
    snr = zpk.abs().clamp_min(1e-6)

    feats = []

    for I, Jd in PAIRS:
        dphi = phase[:, I] - phase[:, Jd]
        feats += [torch.log(snr[:, I] / snr[:, Jd]), torch.cos(dphi), torch.sin(dphi),
                  1e3 * (t_pk[:, I] - t_pk[:, Jd])]
    feats += [torch.asinh(snr[:, j]) for j in range(D)]
    
    return torch.stack(feats, -1)
