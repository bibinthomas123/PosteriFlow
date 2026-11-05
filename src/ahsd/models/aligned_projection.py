"""Time-aligned waveform-basis projection: an encoder branch that sees the
whitened strain linearly, in the frequency domain.

The time-domain stem compresses its input (asinh), and the chirp-mass
posterior width times SNR rises with SNR (2.4 -> 6.0 from SNR 8 to 35+ for
BBH v5): the encoder stops sharpening on loud signals. Chirp-mass precision
comes from phase coherence over many cycles, which a linear projection onto
a waveform basis keeps.

Each detector's data are shifted in time by that detector's arrival time from
the template-bank trigger (McFeatures._trigger), so the merger sits at t = 0
as in the basis (models/build_aligned_basis.py), then projected onto the K
basis vectors. A fixed basis cannot follow a merger anywhere in +-1.5 s; the
alignment is what DINGO does with GNPE, done once from the bank. The
coefficients (unit noise per component, no learned normalization: amplitude
is a distance cue) go through an MLP to one token per detector.

An earlier SVD front-end that REPLACED the time-domain encoder, with random
time shifts folded into the basis, fit the marginal (memory: coherent
encoder notes); this branch is added next to the working encoder.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SR, T_LEN = 4096, 16384
SCALE = 5.0          # fixed: coefficients are N(0,1) per component in noise


class AlignedProjection(nn.Module):
    def __init__(self, basis: str, band_lo: int, n_band: int, d_model: int, n_det: int = 3):
        super().__init__()
        path = Path(basis)
        if not path.is_absolute() and path.parent == Path("."):
            path = Path(__file__).parent / "assets" / path
        b = np.load(path)
        assert int(b["band_lo"]) == band_lo and int(b["n_band"]) == n_band, \
            "aligned basis band does not match the encoder band"
        self.band_lo, self.n_band = band_lo, n_band
        self.register_buffer("v_re", torch.from_numpy(b["v_re"]), persistent=False)
        self.register_buffer("v_im", torch.from_numpy(b["v_im"]), persistent=False)
        self.k = int(b["v_re"].shape[0])
        f = (band_lo + torch.arange(n_band, dtype=torch.float32)) * SR / T_LEN
        self.register_buffer("freqs", f, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(2 * self.k, 512), nn.GELU(),
                                 nn.Linear(512, 256), nn.GELU(), nn.Linear(256, d_model))
        self.det_embed = nn.Parameter(torch.zeros(n_det, d_model))
        self.type_embed = nn.Parameter(torch.zeros(d_model))       # zero-init marker

    def coefficients(self, hi: torch.Tensor, t_arr: torch.Tensor) -> torch.Tensor:
        """hi [B, D, T] whitened window; t_arr [B, D] arrival time (s from the
        window start). Returns complex coefficients [B, D, K], unit noise."""
        d = torch.fft.rfft(hi.float(), dim=-1)[..., self.band_lo:self.band_lo + self.n_band]
        shift = torch.polar(torch.ones_like(d.real),
                            2 * math.pi * self.freqs * t_arr.unsqueeze(-1).float())
        v = torch.complex(self.v_re, self.v_im)
        return (d * shift) @ v.conj().T / math.sqrt(T_LEN)

    def forward(self, hi: torch.Tensor, t_arr: torch.Tensor) -> torch.Tensor:
        c = self.coefficients(hi, t_arr)
        x = torch.cat([c.real, c.imag], dim=-1) / SCALE                  # [B, D, 2K]
        return self.mlp(x) + self.det_embed + self.type_embed            # [B, D, d_model]
