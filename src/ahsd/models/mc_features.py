"""Chirp-mass features from a dense matched-filter bank (build_mc_bank.py).

The network on its own reads chirp mass to ~8% (BNS) and ~126x dynesty's
width (BBH): its features carry no precise chirp-phase information. A matched
filter over a fine chirp-mass grid does: the network-SNR profile peaks at the
true chirp mass, and the peak interpolated between grid points measures BNS
chirp mass to ~0.15% from the 4 s window at SNR >= 8, and to ~0.05% with the
28 s context stream added (scratch probe 2026-10-31).

Per event, from the profile (window SNR^2 + context SNR^2, max over mass
ratio, summed over detectors):
  - interpolated peak log chirp mass, and peak network SNR
  - the profile around the peak (2W+1 points, relative to the peak): width
  - the strongest separate peak (>3% away): its height and position, so a
    second chirp-mass/merger-time mode is visible to the flow
  - window-only and context-only peaks (consistency of the two streams)
  - the peak in each mass-ratio slice of the bank (chirp mass and SNR): the
    chirp mass that fits best depends on mass ratio (and spin), so the flow
    gets the trade-off instead of a single grid maximum
These go through a small MLP into n_tokens transformer tokens.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SR, T_LEN = 4096, 16384
W = 8                    # profile half-window (grid points)
SECOND_PEAK_EXCL = 0.03  # a second peak must be > 3% away in chirp mass


class McFeatures(nn.Module):
    def __init__(self, bank: str, band_lo: int, n_band: int, lowrate_len: int = 0,
                 d_model: int = 192, n_tokens: int = 2):
        super().__init__()
        path = Path(bank)
        if not path.is_absolute() and path.parent == Path("."):
            path = Path(__file__).parent / "assets" / path
        b = np.load(path)
        assert int(b["band_lo"]) == band_lo and int(b["n_band"]) == n_band, \
            "mc bank band does not match the encoder band; rebuild with build_mc_bank.py"
        self.n_q = int(b["n_q"])
        self.dlog = float(b["dlog"])
        grid = np.log(b["mc_grid"].astype(np.float64))
        self.n_mc = len(grid)
        self.log_lo, self.log_hi = float(grid[0]), float(grid[-1])
        # bank buffers are deterministic build artifacts: not saved in checkpoints
        self.register_buffer("log_grid", torch.from_numpy(grid.astype(np.float32)), persistent=False)
        self.register_buffer("win_re", torch.from_numpy(b["win_re"]), persistent=False)
        self.register_buffer("win_im", torch.from_numpy(b["win_im"]), persistent=False)
        self.use_ctx = bool(lowrate_len) and "ctx_re" in b.files
        if lowrate_len and not self.use_ctx:
            raise ValueError(f"{path.name} has no context templates; build it with --context")
        if self.use_ctx:
            assert int(b["ctx_len"]) == lowrate_len, "mc bank context length != model lowrate_len"
            self.ctx_len = int(lowrate_len)
            self.register_buffer("ctx_re", torch.from_numpy(b["ctx_re"]), persistent=False)
            self.register_buffer("ctx_im", torch.from_numpy(b["ctx_im"]), persistent=False)
        self.band_lo, self.n_band = band_lo, n_band
        self.n_features = 2 + (2 * W + 1) + 2 + 4 + (2 * self.n_q if self.n_q > 1 else 0)
        self.mlp = nn.Sequential(nn.LayerNorm(self.n_features), nn.Linear(self.n_features, 128), nn.GELU(),
                                 nn.Linear(128, 128), nn.GELU())
        self.n_tokens = int(n_tokens)
        self.to_tokens = nn.Linear(128, self.n_tokens * d_model)
        self.type_embed = nn.Parameter(torch.zeros(d_model))     # zero-init marker

    @staticmethod
    def _profile(d: torch.Tensor, H: torch.Tensor, n: int, scale: float) -> torch.Tensor:
        """Network peak SNR^2 per template: d [B,D,F] one-sided spectrum,
        H [D,K,F] unit-norm templates; complex ifft over n points (phase-
        maximized matched filter), max over lag, summed over detectors."""
        B, Dn, F = d.shape
        K = H.shape[1]
        chunk = max(1, int(6e7 / max(B * Dn * F * 2, 1)))
        out = []
        for k0 in range(0, K, chunk):
            z = torch.fft.ifft(d.unsqueeze(2) * H[:, k0:k0 + chunk].conj().unsqueeze(0), n=n, dim=-1)
            out.append(((z.abs().amax(-1) * scale) ** 2).sum(1))        # [B, kc]
        return torch.cat(out, dim=1)

    def _peak(self, prof: torch.Tensor):
        """Interpolated peak of a profile over the chirp-mass grid [B, n_mc]:
        (normalized log Mc, asinh peak SNR, argmax index)."""
        k = prof.argmax(-1).clamp(1, self.n_mc - 2)
        g = lambda o: prof.gather(1, (k + o).unsqueeze(1)).squeeze(1)
        y0, y1, y2 = g(-1), g(0), g(1)
        den = y0 - 2 * y1 + y2
        dk = torch.where(den < 0, 0.5 * (y0 - y2) / den.clamp(max=-1e-9), torch.zeros_like(den)).clamp(-1, 1)
        log_mc = self.log_grid[k] + dk * self.dlog
        return self._norm(log_mc), torch.asinh(y1.clamp_min(0).sqrt()), k

    def _norm(self, log_mc):
        return 2 * (log_mc - self.log_lo) / (self.log_hi - self.log_lo) - 1

    def forward(self, hi: torch.Tensor, lo: torch.Tensor = None) -> torch.Tensor:
        B = hi.shape[0]
        d = torch.fft.rfft(hi.float(), dim=-1)[..., self.band_lo:self.band_lo + self.n_band]
        Hw = torch.complex(self.win_re, self.win_im)
        # 2x oversampled lag axis: the peak of a ~1 kHz merger is not clipped
        pw = self._profile(d, Hw, 2 * self.n_band, 2 * self.n_band / math.sqrt(T_LEN / 2.0))
        pw_q = pw.reshape(B, self.n_mc, self.n_q)
        pw = pw_q.amax(-1)
        if self.use_ctx and lo is not None:
            dc = torch.fft.rfft(lo.float(), dim=-1)
            Hc = torch.complex(self.ctx_re, self.ctx_im)
            L = self.ctx_len
            pc_q = self._profile(dc, Hc, L, L / math.sqrt(L / 2.0)).reshape(B, self.n_mc, self.n_q)
            pc = pc_q.amax(-1)
        else:
            pc_q = torch.zeros_like(pw_q)
            pc = torch.zeros_like(pw)
        prof = pw + pc
        x_mc, x_snr, k = self._peak(prof)
        # profile around the peak, relative to it (replicate-padded at the edges)
        idx = (k.unsqueeze(1) + torch.arange(-W, W + 1, device=hi.device)).clamp(0, self.n_mc - 1)
        local = prof.gather(1, idx) / prof.gather(1, k.unsqueeze(1)).clamp_min(1e-6)
        # strongest separate peak
        excl = max(1, int(round(SECOND_PEAK_EXCL / self.dlog)))
        far = (torch.arange(self.n_mc, device=hi.device).unsqueeze(0) - k.unsqueeze(1)).abs() > excl
        p2 = torch.where(far, prof, torch.full_like(prof, -1.0))
        k2 = p2.argmax(-1)
        h2 = (p2.gather(1, k2.unsqueeze(1)).squeeze(1).clamp_min(0)
              / prof.gather(1, k.unsqueeze(1)).squeeze(1).clamp_min(1e-6)).sqrt()
        x2 = self._norm(self.log_grid[k2]) - x_mc
        w_mc, w_snr, _ = self._peak(pw)
        if self.use_ctx and lo is not None:
            c_mc, c_snr, _ = self._peak(pc)
        else:
            c_mc, c_snr = torch.zeros_like(w_mc), torch.zeros_like(w_snr)
        parts = [x_mc[:, None], x_snr[:, None], local, h2[:, None], x2[:, None],
                 w_mc[:, None], w_snr[:, None], c_mc[:, None], c_snr[:, None]]
        if self.n_q > 1:
            pq = pw_q + pc_q
            for j in range(self.n_q):
                q_mc, q_snr, _ = self._peak(pq[..., j])
                parts += [q_mc[:, None], q_snr[:, None]]
        f = torch.cat(parts, dim=1)
        tok = self.to_tokens(self.mlp(f)).reshape(B, self.n_tokens, -1)
        return tok + self.type_embed

    @torch.no_grad()
    def estimate(self, hi: torch.Tensor, lo: torch.Tensor = None) -> torch.Tensor:
        """Peak chirp mass [B] (diagnostics/tests)."""
        B = hi.shape[0]
        d = torch.fft.rfft(hi.float(), dim=-1)[..., self.band_lo:self.band_lo + self.n_band]
        prof = self._profile(d, torch.complex(self.win_re, self.win_im), 2 * self.n_band,
                             2 * self.n_band / math.sqrt(T_LEN / 2.0)).reshape(B, self.n_mc, self.n_q).amax(-1)
        if self.use_ctx and lo is not None:
            L = self.ctx_len
            prof = prof + self._profile(torch.fft.rfft(lo.float(), dim=-1), torch.complex(self.ctx_re, self.ctx_im),
                                        L, L / math.sqrt(L / 2.0)).reshape(B, self.n_mc, self.n_q).amax(-1)
        x, _, _ = self._peak(prof)
        return torch.exp((x + 1) / 2 * (self.log_hi - self.log_lo) + self.log_lo)
