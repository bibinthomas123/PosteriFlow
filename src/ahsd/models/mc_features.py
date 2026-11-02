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

Trigger features (bank spec "<bank>.npz:trigger"): the network's merger-time
posterior was ~0.28 s wide and its sky (ra/dec) and inclination near the
prior, because the convolutional stem downsamples away the sub-millisecond
arrival-time differences and the inter-detector phases. The best template's
complex SNR series, at 8x oversampled lags (0.12 ms), gives them directly:
merger time to ~1-2 ms and the H1-L1 delay to ~0.25 ms at SNR 15-25
(measured 2026-11-01). Per event:
  - the two strongest network-SNR peaks in time (> 50 ms apart): time, SNR,
    so a second merger-time mode is visible to the flow
  - per detector at the main peak: arrival time relative to it, SNR, phase
  - per detector pair (L1-H1, V1-H1): delay and phase difference
These go through their own MLP into N_TRIG_TOKENS tokens.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SR, T_LEN = 4096, 16384
W = 8                    # profile half-window (grid points)
COARSE = 10              # coarse search: every 10th grid chirp mass (multi-q banks)
FINE = 12                # fine search: +-12 grid points around the coarse best
COINC_S = 0.012          # detectors must agree on arrival time within +-12 ms
SECOND_PEAK_EXCL = 0.03  # a second peak must be > 3% away in chirp mass
TRIG_OS = 8              # trigger features: lag oversampling (4 s / (8 x 4016) = 0.12 ms)
TRIG_SECOND_S = 0.05     # a second merger-time peak must be > 50 ms away
N_TRIG = 4 + 3 * 4 + 2 * 3
N_TRIG_TOKENS = 2
CHUNK_ELEMS = 4e7         # complex elements per matched-filter chunk (GPU memory vs speed)
FEAT_VERSION = 3          # feature definition of new models (see __init__)


def fast_len(n: int) -> int:
    """Smallest 2^a 3^b >= n: an FFT length cuFFT runs without Bluestein."""
    best = 1 << (n - 1).bit_length()
    p3 = 1
    while p3 < best:
        p2 = p3
        while p2 < n:
            p2 *= 2
        best = min(best, p2)
        p3 *= 3
    return best


def circ_maxpool(a: torch.Tensor, w: int) -> torch.Tensor:
    """Max over [i-w, i+w] along the last (circular) axis: the same values as
    max_pool1d on the circularly padded tensor, by doubling (log2(2w+1)
    passes instead of 2w+1 reads per output)."""
    k = 2 * w + 1
    x = torch.cat([a[..., -w:], a, a[..., :w]], dim=-1) if w else a
    m, span = x, 1
    while 2 * span <= k:
        m = torch.maximum(m[..., :-span], m[..., span:])
        span *= 2
    L = a.shape[-1]
    return torch.maximum(m[..., :L], m[..., k - span:k - span + L])


class McFeatures(nn.Module):
    def __init__(self, bank: str, band_lo: int, n_band: int, lowrate_len: int = 0,
                 d_model: int = 192, n_tokens: int = 2):
        super().__init__()
        # "<bank>.npz[:opt,...]": options change the module's parameters, so they
        # travel with the bank name that every checkpoint load site passes on
        bank, _, opts = str(bank).partition(":")
        self.trigger = "trigger" in opts.split(",")
        # "flowfeat": the per-(q, spin) bank peaks also go straight to the flow
        self.flowfeat = "flowfeat" in opts.split(",")
        # "scale": the pivot also rescales chirp mass by the bank's width
        self.scale = "scale" in opts.split(",")
        path = Path(bank)
        if not path.is_absolute() and path.parent == Path("."):
            path = Path(__file__).parent / "assets" / path
        b = np.load(path)
        assert int(b["band_lo"]) == band_lo and int(b["n_band"]) == n_band, \
            "mc bank band does not match the encoder band; rebuild with build_mc_bank.py"
        self.n_q = int(b["n_q"])
        # (q, chi) of each slot of a chirp-mass group: labels are (mc, q, chi)
        self.slot_q_chi = np.asarray(b["mc_q"])[: self.n_q, 1:3].astype(np.float32)
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
            # matched filter over the context's 20-256 Hz bins only (baseband)
            self.ctx_lo = int(round(20.0 * self.ctx_len / 512))
        self.band_lo, self.n_band = band_lo, n_band
        # coarse step: a BNS chirp-mass peak (~0.05% wide) is far narrower
        # than a 10-point coarse step (2%), so the coarse pass would miss it;
        # single-mass-ratio banks are small enough to search in full
        self.coarse = 1 if self.n_q == 1 else COARSE
        # detector time coincidence (see _peak_snr2). Measured on 700 BBH
        # events (spin bank): SNR 6-8 median chirp-mass error 16% -> 4.7% in
        # Gaussian noise and 21% -> 19% (within 5%: 19% -> 30%) in real noise,
        # no change at SNR >= 8. Stored in the state dict so a checkpoint keeps
        # the feature definition it was trained with; checkpoints from before
        # the switch existed load with it off.
        self.register_buffer("coinc_flag", torch.tensor(1.0))
        # feature definition version, stored with the checkpoint. 2 fixes three
        # bugs of version 1 (kept bit-for-bit for checkpoints trained with it):
        #  - window-only / context-only peaks read the COMBINED coarse values
        #    at coarse grid points (multi-rate models)
        #  - local-profile and peak-interpolation reads could fall outside the
        #    fine window onto "not evaluated" (-1) or coarse values
        #  - the coincidence max-filter did not wrap around the circular lag
        #    axis (context-stream peaks sit at lag ~0)
        # 3 (speed, 2026-11-01): lag axes zero-padded to fast FFT lengths
        # (4016 = 16 x 251 ran cuFFT's Bluestein path, ~3x slower), so lags are
        # sampled ~2% more densely; the coincidence max-filter by doubling
        # (identical values, ~6 passes instead of one read per window lag).
        # Measured per step (B=128, H100): NSBH spin bank 515 -> 248 ms, BNS
        # 237 -> 124 ms; BNS chirp-mass error in real noise unchanged (0.038%).
        self.register_buffer("feat_version", torch.tensor(float(FEAT_VERSION)))
        self._sync_flags()
        self.n_features = 2 + (2 * W + 1) + 2 + 4 + (2 * self.n_q if self.n_q > 1 else 0)
        self.mlp = nn.Sequential(nn.LayerNorm(self.n_features), nn.Linear(self.n_features, 128), nn.GELU(),
                                 nn.Linear(128, 128), nn.GELU())
        self.n_tokens = int(n_tokens)
        self.to_tokens = nn.Linear(128, self.n_tokens * d_model)
        self.type_embed = nn.Parameter(torch.zeros(d_model))     # zero-init marker
        if self.trigger:
            self.trig_mlp = nn.Sequential(nn.LayerNorm(N_TRIG), nn.Linear(N_TRIG, 128), nn.GELU(),
                                          nn.Linear(128, 128), nn.GELU())
            self.trig_to_tokens = nn.Linear(128, N_TRIG_TOKENS * d_model)
            self.trig_type_embed = nn.Parameter(torch.zeros(d_model))

    def _peak_snr2(self, d: torch.Tensor, H: torch.Tensor, n: int, scale: float,
                   duration: float) -> torch.Tensor:
        """Network peak SNR^2 per template. d [B,D,F] one-sided spectrum;
        H [D,K,F] shared templates or [B,D,K,F] per-event templates (unit
        norm); complex ifft over n points spanning `duration` s (phase-
        maximized matched filter) -> [B, K].

        Detectors must agree on the arrival time: each detector's |z|^2 is
        max-filtered over +-COINC_S (light travel time plus margin) and the
        network sum is maximized over one common time. Maximizing every
        detector over all lags independently lets noise peaks at unrelated
        times add up, which is what made quiet events (SNR < 8) lose to noise."""
        per_event = H.dim() == 4
        B, Dn, F = d.shape
        K = H.shape[-2]
        if self._v3:
            # zero-pad the lag axis to a fast FFT length: same correlation,
            # sampled on a finer lag grid; scale^2 applied after the maxima
            nf = fast_len(n)
            scale2 = (scale * nf / n) ** 2
            n = nf
        chunk = max(1, int(CHUNK_ELEMS / max(B * Dn * max(n, F), 1)))
        w = max(1, int(round(COINC_S * n / duration)))
        out = []
        for k0 in range(0, K, chunk):
            h = H[..., k0:k0 + chunk, :].conj()
            prod = d.unsqueeze(2) * (h if per_event else h.unsqueeze(0))
            if self._v3:
                a = torch.fft.ifft(prod, n=n, dim=-1).abs().square_()            # [B,D,kc,n]
                if self._coinc:
                    a = circ_maxpool(a, w)
                    out.append(a.sum(1).amax(-1) * scale2)
                else:
                    out.append(a.amax(-1).sum(1) * scale2)
                continue
            a = (torch.fft.ifft(prod, n=n, dim=-1).abs() * scale) ** 2          # [B,D,kc,n]
            if self._coinc:
                kc = a.shape[2]
                a = a.reshape(-1, 1, n)
                if self._v2:              # circular lag axis: wrap, then pool
                    a = torch.nn.functional.max_pool1d(
                        torch.nn.functional.pad(a, (w, w), mode="circular"), 2 * w + 1, stride=1)
                else:
                    a = torch.nn.functional.max_pool1d(a, 2 * w + 1, stride=1, padding=w)
                a = a.reshape(B, Dn, kc, n)
                out.append(a.sum(1).amax(-1))                                   # [B, kc]
            else:
                out.append(a.amax(-1).sum(1))
        return torch.cat(out, dim=1)

    def _sync_flags(self):
        """Python copies of the stored switches: reading a CUDA buffer inside
        the matched-filter loop would force a device sync per chunk."""
        self._coinc = bool(self.coinc_flag)
        self._v2 = float(self.feat_version) >= 2
        self._v3 = float(self.feat_version) >= 3

    @property
    def coincident(self) -> bool:
        return self._coinc

    @coincident.setter
    def coincident(self, on: bool):
        self.coinc_flag.fill_(1.0 if on else 0.0)
        self._sync_flags()

    def reset_definition(self):
        """Current feature definition (coincidence on, version 2): call after a
        partial load (--init_from) that may have carried an older one."""
        self.coinc_flag.fill_(1.0)
        self.feat_version.fill_(float(FEAT_VERSION))
        self._sync_flags()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # checkpoints from before a switch existed keep the definition they
        # were trained with: no coincidence, feature version 1
        for key, legacy in (("coinc_flag", 0.0), ("feat_version", 1.0)):
            if prefix + key not in state_dict:
                state_dict[prefix + key] = torch.tensor(legacy)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        self._coinc = bool(state_dict[prefix + "coinc_flag"])
        self._v2 = float(state_dict[prefix + "feat_version"]) >= 2
        self._v3 = float(state_dict[prefix + "feat_version"]) >= 3

    def _profiles(self, hi: torch.Tensor, lo: torch.Tensor = None):
        """Coarse-to-fine chirp-mass profile. Coarse: every COARSE-th chirp
        mass (all mass ratios), plain lag resolution. Fine: the 2*FINE+1 grid
        chirp masses around the best coarse one, 2x oversampled window lags.
        Evaluating every template at full resolution costs ~16x more (2.2 s
        vs ~0.24 s per training step for the NSBH bank).
        Returns (prof [B,n_mc], window part, context part, per-q fine profile
        [B,n_mc,n_q]); grid points never evaluated hold -1."""
        B, dev = hi.shape[0], hi.device
        d = torch.fft.rfft(hi.float(), dim=-1)[..., self.band_lo:self.band_lo + self.n_band]
        Hw = torch.complex(self.win_re, self.win_im)                     # [D,K,F]
        sw1 = self.n_band / math.sqrt(T_LEN / 2.0)
        use_c = self.use_ctx and lo is not None
        if use_c:
            L = self.ctx_len
            dc = torch.fft.rfft(lo.float(), dim=-1)[..., self.ctx_lo:]
            Hc = torch.complex(self.ctx_re, self.ctx_im)[..., self.ctx_lo:]
            nc = dc.shape[-1]
            sc = nc / math.sqrt(L / 2.0)
        nq = self.n_q
        # ---- coarse ----
        cidx = torch.arange(0, self.n_mc, self.coarse, device=dev)
        tidx = (cidx[:, None] * nq + torch.arange(nq, device=dev)).reshape(-1)
        cw = self._peak_snr2(d, Hw[:, tidx], self.n_band, sw1, T_LEN / SR).reshape(B, -1, nq)
        cc = (self._peak_snr2(dc, Hc[:, tidx], nc, sc, L / 512.0).reshape(B, -1, nq)
              if use_c else torch.zeros_like(cw))
        coarse = (cw + cc).amax(-1)                                       # [B, n_coarse]
        center = cidx[coarse.argmax(-1)]                                  # [B]
        # ---- fine window around it ----
        start = (center - FINE).clamp(0, max(self.n_mc - (2 * FINE + 1), 0))
        midx = start[:, None] + torch.arange(min(2 * FINE + 1, self.n_mc), device=dev)   # [B, M]
        fidx = (midx[:, :, None] * nq + torch.arange(nq, device=dev)).reshape(B, -1)     # [B, M*nq]
        gw = Hw[:, fidx].permute(1, 0, 2, 3)                              # [B,D,M*nq,F]
        fw = self._peak_snr2(d, gw, 2 * self.n_band, 2 * sw1, T_LEN / SR).reshape(B, -1, nq)
        if use_c:
            fc = self._peak_snr2(dc, Hc[:, fidx].permute(1, 0, 2, 3), nc, sc, L / 512.0).reshape(B, -1, nq)
        else:
            fc = torch.zeros_like(fw)
        # scatter onto the grid: fine values in the window, coarse elsewhere
        def grid(fine_vals, coarse_vals):
            g = torch.full((B, self.n_mc), -1.0, device=dev)
            g[:, cidx] = coarse_vals
            return g.scatter(1, midx, fine_vals)
        # v1 seeded every profile with the combined coarse values
        cw_, cc_ = (cw.amax(-1), cc.amax(-1)) if self._v2 else (coarse, coarse)
        prof_q = torch.full((B, self.n_mc, nq), -1.0, device=dev)
        prof_q.scatter_(1, midx[:, :, None].expand(-1, -1, nq), fw + fc)
        self._win = (midx[:, 0], midx[:, -1])          # fine-window bounds per event
        return (grid((fw + fc).amax(-1), coarse), grid(fw.amax(-1), cw_),
                grid(fc.amax(-1), cc_) if use_c else None, prof_q)

    def _peak(self, prof: torch.Tensor):
        """Interpolated peak of a profile over the chirp-mass grid [B, n_mc]:
        (normalized log Mc, asinh peak SNR, argmax index)."""
        k = prof.argmax(-1).clamp(1, self.n_mc - 2)
        g = lambda o: prof.gather(1, (k + o).unsqueeze(1)).squeeze(1)
        y0, y1, y2 = g(-1), g(0), g(1)
        den = y0 - 2 * y1 + y2
        ok = den < 0
        if self._v2 and getattr(self, "_win", None) is not None:
            # both neighbours must be fine-window values, not coarse/-1 fill
            lo_k, hi_k = self._win
            ok = ok & (k - 1 >= lo_k) & (k + 1 <= hi_k)
        dk = torch.where(ok, 0.5 * (y0 - y2) / den.clamp(max=-1e-9), torch.zeros_like(den)).clamp(-1, 1)
        log_mc = self.log_grid[k] + dk * self.dlog
        return self._norm(log_mc), torch.asinh(y1.clamp_min(0).sqrt()), k

    def _norm(self, log_mc):
        return 2 * (log_mc - self.log_lo) / (self.log_hi - self.log_lo) - 1

    def forward(self, hi: torch.Tensor, lo: torch.Tensor = None) -> torch.Tensor:
        B = hi.shape[0]
        prof, pw, pc, prof_q = self._profiles(hi, lo)
        x_mc, x_snr, k = self._peak(prof)
        # the bank's chirp-mass estimate, for LeanNPE's pivot (log Mc)
        self.last_log_mc = ((x_mc + 1) / 2 * (self.log_hi - self.log_lo) + self.log_lo).detach()
        self.last_t = None
        # profile around the peak, relative to it (replicate-padded at the edges)
        idx = (k.unsqueeze(1) + torch.arange(-W, W + 1, device=hi.device)).clamp(0, self.n_mc - 1)
        if self._v2:
            lo_k, hi_k = self._win            # replicate the window edge, never read fill values
            idx = torch.maximum(torch.minimum(idx, hi_k.unsqueeze(1)), lo_k.unsqueeze(1))
        local = prof.gather(1, idx) / prof.gather(1, k.unsqueeze(1)).clamp_min(1e-6)
        # strongest separate peak
        excl = max(1, int(round(SECOND_PEAK_EXCL / self.dlog)))
        far = (torch.arange(self.n_mc, device=hi.device).unsqueeze(0) - k.unsqueeze(1)).abs() > excl
        p2 = torch.where(far, prof, torch.full_like(prof, -1.0))
        k2 = p2.argmax(-1)
        h2 = (p2.gather(1, k2.unsqueeze(1)).squeeze(1).clamp_min(0)
              / prof.gather(1, k.unsqueeze(1)).squeeze(1).clamp_min(1e-6)).sqrt()
        x2 = self._norm(self.log_grid[k2]) - x_mc
        # chirp-mass width (log units) for the scale-adaptive pivot: 3x the
        # profile width from its curvature (ln L ~ SNR^2 / 2), widened to
        # cover a plausible second peak (SNR^2 within 9), floor 1%
        with torch.no_grad():
            kk = k.clamp(1, self.n_mc - 2)
            g = lambda o: prof.gather(1, (kk + o).unsqueeze(1)).squeeze(1)
            curv = (g(-1) - 2 * g(0) + g(1)).clamp(max=-1e-6)
            sig = torch.sqrt(-2.0 / curv) * self.dlog
            snr2 = g(0).clamp_min(1e-6)
            second = (h2 ** 2 * snr2 >= snr2 - 9.0).float() * x2.abs() * (self.log_hi - self.log_lo) / 2
            self.last_log_mc_scale = torch.maximum(torch.maximum(3 * sig, second),
                                                   torch.full_like(sig, 0.01)).detach()
        w_mc, w_snr, _ = self._peak(pw)
        if pc is not None:
            c_mc, c_snr, _ = self._peak(pc)
        else:
            c_mc, c_snr = torch.zeros_like(w_mc), torch.zeros_like(w_snr)
        parts = [x_mc[:, None], x_snr[:, None], local, h2[:, None], x2[:, None],
                 w_mc[:, None], w_snr[:, None], c_mc[:, None], c_snr[:, None]]
        if self.n_q > 1:
            slot = []
            for j in range(self.n_q):
                q_mc, q_snr, _ = self._peak(prof_q[..., j])
                parts += [q_mc[:, None], q_snr[:, None]]
                slot += [q_mc, q_snr]
            # per-slot (normalized log Mc, asinh SNR): which mass ratio / spin
            # fits best, and how much better than the others (branch evidence)
            self.last_slot = torch.stack(slot, dim=1).detach()
        f = torch.cat(parts, dim=1)
        tok = self.to_tokens(self.mlp(f)).reshape(B, self.n_tokens, -1) + self.type_embed
        if self.trigger:
            tf = self._trigger(hi, prof, prof_q)
            ttok = self.trig_to_tokens(self.trig_mlp(tf)).reshape(B, N_TRIG_TOKENS, -1)
            tok = torch.cat([tok, ttok + self.trig_type_embed], dim=1)
        return tok

    @torch.no_grad()
    def _trigger(self, hi: torch.Tensor, prof: torch.Tensor, prof_q: torch.Tensor):
        """Arrival times, SNRs and phases of the best template [B, N_TRIG]
        (see module docstring), from the window stream."""
        B, dev = hi.shape[0], hi.device
        nb = self.n_band
        n = fast_len(TRIG_OS * nb) if self._v3 else TRIG_OS * nb
        k = prof.argmax(-1)
        t = k * self.n_q + prof_q[torch.arange(B, device=dev), k].argmax(-1)
        H = torch.complex(self.win_re, self.win_im)[:, t].permute(1, 0, 2)          # [B,D,F]
        d = torch.fft.rfft(hi.float(), dim=-1)[..., self.band_lo:self.band_lo + nb]
        z = torch.fft.ifft(d * H.conj(), n=n, dim=-1) * (n / math.sqrt(T_LEN / 2.0))  # [B,D,n]
        a2 = z.abs() ** 2
        w = max(1, int(round(COINC_S * n * SR / T_LEN)))
        pooled = torch.nn.functional.max_pool1d(
            torch.nn.functional.pad(a2.reshape(-1, 1, n), (w, w), mode="circular"),
            2 * w + 1, stride=1).reshape(a2.shape).sum(1)                           # [B,n] network SNR^2
        dt = T_LEN / SR / n                                                         # s per lag
        c1 = pooled.argmax(-1)
        lag = torch.arange(n, device=dev)
        sep = ((lag[None] - c1[:, None]) % n).clamp(max=n // 2)
        sep = torch.minimum(sep, (c1[:, None] - lag[None]) % n)
        c2 = torch.where(sep > TRIG_SECOND_S / dt, pooled, torch.zeros_like(pooled)).argmax(-1)
        # the pooled network SNR is flat over a +-w plateau and argmax returns its
        # left edge, so each detector is searched over [c - w, c + 3w]
        span = torch.arange(-w, 3 * w + 1, device=dev)
        net = a2.sum(1)

        def refine(c):                    # unpooled network-SNR peak near c: lag, SNR
            lg = (c[:, None] + span) % n
            v = net.gather(1, lg)
            return lg.gather(1, v.argmax(-1, keepdim=True)).squeeze(1), v.amax(-1).clamp_min(0).sqrt()
        # per detector around the main peak: interpolated arrival lag, SNR, phase
        lags = (c1[:, None] + span) % n                                             # [B,S]
        zz = z.gather(2, lags[:, None, :].expand(-1, z.shape[1], -1))                # [B,D,S]
        mag = zz.abs()
        a = mag.argmax(-1).clamp(1, span.numel() - 2)
        y0, y1, y2 = (mag.gather(2, (a + o)[..., None]).squeeze(-1) for o in (-1, 0, 1))
        dk = (0.5 * (y0 - y2) / (y0 - 2 * y1 + y2).clamp(max=-1e-9)).clamp(-1, 1)
        arr = (c1[:, None] - w + a).float() + dk                                     # arrival lag per detector
        wgt = y1 ** 2 / (y1 ** 2).sum(-1, keepdim=True).clamp_min(1e-9)
        ref = (wgt * arr).sum(-1)                                                    # SNR^2-weighted arrival
        zc = zz.gather(2, a[..., None]).squeeze(-1)
        # baseband ifft: undo the band-start frequency shift to get the phase
        phi = torch.angle(zc) + 2 * math.pi * self.band_lo * ((c1[:, None] - w + a) % n).float() / n
        tau = (arr - ref[:, None]) * dt / COINC_S                                    # in units of +-12 ms
        # lag -> merger time in s from the window centre: the template's merger
        # sits at lag 0 of its circular frame, i.e. at the window start
        tm = lambda lag: torch.remainder(lag * dt, T_LEN / SR) - T_LEN / SR / 2
        l2, s2 = refine(c2)
        self.last_t = tm(ref).detach()       # merger-time estimate (s), for the pivot
        self.last_t_h1 = tm(arr[:, 0]).detach()   # arrival at H1 (sky_frame pivot)
        feats = [tm(ref) / 1.5, torch.asinh((y1 ** 2).sum(-1).sqrt()), tm(l2.float()) / 1.5, torch.asinh(s2)]
        for j in range(z.shape[1]):
            feats += [tau[:, j], torch.asinh(y1[:, j]), torch.cos(phi[:, j]), torch.sin(phi[:, j])]
        for j in (1, 2):
            feats += [tau[:, j] - tau[:, 0], torch.cos(phi[:, j] - phi[:, 0]), torch.sin(phi[:, j] - phi[:, 0])]
        return torch.stack(feats, dim=1)

    @torch.no_grad()
    def estimate(self, hi: torch.Tensor, lo: torch.Tensor = None) -> torch.Tensor:
        """Peak chirp mass [B] (diagnostics/tests)."""
        x, _, _ = self._peak(self._profiles(hi, lo)[0])
        return torch.exp((x + 1) / 2 * (self.log_hi - self.log_lo) + self.log_lo)
