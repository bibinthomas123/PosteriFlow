#!/usr/bin/env python3
"""Chirp-mass error floor of the matched-filter bank, and whether sub-grid
estimates recover it.

For held-out single signals (validation split: signal + one noise draw) and a
model's chirp-mass bank, five raw ln Mc estimates (no network):

  max      interpolated peak of the max-over-(q, chi)-slots SNR^2 profile
           (what the encoder's coarse "best match" feature carries)
  marg     interpolated peak of logsumexp over slots of SNR^2 / 2
  softavg  per-slot peak ln Mc averaged with weights exp(SNR^2_slot / 2)
  subgrid  2-D quadratic fit of the per-slot peak SNR^2 over the 3 x 3 slot
           neighbourhood of the best slot (in slot-index units, ln q x chi),
           its maximum (clamped to +-1 slot), and the per-slot peak ln Mc
           interpolated (biquadratic) there
  oracle   the per-slot peak ln Mc of the slot NEAREST the truth's (q,
           chi_eff): a ceiling for any slot-selection / interpolation scheme

Reported per SNR bin: median |err ln Mc| x SNR (flat in SNR = the estimator
keeps up with the information; rising = a floor). Also split by distance of
the truth from its nearest slot.

    python analysis/bank_floor/bank_floor_probe.py --model model/<ab8_dense>/best_model.pth \
        --data data/dataset_bbh_v6 [--n 3000] --out analysis/bank_floor/<name>.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ahsd.inference.pipeline import load_model  # noqa: E402
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402

J = {n: i for i, n in enumerate(PARAM_NAMES)}
BINS = ((8, 12), (12, 20), (20, 35), (35, 60), (60, 1e9))


def peak_interp(prof, valid, log_grid, dlog):
    """Interpolated ln Mc of the maximum of prof [..., n_mc] (parabola through
    the top 3 grid points) and the peak value."""
    p = torch.where(valid, prof, torch.full_like(prof, -1e30))
    n = p.shape[-1]
    k = p.argmax(-1).clamp(1, n - 2)
    g = lambda o: p.gather(-1, (k + o).unsqueeze(-1)).squeeze(-1)  # noqa: E731
    y0, y1, y2 = g(-1), g(0), g(1)
    den = y0 - 2 * y1 + y2
    ok = (den < 0) & (y0 > -1e29) & (y2 > -1e29)
    dk = torch.where(ok, 0.5 * (y0 - y2) / den.clamp(max=-1e-9), torch.zeros_like(den)).clamp(-1, 1)
    peak = torch.where(ok, y1 - 0.25 * (y0 - y2) * dk, y1)
    return log_grid[k] + dk * dlog, peak


def quad9(z):
    """Least-squares 2-D quadratic over a 3x3 stencil at (u, v) in {-1,0,1}^2:
    returns coefficients (c, a_u, a_v, b_uu, b_vv, b_uv) per row of z [N, 9]."""
    u, v = np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij")
    u, v = u.ravel(), v.ravel()
    A = np.stack([np.ones(9), u, v, u * u, v * v, u * v], 1)
    return np.linalg.lstsq(A, z.T, rcond=None)[0].T


def subgrid(snr2, lmc, nq, nc):
    """snr2, lmc: [N, nq*nc] per-slot peak SNR^2 and peak ln Mc (slot =
    qi*nc + ci). Quadratic fit of SNR^2 around the best slot (stencil shifted
    inside the grid at edges), continuous maximum clamped to the stencil, and
    the biquadratic interpolation of ln Mc there."""
    N = len(snr2)
    S, M = snr2.reshape(N, nq, nc), lmc.reshape(N, nq, nc)
    best = snr2.argmax(1)
    bi, bj = best // nc, best % nc
    ci, cj = np.clip(bi, 1, nq - 2), np.clip(bj, 1, nc - 2)
    out = np.empty(N)
    for r in range(N):
        s = S[r, ci[r] - 1:ci[r] + 2, cj[r] - 1:cj[r] + 2].ravel()
        m = M[r, ci[r] - 1:ci[r] + 2, cj[r] - 1:cj[r] + 2].ravel()
        c, au, av, buu, bvv, buv = quad9(s[None])[0]
        H = np.array([[2 * buu, buv], [buv, 2 * bvv]])
        u0 = np.array([bi[r] - ci[r], bj[r] - cj[r]], float)       # best slot in stencil coords
        if np.linalg.det(H) > 0 and H[0, 0] < 0:                    # concave: a real maximum
            x = np.linalg.solve(H, -np.array([au, av]))
        else:
            x = u0
        x = np.clip(x, -1, 1)
        cm = quad9(m[None])[0]
        uu, vv = x
        out[r] = cm[0] + cm[1] * uu + cm[2] * vv + cm[3] * uu * uu + cm[4] * vv * vv + cm[5] * uu * vv
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.model, dev)[0].eval()
    mf = model.encoder.mc_feat
    model_meta = torch.load(a.model, map_location="cpu", weights_only=False)["args"]
    bank_name = str(model_meta.get("mc_bank")).split(":")[0]
    b = np.load(ROOT / "src/ahsd/models/assets" / bank_name)
    lab = b["mc_q"]                                   # (mc, q, chi) per template
    nq_tot = int(b["n_q"])
    slots = lab[:nq_tot, 1:3]                         # (q, chi) of slot j
    qs = np.unique(slots[:, 0])[::-1]
    cs = np.unique(slots[:, 1])
    nq, nc = len(qs), len(cs)
    assert nq * nc == nq_tot, (nq, nc, nq_tot)
    print(f"bank {bank_name}: {nq} q x {nc} chi slots", flush=True)

    c = Path(a.data) / "memmap" / "validation"
    ev = json.load(open(c / "events.json"))["events"]
    sig, noi, par = (np.load(c / f, mmap_mode="r") for f in ("signals.npy", "noise.npy", "params.npy"))
    sel = [(e, s) for e, (s, n) in enumerate(ev) if n == 1][:a.n]

    rows = []
    with torch.no_grad():
        for b0 in range(0, len(sel), a.batch):
            ch = sel[b0:b0 + a.batch]
            x = torch.from_numpy(np.stack([np.asarray(noi[e % len(noi)], np.float32)
                                           + np.asarray(sig[s], np.float32) for e, s in ch])).to(dev)
            out = mf._profiles(x)
            prof_q = out[-1]                                    # [B, n_mc, n_q] SNR^2, -1 = not evaluated
            valid = prof_q > -0.5
            lg, dl = mf.log_grid, mf.dlog
            e_max, _ = peak_interp(prof_q.amax(-1), valid.any(-1), lg, dl)
            lse = torch.logsumexp(torch.where(valid, prof_q / 2, torch.full_like(prof_q, -1e30)), -1)
            e_marg, _ = peak_interp(lse, valid.any(-1), lg, dl)
            lmc_s, pk_s = peak_interp(prof_q.transpose(1, 2), valid.transpose(1, 2), lg, dl)   # [B, n_q]
            lmc_s, pk_s = lmc_s.cpu().numpy(), pk_s.cpu().numpy()
            w = np.exp((pk_s - pk_s.max(1, keepdims=True)) / 2)
            e_soft = (w * lmc_s).sum(1) / w.sum(1)
            e_sub = subgrid(pk_s, lmc_s, nq, nc)
            for i, (_, s) in enumerate(ch):
                t = np.asarray(par[s][:len(PARAM_NAMES)], np.float64)
                m1, m2 = max(t[0], t[1]), min(t[0], t[1])
                a1, a2 = (t[J["a1"]], t[J["a2"]]) if t[0] >= t[1] else (t[J["a2"]], t[J["a1"]])
                q, ce = m2 / m1, (m1 * a1 + m2 * a2) / (m1 + m2)
                lt = np.log((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2)
                d = np.abs(np.log(slots[:, 0]) - np.log(q)) / 0.5 + np.abs(slots[:, 1] - ce) / 0.3
                j = int(d.argmin())
                snr = float(np.sqrt((np.asarray(sig[s], np.float64) ** 2).sum()))
                rows.append((snr, float(e_max[i]) - lt, float(e_marg[i]) - lt, e_soft[i] - lt,
                             e_sub[i] - lt, lmc_s[i, j] - lt, float(d.min()), q, ce))
            if b0 % (a.batch * 20) == 0:
                print(f"  {b0}/{len(sel)}", flush=True)
    r = np.array(rows)
    names = ["max", "marg", "softavg", "subgrid", "oracle"]
    rep = {"model": a.model, "bank": bank_name, "slots": [nq, nc], "n": len(r), "bins": {}}
    print(f"\n{bank_name}: median |err ln Mc| x SNR (flat = no floor); raw % in brackets")
    print(f"{'SNR bin':9s} {'n':>5s} " + " ".join(f"{n:>14s}" for n in names))
    for lo, hi in BINS:
        k = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if k.sum() < 5:
            continue
        s = float(np.median(r[k, 0]))
        v = [float(np.median(np.abs(r[k, 1 + i]))) for i in range(5)]
        rep["bins"][f"{lo}-{hi:g}"] = {"n": int(k.sum()), "snr": s,
                                      **{n: {"err_x_snr": x * s, "raw_pct": 100 * x} for n, x in zip(names, v)}}
        print(f"{lo:>3}-{hi:<5g} {k.sum():5d} " + " ".join(f"{x * s:6.3f} ({100 * x:4.2f})" for x in v))
    # loud events, split by distance of the truth from its nearest slot
    print("\nSNR > 35, by distance of truth from nearest slot (|dln q|/0.5 + |dchi_eff|/0.3):")
    loud = r[:, 0] >= 35
    rep["loud_by_slot_distance"] = {}
    for dlo, dhi in ((0, 0.2), (0.2, 0.4), (0.4, 1e9)):
        k = loud & (r[:, 6] >= dlo) & (r[:, 6] < dhi)
        if k.sum() < 5:
            continue
        s = r[k, 0]
        v = [float(np.median(np.abs(r[k, 1 + i]) * s)) for i in range(5)]
        rep["loud_by_slot_distance"][f"{dlo}-{dhi:g}"] = {"n": int(k.sum()), **dict(zip(names, v))}
        print(f"  d {dlo:.1f}-{dhi:<4g} n={k.sum():4d} " + " ".join(f"{n} {x:.3f}" for n, x in zip(names, v)))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
