"""M7 - WHY GW150914/GW170814 specifically? Separate the two causal factors.

Degradation = (global overconfidence, grows with SNR) x (event-specific bias).
Event-specific bias has two sources we can separate:
  (a) MASS/RATIO coverage: sparse heavy + near-equal-mass corner -> biased flow.
  (b) PSD-OOD: real detector ASDs differ from training (Step 4).

Design-whitened VALIDATION events have asd_bands=0 => ZERO PSD-OOD. So the
validation map isolates factor (a). We build base-space concentration ||z||^2/D
and spurious railing as functions of (SNR, Mc, q) on validation, find the worst
corner, and check that GW150914/GW170814 sit in it while GW170729 / the light
events do not. Then we quantify the PSD-OOD add-on by comparing each real
event's railing to the validation railing at the SAME (Mc,q,SNR).
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis/divergence_localization"))
from _events import EVENTS, PROBLEM, chirp_mass  # noqa: E402
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402
from ahsd.inference.pipeline import load_model  # noqa: E402
from experiments.remix_data import RemixDataset  # noqa: E402

HERE = Path(__file__).resolve().parent
VCACHE = "data/dataset/memmap/validation"
MODEL = "model/long_run/best_model.pth"
N = 4000
K = 120
D = len(PARAM_NAMES)
M1I, M2I, DI = 0, 1, 2


@torch.no_grad()
def val_map(net, dev):
    ds = RemixDataset(VCACHE, remix=False, seed=1234, return_asd_bands=net.psd_cond)
    n = min(N, len(ds))
    noncirc = (~net.scaler.circ_mask).cpu().numpy()
    Mc = np.empty(n); q = np.empty(n); snr = np.empty(n)
    bconc = np.empty(n); rail = np.empty(n)
    B = 128
    for s in range(0, n, B):
        idx = list(range(s, min(s + B, n)))
        st, tr, ns, ab = [], [], [], []
        for i in idx:
            item = ds[i]
            if net.psd_cond:
                a, pv, ng, sn, band = item; ab.append(band.numpy())
            else:
                a, pv, ng, sn = item
            st.append(a.numpy()); tr.append(pv.numpy()[0]); ns.append(float(sn))
        st = torch.from_numpy(np.stack(st)).to(dev).float()
        tr = np.stack(tr)
        trt = torch.from_numpy(tr).to(dev).float()
        abt = torch.from_numpy(np.stack(ab)).to(dev).float() if net.psd_cond else None
        r = torch.zeros(len(idx), dtype=torch.long, device=dev)
        ctx = net.encode(st, abt)
        full = net._full_context(ctx, r)
        # base-space concentration of the TRUTH
        yt = net.scaler.normalize(trt)
        z, _ = net.flow.forward(yt, full)
        bc = (z.pow(2).sum(1) / D).cpu().numpy()
        # spurious railing
        crep = full.unsqueeze(1).expand(len(idx), K, full.shape[1]).reshape(-1, full.shape[1])
        zz = torch.randn(len(idx) * K, D, device=dev)
        yy, _ = net.flow.inverse(zz, crep)
        yw = net.scaler.wrap(yy).reshape(len(idx), K, D).cpu().numpy()
        ytn = yt.unsqueeze(1).cpu().numpy()
        spur = ((yw > 0.999) & (ytn < 0.9)) | ((yw < -0.999) & (ytn > -0.9))
        rb = spur[:, :, noncirc].any(-1).mean(1)
        for jj, i in enumerate(idx):
            Mc[i] = chirp_mass(tr[jj, M1I], tr[jj, M2I]); q[i] = tr[jj, M2I] / tr[jj, M1I]
            snr[i] = ns[jj]; bconc[i] = bc[jj]; rail[i] = rb[jj]
    return dict(Mc=Mc, q=q, snr=snr, bconc=bconc, rail=rail, n=n)


def binned(x, y, edges, label):
    print(f"\n  {label:24s} " + "".join(f"{f'[{edges[i]:g},{edges[i+1]:g})':>12}"
                                        for i in range(len(edges) - 1)))
    concs, rails = [], []
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x < edges[i + 1])
        concs.append(np.nan if m.sum() < 15 else float(np.mean(y[0][m])))
        rails.append(np.nan if m.sum() < 15 else float(np.mean(y[1][m])))
    print(f"  {'base_conc ||z||^2/D':24s} " + "".join(f"{c:12.2f}" for c in concs))
    print(f"  {'spurious railing':24s} " + "".join(f"{r:12.2f}" for r in rails))
    return concs, rails


def main():
    net, meta = load_model(MODEL)
    dev = meta["device"]
    v = val_map(net, dev)
    print(f"=== M7: causal selectivity — validation ({v['n']} events, ZERO PSD-OOD) ===")

    out = {"one_way": {}}
    out["one_way"]["SNR"] = binned(v["snr"], (v["bconc"], v["rail"]),
                                   [5, 12, 18, 25, 40], "by SNR")
    out["one_way"]["Mc"] = binned(v["Mc"], (v["bconc"], v["rail"]),
                                  [5, 12, 20, 30, 60], "by chirp mass")
    out["one_way"]["q"] = binned(v["q"], (v["bconc"], v["rail"]),
                                 [0.1, 0.4, 0.6, 0.8, 1.01], "by mass ratio q")

    # 2D (Mc high/low) x (q high/low), at matched high SNR (>18) — isolate the corner
    print("\n  --- 2D at SNR>18: base_conc / railing in (Mc, q) quadrants ---")
    hs = v["snr"] > 18
    quad = {}
    for mlab, mmask in [("Mc<20", v["Mc"] < 20), ("Mc>=20", v["Mc"] >= 20)]:
        for qlab, qmask in [("q<0.7", v["q"] < 0.7), ("q>=0.7", v["q"] >= 0.7)]:
            m = hs & mmask & qmask
            if m.sum() >= 15:
                quad[f"{mlab},{qlab}"] = dict(n=int(m.sum()),
                                              base_conc=float(v["bconc"][m].mean()),
                                              rail=float(v["rail"][m].mean()))
    for k, val in quad.items():
        print(f"    {k:18s} n={val['n']:5d}  base_conc={val['base_conc']:.2f}  "
              f"railing={val['rail']:.2f}")
    out["quadrants_snr_gt18"] = quad

    # locate the 6 real events + their measured ep19 railing (from events_ep19.json)
    ev19 = json.load(open(HERE.parent / "events_ep19.json"))
    print("\n  --- real events located on the map (measured ep19 railing) ---")
    print(f"    {'event':10s} {'Mc':>5} {'q':>5} {'SNR':>5} {'railing':>8} "
          f"{'corner':>14}")
    ev_out = {}
    for name, e in EVENTS.items():
        Mc = chirp_mass(e["m1"], e["m2"]); q = e["m2"] / e["m1"]
        corner = f"{'Mc>=20' if Mc>=20 else 'Mc<20'},{'q>=0.7' if q>=0.7 else 'q<0.7'}"
        rl = ev19.get(name, {}).get("rail", float("nan"))
        flag = " <==" if name in PROBLEM else ""
        print(f"    {name:10s} {Mc:5.1f} {q:5.2f} {e['snr']:5.1f} {rl:8.2f} "
              f"{corner:>14}{flag}")
        ev_out[name] = dict(Mc=Mc, q=q, snr=e["snr"], rail=rl, corner=corner)
    out["events"] = ev_out
    json.dump(out, open(HERE / "m7_event_selectivity.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
