"""M17 - ENCODER or FLOW? Linear probe for mass ratio on the context vector.

Established finding: the model reports q~0.99 for heavy (m1>~29) non-edge-on
systems regardless of the true q (m15/m16), it is reproducible on pure simulation
(m14 rung 1), and it is NOT a coverage effect (the collapse occurs in the four
best-covered theta_jn bins, ~10k events each). Dynesty recovers q=0.89 from the
same data, so the information is in the strain.

Question: is q information ABSENT from the encoder context (encoder bottleneck --
consistent with the known 65 ms temporal stride), or PRESENT but discarded by the
flow (flow/representation problem)?

Method (this project's own post-mortem methodology): ridge-regression linear probe
context -> q, fit and scored on held-out data, split by mass regime. Compare
against the flow's own sampled q for the same events.

  probe R^2 ~ 0 on heavy -> encoder never extracts q  -> encoder is the bottleneck
  probe R^2 high on heavy -> info is there; flow collapses -> flow/representation
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from ahsd.inference.pipeline import load_model  # noqa: E402
from experiments.remix_data import RemixDataset  # noqa: E402

HERE = Path(__file__).resolve().parent
N, K = 4000, 100


def chirp(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


@torch.no_grad()
def collect(net, dev):
    ds = RemixDataset("data/dataset/memmap/validation", remix=False, seed=1234,
                      return_asd_bands=net.psd_cond)   # design/Gaussian: cleanest
    n = min(N, len(ds))
    C = np.empty((n, net.context_dim), np.float32)
    T = np.empty((n, 11)); snr = np.empty(n); qflow = np.empty(n)
    B = 128
    for s in range(0, n, B):
        idx = list(range(s, min(s + B, n)))
        st, tr, ab, sv = [], [], [], []
        for i in idx:
            it = ds[i]
            if net.psd_cond:
                a, pv, ng, x, band = it; ab.append(band.numpy())
            else:
                a, pv, ng, x = it
            st.append(a.numpy()); tr.append(pv.numpy()[0]); sv.append(float(x))
        stt = torch.from_numpy(np.stack(st)).to(dev).float()
        abt = torch.from_numpy(np.stack(ab)).to(dev).float() if net.psd_cond else None
        ctx = net.encode(stt, abt)                       # <-- the probe target
        full = net._full_context(ctx, torch.zeros(len(idx), dtype=torch.long, device=dev))
        rep = full.unsqueeze(1).expand(len(idx), K, full.shape[1]).reshape(-1, full.shape[1])
        y, _ = net.flow.inverse(torch.randn(len(idx) * K, 11, device=dev), rep)
        smp = net.scaler.denormalize(net.scaler.wrap(y)).reshape(len(idx), K, 11).cpu().numpy()
        a_ = np.maximum(smp[:, :, 0], smp[:, :, 1]); b_ = np.minimum(smp[:, :, 0], smp[:, :, 1])
        cc = ctx.cpu().numpy()
        for jj, i in enumerate(idx):
            C[i] = cc[jj]; T[i] = tr[jj]; snr[i] = sv[jj]
            qflow[i] = float(np.median(b_[jj] / a_[jj]))
    return C, T, snr, qflow


def probe(X, y, seed=0):
    """Ridge probe with a held-out split; returns test R^2."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); ntr = int(0.7 * len(X))
    tr, te = idx[:ntr], idx[ntr:]
    Xtr, Xte = X[tr], X[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))]); Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    lam = 1.0
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ y[tr])
    pred = Xte @ w
    ss_res = float(((y[te] - pred) ** 2).sum()); ss_tot = float(((y[te] - y[te].mean()) ** 2).sum())
    return 1 - ss_res / ss_tot


def main():
    net, meta = load_model("model/lean_npe/best_model.pth"); dev = meta["device"]
    print(f"model epoch {meta['model_epoch']}  (Gaussian domain, n={N})\n")
    C, T, snr, qflow = collect(net, dev)
    Mc = chirp(T[:, 0], T[:, 1]); qt = T[:, 1] / T[:, 0]

    print("Linear probe on the ENCODER CONTEXT (held-out R^2), vs the FLOW's own q:\n")
    print(f"{'regime':30} {'n':>5} {'probe R2(q)':>12} {'probe R2(Mc)':>13} "
            f"{'flow q err':>11} {'flow q':>7} {'true q':>7}")
    groups = [("light  m1<20", T[:, 0] < 20),
              ("mid    20<=m1<29", (T[:, 0] >= 20) & (T[:, 0] < 29),),
              ("HEAVY  m1>=29", T[:, 0] >= 29),
              ("HEAVY & loud (SNR>18)", (T[:, 0] >= 29) & (snr > 18))]
    out = {}
    for lab, m in groups:
        if m.sum() < 50:
            continue
        r2q = probe(C[m], qt[m]); r2mc = probe(C[m], np.log(Mc[m]))
        err = float(np.median(np.abs(qflow[m] - qt[m])))
        out[lab] = dict(n=int(m.sum()), r2_q=r2q, r2_logMc=r2mc, flow_q_abs_err=err)
        print(f"{lab:30} {int(m.sum()):5d} {r2q:12.3f} {r2mc:13.3f} {err:11.3f} "
              f"{np.median(qflow[m]):7.3f} {np.median(qt[m]):7.3f}")
    print("\n  probe R2(q) ~0 on HEAVY  -> encoder never extracts q (encoder bottleneck)")
    print("  probe R2(q) high on HEAVY -> info present, flow collapses it")
    print("  (R2(log Mc) is the control: chirp mass IS recovered well everywhere)")
    json.dump(out, open(HERE / "m17_q_probe.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
