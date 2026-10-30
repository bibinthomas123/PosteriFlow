"""M3 (+ M4 param-norms, + clean M1 overconfidence metric).

Model-based flow diagnostics on a COMMON validation batch across the 3 available
checkpoints. The clean overconfidence measure is BASE-SPACE concentration: a
calibrated flow maps truths to z ~ N(0, I), so E[||z||^2/D] = 1 and per-dim
Var(z)=1. Overconfident (posteriors too narrow where the truth sits) => truths
land in the base tails => E[||z||^2/D] > 1. This is immune to the boundary-
railing bimodality that confounds sample variance (see m1).

Also: Jacobian log-det (transform expansion = sharpness), spline saturation
(frac |z|>3 vs tail_bound 5), sampled conditional variance, and encoder-vs-flow
parameter norms (does the flow carry the late capacity?).
"""
import json, sys, math
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis/divergence_localization"))
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402
from ahsd.inference.pipeline import load_model  # noqa: E402
from experiments.remix_data import RemixDataset  # noqa: E402

HERE = Path(__file__).resolve().parent
VCACHE = "data/dataset/memmap/validation"
CKPTS = [("long_run(ep19)", "model/long_run/best_model.pth", 0.864),
         ("coherentC(ep40)", "model/lean_coherentC/best_model.pth", 1.368),
         ("v8_ft(ep25)*", "model/lean_v8_psdcond_ft/best_model.pth", -1.188)]
N = 1500
K = 100
SNR_BINS = [(5, 15), (15, 25), (25, 1e9)]
D = len(PARAM_NAMES)


def load_val_batch(psd_bands=16, n=N):
    ds = RemixDataset(VCACHE, remix=False, seed=1234, return_asd_bands=True,
                      psd_bands=psd_bands)
    n = min(n, len(ds))
    strains, truths, snrs = [], [], []
    for i in range(n):
        st, pv, nsig, ns, ab = ds[i]
        strains.append(st.numpy()); truths.append(pv.numpy()[0]); snrs.append(float(ns))
    return (np.stack(strains), np.stack(truths), np.array(snrs))


def param_norms(model):
    enc, flow, other = 0.0, 0.0, 0.0
    for name, p in model.named_parameters():
        s = float((p.detach() ** 2).sum())
        if name.startswith("encoder"):
            enc += s
        elif name.startswith("flow"):
            flow += s
        else:
            other += s
    return dict(encoder=math.sqrt(enc), flow=math.sqrt(flow), other=math.sqrt(other))


@torch.no_grad()
def internals(model, dev, strain, truth, snr):
    st = torch.from_numpy(strain).to(dev).float()
    tr = torch.from_numpy(truth).to(dev).float()
    ab = torch.zeros(st.shape[0], 3, model.encoder.psd_bands, device=dev) if model.psd_cond else None
    y_true = model.scaler.normalize(tr)
    # encode in chunks
    ctxs = []
    for i in range(0, st.shape[0], 128):
        ctxs.append(model.encode(st[i:i+128], ab[i:i+128] if ab is not None else None))
    ctx = torch.cat(ctxs)
    r = torch.zeros(st.shape[0], dtype=torch.long, device=dev)
    fullc = model._full_context(ctx, r)
    z, logdet = model.flow.forward(y_true, fullc)          # theta_true -> base
    z2 = (z ** 2).sum(1)                                    # ||z||^2 per event
    base_conc = float((z2 / D).mean())
    perdim_var = z.var(0).cpu().numpy()                     # should be ~1
    frac_z_gt3 = float((z.abs() > 3).any(1).float().mean())
    max_absz = float(z.abs().max())
    mean_logdet = float(logdet.mean())
    nll = (0.5 * z2 - logdet + 0.5 * D * math.log(2 * math.pi)).mean().item()

    # sampled conditional variance (scaled), mean over non-nuisance handled elsewhere
    crep = fullc.unsqueeze(1).expand(st.shape[0], K, fullc.shape[1]).reshape(-1, fullc.shape[1])
    zz = torch.randn(st.shape[0] * K, D, device=dev)
    ys, _ = model.flow.inverse(zz, crep)
    ysw = model.scaler.wrap(ys).reshape(st.shape[0], K, D)
    cond_var = ysw.var(1).mean(0).cpu().numpy()             # per-param scaled var

    # SNR-split base concentration
    snr_t = torch.from_numpy(snr)
    snr_split = {}
    for lo, hi in SNR_BINS:
        m = (snr_t >= lo) & (snr_t < hi)
        if m.sum() >= 10:
            snr_split[f"{lo:g}-{hi:g}" if hi < 1e9 else f">{lo:g}"] = \
                float((z2[m.to(dev)] / D).mean())
    return dict(base_conc=base_conc, mean_logdet=mean_logdet, nll=nll,
                frac_z_gt3=frac_z_gt3, max_absz=max_absz,
                perdim_var=perdim_var.tolist(), cond_var=cond_var.tolist(),
                snr_split_base_conc=snr_split)


def main():
    strain, truth, snr = load_val_batch()
    print(f"common validation batch: {strain.shape[0]} events\n")
    out = {}
    print(f"{'checkpoint':16s} {'valNLL':>7} {'baseConc':>9} {'logdet':>8} "
          f"{'recNLL':>7} {'fz>3':>6} {'max|z|':>7} {'enc||':>7} {'flow||':>7} {'f/e':>5}")
    for name, path, valnll in CKPTS:
        model, meta = load_model(path)
        dev = meta["device"]
        pn = param_norms(model)
        it = internals(model, dev, strain, truth, snr)
        out[name] = {"val_nll": valnll, **it, "param_norms": pn}
        print(f"{name:16s} {valnll:>7.3f} {it['base_conc']:>9.3f} "
              f"{it['mean_logdet']:>8.2f} {it['nll']:>7.2f} {it['frac_z_gt3']:>6.2f} "
              f"{it['max_absz']:>7.1f} {pn['encoder']:>7.1f} {pn['flow']:>7.1f} "
              f"{pn['flow']/pn['encoder']:>5.2f}")

    print("\nBASE-SPACE concentration ||z||^2/D by SNR (calibrated=1.0; >1 => "
          "overconfident):")
    for name, _, _ in CKPTS:
        ss = out[name]["snr_split_base_conc"]
        print(f"  {name:16s} " + "  ".join(f"{k}:{v:.2f}" for k, v in ss.items()))

    print("\nPer-dim base variance Var(z_j) (calibrated=1.0), long_run(ep19):")
    lr = out["long_run(ep19)"]
    for j, nm in enumerate(PARAM_NAMES):
        print(f"  {nm:20s} Var(z)={lr['perdim_var'][j]:.2f}  cond_var(scaled)={lr['cond_var'][j]:.3f}")

    json.dump(out, open(HERE / "m3_flow_internals.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
