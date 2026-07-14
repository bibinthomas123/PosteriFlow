"""
Experiment 1 (progressively stronger probes on the frozen final context) +
Experiment 4 (amplitude representation: does the encoder preserve peak
amplitude / per-detector matched-filter SNR / detector energy, or mostly
phase/frequency content?).

Frozen encoder, no training. Reuses the true per-detector SIGNAL-ONLY arrays
(RemixDataset.signals, remix=False so no augmentation) to compute clean
per-detector peak amplitude, energy, and matched-filter SNR (L2 norm of the
whitened signal -- exact for unit-variance whitened data, same formula
remix_data.py itself uses for network_snr), uncontaminated by noise.

Usage:
  python scripts/probe_tiers_and_amplitude.py --model model/lean_v6/best_model.pth \
      --n_events 2500 --outdir analysis/probe_tiers_amplitude
"""
import argparse
import json
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _ROOT)
sys.path.insert(0, f"{_ROOT}/src")
sys.path.insert(0, f"{_ROOT}/experiments")

import numpy as np
import torch
from scipy import stats as sps
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from sklearn.metrics import r2_score

from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES
from remix_data import RemixDataset

JD = PARAM_NAMES.index("luminosity_distance")


class SmallMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.GELU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LargeMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.GELU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_torch_probe(model_cls, X, y, epochs=250, lr=1e-3, wd=1e-2, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.2, random_state=seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr_t = torch.from_numpy(((Xtr - mu) / sd).astype(np.float32))
    Xval_t = torch.from_numpy(((Xval - mu) / sd).astype(np.float32))
    Xte_t = torch.from_numpy(((Xte - mu) / sd).astype(np.float32))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    torch.manual_seed(seed)
    probe = model_cls(X.shape[1])
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=wd)
    best_val_r2, best_state, patience, bad = -np.inf, None, 25, 0
    for ep in range(epochs):
        probe.train(); opt.zero_grad()
        loss = torch.nn.functional.mse_loss(probe(Xtr_t), ytr_t)
        loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            val_r2 = r2_score(yval, probe(Xval_t).numpy())
        if val_r2 > best_val_r2:
            best_val_r2, best_state, bad = val_r2, {k: v.clone() for k, v in probe.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    probe.load_state_dict(best_state); probe.eval()
    with torch.no_grad():
        pred_te = probe(Xte_t).numpy()
    return pred_te, yte


def eval_tier(pred, true):
    return {
        "r2": float(r2_score(true, pred)),
        "mae": float(np.mean(np.abs(true - pred))),
        "pearson": float(sps.pearsonr(true, pred)[0]),
        "spearman": float(sps.spearmanr(true, pred)[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/lean_v6/best_model.pth")
    ap.add_argument("--data", default="data/dataset/memmap/validation")
    ap.add_argument("--outdir", default="analysis/probe_tiers_amplitude")
    ap.add_argument("--n_events", type=int, default=2500)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = LeanNPE(premerger=ckpt["args"].get("premerger", False), psd_cond=ckpt["args"].get("psd_cond", False) or False, psd_bands=ckpt["args"].get("psd_bands", 16), encoder_type=ckpt["args"].get("encoder_type", "conv"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"checkpoint: epoch={ckpt['epoch']} val_nll={ckpt['val_nll']:.4f}")

    ds = RemixDataset(args.data, remix=False, seed=1234)
    N = min(args.n_events, len(ds))
    print(f"events: {N}")

    # ---- pull true per-detector SIGNAL-ONLY arrays directly (bypass noise) ----
    strain_list, theta_list, snr_list = [], [], []
    peak_amp = np.zeros((N, 3), dtype=np.float32)
    det_energy = np.zeros((N, 3), dtype=np.float32)
    det_mf_snr = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        item = ds[i]
        strain_list.append(item[0])
        theta_list.append(item[1][0])
        snr_list.append(item[3])
        start, nsig = ds.events[i]
        sig_sum = np.zeros((3, ds.signals.shape[-1]), dtype=np.float32)
        for k in range(nsig):
            sig_sum += ds.signals[start + k].astype(np.float32)
        peak_amp[i] = np.abs(sig_sum).max(axis=-1)
        det_energy[i] = (sig_sum ** 2).mean(axis=-1)
        det_mf_snr[i] = np.sqrt((sig_sum ** 2).sum(axis=-1))  # exact MF SNR for whitened data

    strain = torch.stack(strain_list)
    theta = torch.stack(theta_list).numpy()
    net_snr = torch.stack(snr_list).numpy()
    m1, m2, dL = theta[:, 0], theta[:, 1], theta[:, JD]
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    total_mass = m1 + m2

    with torch.no_grad():
        ctx = torch.cat([model.encoder(strain[i:i + 64].to(device)) for i in range(0, N, 64)]).numpy()
    print(f"context extracted: {ctx.shape}")

    # ================= EXPERIMENT 1: progressively stronger probes (distance) ====
    print("\n=== Experiment 1: probe tiers, distance only ===")
    y_log_dL = np.log(dL)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    results = {"n_events": N, "experiment_1_probe_tiers": {}, "experiment_4_amplitude": {}}

    pred_linear = cross_val_predict(LinearRegression(), ctx, y_log_dL, cv=kf)
    pred_ridge = cross_val_predict(Ridge(alpha=1.0), ctx, y_log_dL, cv=kf)
    pred_small_mlp, y_small = fit_torch_probe(SmallMLP, ctx, y_log_dL)
    pred_large_mlp, y_large = fit_torch_probe(LargeMLP, ctx, y_log_dL)
    Xtr, Xte, ytr, yte = train_test_split(ctx, y_log_dL, test_size=0.2, random_state=0)
    gbm = HistGradientBoostingRegressor(max_depth=4, max_iter=200, random_state=0).fit(Xtr, ytr)
    pred_gbm = gbm.predict(Xte)

    tiers = {
        "linear": eval_tier(pred_linear, y_log_dL),
        "ridge": eval_tier(pred_ridge, y_log_dL),
        "small_mlp": eval_tier(pred_small_mlp, y_small),
        "large_mlp": eval_tier(pred_large_mlp, y_large),
        "gbm": eval_tier(pred_gbm, yte),
    }
    results["experiment_1_probe_tiers"] = tiers
    for name, m in tiers.items():
        print(f"  {name:12s} R2={m['r2']:+.3f}  MAE={m['mae']:.3f}  pearson={m['pearson']:+.3f}  spearman={m['spearman']:+.3f}")
    best_tier = max(tiers, key=lambda k: tiers[k]["r2"])
    gain_over_linear = tiers[best_tier]["r2"] - tiers["linear"]["r2"]
    results["experiment_1_summary"] = {"best_tier": best_tier, "gain_over_linear_r2": float(gain_over_linear)}
    print(f"  best tier: {best_tier}  (gain over linear: {gain_over_linear:+.3f} R2)")

    # ================= EXPERIMENT 4: amplitude representation ======================
    print("\n=== Experiment 4: amplitude representation (final context probes) ===")
    amp_targets = {}
    for d_idx, d_name in enumerate(("H1", "L1", "V1")):
        amp_targets[f"peak_amplitude_{d_name}"] = np.log(np.clip(peak_amp[:, d_idx], 1e-8, None))
        amp_targets[f"detector_energy_{d_name}"] = np.log(np.clip(det_energy[:, d_idx], 1e-10, None))
        amp_targets[f"matched_filter_snr_{d_name}"] = np.log(np.clip(det_mf_snr[:, d_idx], 1e-3, None))
    amp_targets["network_snr"] = np.log(np.clip(net_snr, 1e-3, None))
    # phase/frequency-content proxy for comparison: chirp mass (frequency evolution rate)
    amp_targets["chirp_mass_reference"] = np.log(mc)

    for name, y in amp_targets.items():
        pred = cross_val_predict(Ridge(alpha=1.0), ctx, y, cv=kf)
        m = eval_tier(pred, y)
        results["experiment_4_amplitude"][name] = m
        print(f"  {name:26s} R2={m['r2']:+.3f}  pearson={m['pearson']:+.3f}")

    with open(out / "report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\ndone -> {out / 'report.json'}")


if __name__ == "__main__":
    main()
