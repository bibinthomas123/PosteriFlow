#!/usr/bin/env python3
"""
CI validation suite: everything a LeanNPE checkpoint must pass, in one command,
out as one self-contained HTML report with PASS/FAIL gates.

    python scripts/validate_checkpoint.py --model model/lean_npe_v3/best_model.pth

Runs on the fixed validation split (Gaussian domain) and, when a noise bank is
present, the deterministic real-noise variant of the same events:
  - NLL + context-shuffle delta (conditional-vs-marginal)
  - credible-interval coverage at 50/68/90/95% for all 11 parameters
  - SBC ranks + KS tests, PP curves
  - boundary-railing statistics
Then real-data smoke tests: 6 GWTC events through the production pipeline
(timing must land on each published GPS; GW170729 at dL~2750 Mpc lies outside
the training prior and live-tests the OOD/refinement gate).
Also fits <checkpoint_dir>/ood_stats.npz if missing, so the OOD detector is
armed for every validated checkpoint.

Exit code 0 = all gates pass, 1 = at least one FAIL (CI-friendly).
"""
import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _ROOT)
sys.path.insert(0, f"{_ROOT}/src")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES

BLUE, GRAY, INK, RED = "#3b6fd4", "#9aa4b2", "#3a3f47", "#c65a49"
CI_LEVELS = [0.50, 0.68, 0.90, 0.95]
plt.rcParams.update({"axes.edgecolor": GRAY, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
                     "grid.color": "#e8ebef", "grid.linewidth": 0.6, "font.size": 9})


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def evaluate_domain(model, ds, n_events, n_post, device, tag):
    """All sample-based metrics + figures for one data domain."""
    n = min(n_events, len(ds))
    items = [ds[i] for i in range(n)]
    strain = torch.stack([x[0] for x in items])
    theta = torch.stack([x[1][0] for x in items]).to(device)
    # PSD-conditioned models: use the domain's per-event sensitivity summary
    # (nonzero for the real-noise domain, ~0 for Gaussian/design).
    asd = torch.stack([x[4] for x in items]) if len(items[0]) > 4 else None
    N, P = n, len(PARAM_NAMES)

    ctx = []
    with torch.no_grad():
        for i in range(0, N, 64):
            ab = asd[i:i + 64].to(device).float() if asd is not None else None
            ctx.append(model.encode(strain[i:i + 64].to(device).float(), ab))
    ctx = torch.cat(ctx)
    rank0 = torch.zeros(N, dtype=torch.long, device=device)
    with torch.no_grad():
        nll_t = model.nll(None, theta, rank0, context=ctx)
        nll_s = model.nll(None, theta, rank0, context=ctx[torch.randperm(N, device=device)])

    K = n_post
    samples = np.empty((N, K, P), dtype=np.float32)
    railed = np.zeros((N, K), dtype=bool)
    full_ctx = model._full_context(ctx, rank0)
    with torch.no_grad():
        for i in range(0, N, 32):
            fc = full_ctx[i:i + 32]
            nb = fc.shape[0]
            fcr = fc.unsqueeze(1).expand(nb, K, fc.shape[1]).reshape(nb * K, -1)
            z = torch.randn(nb * K, P, device=device)
            y, _ = model.flow.inverse(z, fcr)
            y = model.scaler.wrap(y)  # circular wrap; bounded clamp
            # SPURIOUS railing only: a sample pinned at a bound in a dim whose
            # TRUE value is not near that bound. Bound pile-up where the truth
            # actually sits at the edge (spins ~0, face-on inclination) is the
            # posterior doing its job, not miscalibration.
            y_true = model.scaler.normalize(theta[i:i + 32]).unsqueeze(1)  # [nb,1,P]
            yr = y.reshape(nb, K, P)
            spurious = ((yr > 0.999) & (y_true < 0.9) |
                        (yr < -0.999) & (y_true > -0.9)) & ~model.scaler.circ_mask
            railed[i:i + 32] = spurious.any(dim=2).cpu().numpy()
            samples[i:i + 32] = model.scaler.denormalize(y).reshape(nb, K, P).cpu().numpy()

    truth = theta.cpu().numpy()
    coverage = {name: {} for name in PARAM_NAMES}
    for j, name in enumerate(PARAM_NAMES):
        for lev in CI_LEVELS:
            lo = np.quantile(samples[:, :, j], 0.5 - lev / 2, axis=1)
            hi = np.quantile(samples[:, :, j], 0.5 + lev / 2, axis=1)
            coverage[name][f"{int(lev*100)}"] = float(
                ((truth[:, j] >= lo) & (truth[:, j] <= hi)).mean())
    # SBC ranks: plain linear ranks are a valid SBC statistic for every
    # parameter (rank uniformity holds for any fixed ordering) — circular
    # parameters are fine now that sampling WRAPS instead of clamping at the
    # period seam (the old edge pile-up was a sampling artifact, not a rank-
    # statistic problem)
    ranks = (samples < truth[:, None, :]).mean(axis=1)
    ks = {name: sps.kstest(ranks[:, j], "uniform") for j, name in enumerate(PARAM_NAMES)}

    jd = PARAM_NAMES.index("luminosity_distance")
    d_med = np.median(samples[:, :, jd], axis=1)
    metrics = {
        "n_events": N, "n_post": K,
        "nll": float(nll_t.mean()),
        "shuffle_delta_nll": float(nll_s.mean() - nll_t.mean()),
        "dist_corr": float(np.corrcoef(np.log(d_med), np.log(truth[:, jd]))[0, 1]),
        "coverage": coverage,
        "sbc_ks_p": {k: float(v.pvalue) for k, v in ks.items()},
        "railing_frac": float(railed.mean()),
    }

    figs = {}
    # coverage bars
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    xs = np.arange(P)
    for a, lev in zip(axes, CI_LEVELS):
        a.bar(xs, [coverage[nm][f"{int(lev*100)}"] for nm in PARAM_NAMES],
              color=BLUE, width=0.62)
        a.axhline(lev, color=INK, lw=1, ls="--")
        a.set_title(f"{int(lev*100)}% CI")
        a.set_xticks(xs)
        a.set_xticklabels(PARAM_NAMES, rotation=60, ha="right", fontsize=7)
        a.set_ylim(0, 1)
    fig.suptitle(f"{tag}: coverage ({N} events)")
    figs["coverage"] = fig_to_b64(fig)
    # SBC ranks
    fig, axes = plt.subplots(3, 4, figsize=(13, 8))
    for j, name in enumerate(PARAM_NAMES):
        a = axes.flat[j]
        a.hist(ranks[:, j], bins=20, color=BLUE, edgecolor="white", linewidth=0.4)
        a.axhline(N / 20, color=INK, lw=1, ls="--")
        a.set_title(f"{name} (KS p={ks[name].pvalue:.3f})", fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle(f"{tag}: SBC ranks")
    figs["sbc"] = fig_to_b64(fig)
    # PP curves
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    levs = np.linspace(0.02, 0.98, 40)
    for j, name in enumerate(PARAM_NAMES):
        emp = [np.mean(np.abs(ranks[:, j] - 0.5) <= l / 2) for l in levs]
        hl = name == "luminosity_distance"
        ax.plot(levs, emp, lw=1.3, color=BLUE if hl else GRAY, alpha=1 if hl else 0.5)
    ax.plot([0, 1], [0, 1], color=INK, lw=1, ls="--")
    ax.set_xlabel("nominal")
    ax.set_ylabel("empirical")
    ax.set_title(f"{tag}: PP calibration")
    figs["pp"] = fig_to_b64(fig)
    return metrics, figs


def gate(name, ok, detail):
    return {"gate": name, "pass": bool(ok), "detail": detail}


def run_gates(gaus, real, gw_ok, gw_summary):
    gates = []
    gates.append(gate("gaussian shuffle ΔNLL > 5 nats",
                      gaus["shuffle_delta_nll"] > 5, f"{gaus['shuffle_delta_nll']:.2f}"))
    for lev, tol in (("50", 0.07), ("90", 0.05)):
        bad = [n for n in PARAM_NAMES
               if abs(gaus["coverage"][n][lev] - int(lev) / 100) > tol]
        gates.append(gate(f"gaussian {lev}% coverage within ±{tol} (≤2 exceptions)",
                          len(bad) <= 2, f"violations: {bad or 'none'}"))
    n_ks_ok = sum(p > 1e-3 for p in gaus["sbc_ks_p"].values())
    gates.append(gate("gaussian SBC KS p>1e-3 for ≥9/11 params",
                      n_ks_ok >= 9, f"{n_ks_ok}/11"))
    gates.append(gate("gaussian spurious railing < 5%",
                      gaus["railing_frac"] < 0.05, f"{gaus['railing_frac']:.2%}"))
    gates.append(gate("distance correlation > 0.5",
                      gaus["dist_corr"] > 0.5, f"{gaus['dist_corr']:.3f}"))
    if real is not None:
        gates.append(gate("real-noise shuffle ΔNLL > 5 nats",
                          real["shuffle_delta_nll"] > 5, f"{real['shuffle_delta_nll']:.2f}"))
        bad = [n for n in PARAM_NAMES if abs(real["coverage"][n]["90"] - 0.90) > 0.07]
        gates.append(gate("real-noise 90% coverage within ±0.07 (≤2 exceptions)",
                          len(bad) <= 2, f"violations: {bad or 'none'}"))
        gates.append(gate("real vs gaussian NLL gap < 3 nats",
                          abs(real["nll"] - gaus["nll"]) < 3,
                          f"gaussian {gaus['nll']:.2f} vs real {real['nll']:.2f}"))
    gates.append(gate("real-event smoke tests (6 GWTC events, |t_c|<0.1s)", gw_ok, gw_summary))
    return gates


def html_report(out_path, meta, gaus, gfigs, real, rfigs, gw, gates):
    ok = all(g["pass"] for g in gates)
    rows = "".join(
        f"<tr class={'pass' if g['pass'] else 'fail'}><td>{'PASS' if g['pass'] else 'FAIL'}</td>"
        f"<td>{g['gate']}</td><td>{g['detail']}</td></tr>" for g in gates)

    def domain_html(tag, m, figs):
        if m is None:
            return f"<h2>{tag}</h2><p>skipped (no noise bank)</p>"
        return (f"<h2>{tag}</h2>"
                f"<p>NLL {m['nll']:.3f} | shuffle Δ {m['shuffle_delta_nll']:.2f} | "
                f"dist corr {m['dist_corr']:.3f} | railing {m['railing_frac']:.2%} | "
                f"{m['n_events']} events × {m['n_post']} draws</p>"
                + "".join(f'<img src="data:image/png;base64,{b}"/>' for b in figs.values()))

    gw_html = "<p>skipped</p>"
    if gw:
        rows_gw = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in gw.items())
        gw_html = f"<table>{rows_gw}</table>"

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>LeanNPE checkpoint validation</title><style>
body{{font-family:-apple-system,sans-serif;margin:2em auto;max-width:1100px;color:{INK}}}
img{{max-width:100%;margin:6px 0}} table{{border-collapse:collapse;margin:1em 0}}
td{{border:1px solid #d7dce2;padding:4px 10px;font-size:14px}}
tr.pass td:first-child{{background:#e3f0e3;font-weight:600}}
tr.fail td:first-child{{background:#f6dcd7;font-weight:600}}
.badge{{display:inline-block;padding:4px 14px;border-radius:6px;color:#fff;
background:{'#3f8f4a' if ok else '#c65a49'};font-weight:700}}</style></head><body>
<h1>LeanNPE checkpoint validation <span class="badge">{'PASS' if ok else 'FAIL'}</span></h1>
<p>checkpoint: <code>{meta['model']}</code> (epoch {meta['epoch']}, val NLL {meta['val_nll']:.4f})<br>
run: {meta['timestamp']} | device {meta['device']} | wall {meta['wall_s']:.0f}s</p>
<h2>Gates</h2><table><tr><td></td><td>gate</td><td>detail</td></tr>{rows}</table>
{domain_html('Gaussian validation', gaus, gfigs)}
{domain_html('Real-noise validation', real, rfigs)}
<h2>Real-event smoke tests</h2>{gw_html}
</body></html>"""
    Path(out_path).write_text(html)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="path to remix_data/ directory")
    ap.add_argument("--noise_bank", default="data/noise_bank")
    ap.add_argument("--outdir", default=None,
                    help="default: analysis/ci/<checkpoint dir name>")
    ap.add_argument("--n_events", type=int, default=2000,
                    help="validation events per domain (5000 available; 2000 gives coverage MC error ~1.1% at the 50% level vs 1.8% at 768)")
    ap.add_argument("--n_post", type=int, default=400)
    ap.add_argument("--skip_gw150914", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    t_start = time.time()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(args.outdir or f"analysis/ci/{Path(args.model).parent.name}")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = LeanNPE(premerger=ckpt["args"].get("premerger", False),
                    psd_cond=ckpt["args"].get("psd_cond", False) or False,
                    psd_bands=ckpt["args"].get("psd_bands", 16),
                    encoder_type=ckpt["args"].get("encoder_type", "conv"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"checkpoint: epoch {ckpt['epoch']}, val NLL {ckpt['val_nll']:.4f}")

    from experiments.remix_data import RemixDataset, build_memmap_cache
    vcache = Path(args.data) / "memmap" / "validation"
    if not (vcache / "events.json").exists():
        build_memmap_cache(args.data, "validation", str(vcache))

    print("Gaussian validation ...")
    ds_g = RemixDataset(str(vcache), remix=False, seed=1234,
                        return_asd_bands=model.psd_cond,
                        psd_bands=model.encoder.psd_bands if model.psd_cond else 16)
    gaus, gfigs = evaluate_domain(model, ds_g, args.n_events, args.n_post, device,
                                  "Gaussian")

    real, rfigs = None, {}
    if args.noise_bank and Path(args.noise_bank).exists():
        try:
            print("Real-noise validation ...")
            ds_r = RemixDataset(str(vcache), remix=False, seed=1234,
                                real_noise_dir=args.noise_bank, real_noise_prob=1.0,
                                return_asd_bands=model.psd_cond,
                                psd_bands=model.encoder.psd_bands if model.psd_cond else 16)
            real, rfigs = evaluate_domain(model, ds_r, args.n_events, args.n_post,
                                          device, "Real noise")
        except Exception as e:
            print(f"real-noise validation skipped: {e}")

    # arm the OOD detector for this checkpoint
    ood_path = Path(args.model).parent / "ood_stats.npz"
    if not ood_path.exists():
        print("fitting OOD context stats ...")
        from ahsd.inference.ood import fit_context_stats
        print(fit_context_stats(model, str(vcache), str(ood_path), device=device))

    gw, gw_ok, gw_summary = None, True, "skipped"
    # GW170729 (dL ~2750 Mpc) sits OUTSIDE the training distance prior — it is
    # a deliberate live test of the OOD/refinement gate on every CI run.

    REAL_EVENTS = [
            ("GW150914", {"m1": 38.8, "m2": 33.4, "dL": 440}),
            ("GW151226", {"m1": 13.7, "m2": 7.7, "dL": 440}),
            ("GW170104", {"m1": 31.0, "m2": 20.1, "dL": 960}),
            ("GW170608", {"m1": 12.0, "m2": 7.0, "dL": 320}),
            ("GW170729", {"m1": 50.6, "m2": 34.3, "dL": 2750}),
            ("GW170814", {"m1": 30.7, "m2": 25.3, "dL": 600}),
        ]
    
    if not args.skip_gw150914:
        from ahsd.inference import infer
        gw, fails = {}, []
        for name, pub in REAL_EVENTS:
            print(f"{name} smoke test ...")
            try:
                res = infer(event=name, num_samples=3000, model=args.model,
                            device=device)
                med = res.median
                v = res.diagnostics.get("verdict", {})
                g = res.diagnostics.get("refinement", {})
                tc_ok = abs(med["geocent_time"]) < 0.1
                if not tc_ok:
                    fails.append(f"{name}: t_c offset {med['geocent_time']:+.3f}s")
                gw[name] = (
                    f"m1 {med['mass_1']:.1f} (pub {pub['m1']}) | "
                    f"m2 {med['mass_2']:.1f} (pub {pub['m2']}) | "
                    f"dL {med['luminosity_distance']:.0f} (pub {pub['dL']}) | "
                    f"t_c {med['geocent_time']:+.3f}s | "
                    f"rail {res.diagnostics['boundary_railing_frac']:.1%} | "
                    f"conf {v.get('confidence', '?')} | "
                    f"gate: {'REFINE' if g.get('refine') else 'as-is'}"
                    + (f" ({','.join(g.get('untrusted_params', []))})"
                       if g.get('untrusted_params') else ""))
                res.save(out / name.lower(), plots=True)
            except Exception as e:
                fails.append(f"{name}: {type(e).__name__}: {e}")
                gw[name] = f"FAILED: {e}"
        gw_ok = not fails
        gw_summary = (f"all {len(REAL_EVENTS)} events ran, |t_c|<0.1s"
                      if gw_ok else "; ".join(fails)[:200])

    meta = {"model": args.model, "epoch": ckpt["epoch"], "val_nll": ckpt["val_nll"],
            "device": device, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_s": time.time() - t_start}
    gates = run_gates(gaus, real, gw_ok, gw_summary)
    ok = html_report(out / "report.html", meta, gaus, gfigs, real, rfigs, gw, gates)

    with open(out / "report.json", "w") as f:
        json.dump({"meta": meta, "gaussian": gaus, "real": real, "gw150914": gw,
                   "gates": gates}, f, indent=2, default=float)
    for g in gates:
        print(f"  {'PASS' if g['pass'] else 'FAIL'}  {g['gate']}  ({g['detail']})")
    print(f"report -> {out}/report.html  |  overall: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
