"""STEP 5 - High-SNR conditioning audit on the validation population.

Runs the current long_run (ep19) checkpoint over the validation split, in both
the Gaussian and real-noise domains, and bins events by network SNR. Per bin:
NLL, distance correlation, chirp-mass correlation, 50/90% coverage
(calibration), SBC uniformity (KS p over the rank-0 truth quantiles), and
posterior railing. Answers: does in-distribution performance deteriorate at
high SNR the way the real GWTC events do?
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "src")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))          # for `experiments`
sys.path.insert(0, str(ROOT / "src"))  # for `ahsd`
sys.path.insert(0, str(HERE))
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402
from ahsd.inference.pipeline import load_model  # noqa: E402
from experiments.remix_data import RemixDataset  # noqa: E402

MODEL = "model/long_run/best_model.pth"
VCACHE = "data/dataset/memmap/validation"
NBANK = "data/noise_bank"
N_EVENTS = 4000
K = 200                     # posterior samples per event
BINS = [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 1e9)]
CIRC = None                 # set after model load
DIDX = PARAM_NAMES.index("luminosity_distance")
M1I, M2I = PARAM_NAMES.index("mass_1"), PARAM_NAMES.index("mass_2")


def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def ks_uniform(x):
    """KS statistic + p-value of x vs Uniform(0,1). Small p => non-uniform."""
    from scipy import stats
    if len(x) < 8:
        return float("nan"), float("nan")
    return tuple(float(v) for v in stats.kstest(np.clip(x, 1e-6, 1 - 1e-6), "uniform"))


# science params (exclude circular + spins for the "science" railing/coverage)
SCI = [PARAM_NAMES.index(p) for p in ("mass_1", "mass_2", "luminosity_distance")]
SPIN = [PARAM_NAMES.index(p) for p in ("a1", "a2")]


def run_domain(net, dev, domain, stats_circ):
    kw = dict(remix=False, seed=1234, return_asd_bands=net.psd_cond)
    if domain == "real":
        ds = RemixDataset(VCACHE, real_noise_dir=NBANK, real_noise_prob=1.0, **kw)
    else:
        ds = RemixDataset(VCACHE, **kw)
    n = min(N_EVENTS, len(ds))
    noncirc = (~stats_circ).cpu().numpy()

    snr = np.empty(n); nll = np.empty(n)
    rail_naive = np.empty(n); rail_spur = np.empty(n)     # any non-circ dim
    rail_pp_naive = np.zeros((n, len(PARAM_NAMES)))       # per-param naive
    rail_pp_spur = np.zeros((n, len(PARAM_NAMES)))        # per-param spurious
    dL_true = np.empty(n); dL_pred = np.empty(n)
    Mc_true = np.empty(n); Mc_pred = np.empty(n)
    quant = np.empty((n, len(PARAM_NAMES)))

    B = 128
    with torch.no_grad():
        for start in range(0, n, B):
            idxs = list(range(start, min(start + B, n)))
            strains, truths, snrs, abs = [], [], [], []
            for i in idxs:
                item = ds[i]
                if net.psd_cond:
                    st, pv, nsig, ns, ab = item; abs.append(ab.numpy())
                else:
                    st, pv, nsig, ns = item
                strains.append(st.numpy()); truths.append(pv.numpy()[0]); snrs.append(float(ns))
            st = torch.from_numpy(np.stack(strains)).to(dev).float()
            ab = (torch.from_numpy(np.stack(abs)).to(dev).float() if net.psd_cond else None)
            tr = torch.from_numpy(np.stack(truths)).to(dev).float()
            y_true = net.scaler.normalize(tr).cpu().numpy()   # [b,11] normalized truth
            r = torch.zeros(len(idxs), dtype=torch.long, device=dev)

            ctx = net.encode(st, ab)
            fullc = net._full_context(ctx, r)
            nll_b = net.flow.compute_psd_aware_nll(
                net.scaler.normalize(tr), fullc, torch.zeros_like(tr)).cpu().numpy()
            crep = fullc.unsqueeze(1).expand(len(idxs), K, fullc.shape[1]).reshape(-1, fullc.shape[1])
            z = torch.randn(len(idxs) * K, len(PARAM_NAMES), device=dev)
            y, _ = net.flow.inverse(z, crep)
            yw_t = net.scaler.wrap(y)
            samp = net.scaler.denormalize(yw_t).reshape(len(idxs), K, len(PARAM_NAMES)).cpu().numpy()
            yw = yw_t.reshape(len(idxs), K, len(PARAM_NAMES)).cpu().numpy()
            a = np.maximum(samp[:, :, M1I], samp[:, :, M2I])
            b = np.minimum(samp[:, :, M1I], samp[:, :, M2I])
            samp[:, :, M1I], samp[:, :, M2I] = a, b
            truths = np.stack(truths)

            hi = yw > 0.999; lo = yw < -0.999                 # [b,K,11]
            yt = y_true[:, None, :]                            # [b,1,11]
            # spurious: railed at a bound where truth is NOT near that bound
            spur = (hi & (yt < 0.9)) | (lo & (yt > -0.9))
            naive = hi | lo
            for jj, i in enumerate(idxs):
                snr[i] = snrs[jj]; nll[i] = nll_b[jj]
                nb = naive[jj][:, noncirc].any(1).mean()
                sb = spur[jj][:, noncirc].any(1).mean()
                rail_naive[i] = float(nb); rail_spur[i] = float(sb)
                rail_pp_naive[i] = naive[jj].mean(0)
                rail_pp_spur[i] = spur[jj].mean(0)
                dL_true[i] = truths[jj, DIDX]; dL_pred[i] = np.median(samp[jj, :, DIDX])
                Mc_true[i] = chirp_mass(truths[jj, M1I], truths[jj, M2I])
                Mc_pred[i] = np.median(chirp_mass(samp[jj, :, M1I], samp[jj, :, M2I]))
                quant[i] = (samp[jj] < truths[jj][None, :]).mean(0)
    return dict(snr=snr, nll=nll, rail_naive=rail_naive, rail_spur=rail_spur,
                rail_pp_naive=rail_pp_naive, rail_pp_spur=rail_pp_spur,
                dL_true=dL_true, dL_pred=dL_pred, Mc_true=Mc_true, Mc_pred=Mc_pred,
                quant=quant, n=n)


def summarize(res, domain):
    snr = res["snr"]
    rows = []
    for lo, hi in BINS:
        m = (snr >= lo) & (snr < hi)
        cnt = int(m.sum())
        label = f"{lo:g}-{hi:g}" if hi < 1e9 else f">{lo:g}"
        if cnt < 10:
            rows.append(dict(bin=label, n=cnt))
            continue
        dcorr = np.corrcoef(res["dL_true"][m], res["dL_pred"][m])[0, 1]
        mcorr = np.corrcoef(res["Mc_true"][m], res["Mc_pred"][m])[0, 1]
        q = res["quant"][m]
        qsci = q[:, SCI]
        cov50_sci = float(((qsci > 0.25) & (qsci < 0.75)).mean())
        cov90_sci = float(((qsci > 0.05) & (qsci < 0.95)).mean())
        _, ksp = ks_uniform(q[:, DIDX])
        rows.append(dict(
            bin=label, n=cnt, nll=float(np.mean(res["nll"][m])),
            dcorr=float(dcorr), mcorr=float(mcorr),
            cov50_sci=cov50_sci, cov90_sci=cov90_sci,
            rail_naive=float(np.mean(res["rail_naive"][m])),
            rail_spur=float(np.mean(res["rail_spur"][m])),
            sbc_p=float(ksp),
            rail_dL_spur=float(res["rail_pp_spur"][m][:, DIDX].mean()),
            rail_m1_spur=float(res["rail_pp_spur"][m][:, M1I].mean()),
            rail_a2_naive=float(res["rail_pp_naive"][m][:, SPIN[1]].mean()),
            rail_a2_spur=float(res["rail_pp_spur"][m][:, SPIN[1]].mean()),
        ))
    return rows


def main():
    net, meta = load_model(MODEL)
    dev = meta["device"]
    stats_circ = net.scaler.circ_mask.to(dev)
    print(f"model epoch {meta['model_epoch']} val_nll {meta['model_val_nll']:.4f} "
          f"dev {dev}\n")

    out = {}
    for domain in ["gaussian", "real"]:
        print(f"===== {domain.upper()} DOMAIN =====")
        try:
            res = run_domain(net, dev, domain, stats_circ)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"{domain} skipped: {e}"); continue
        rows = summarize(res, domain)
        out[domain] = {"rows": rows,
                       "overall": {"nll": float(np.mean(res["nll"])),
                                   "rail_naive": float(np.mean(res["rail_naive"])),
                                   "rail_spur": float(np.mean(res["rail_spur"])),
                                   "n": res["n"]}}
        print(f"{'SNRbin':>7} {'n':>5} {'NLL':>6} {'dcorr':>5} {'mcorr':>5} "
              f"{'cov50s':>6} {'cov90s':>6} {'rNAIV':>6} {'rSPUR':>6} "
              f"{'r_dL_s':>6} {'r_m1_s':>6} {'a2naiv':>6} {'a2spur':>6} {'SBC_p':>8}")
        for r in rows:
            if r["n"] < 10:
                print(f"{r['bin']:>7} {r['n']:>5}  (too few)"); continue
            print(f"{r['bin']:>7} {r['n']:>5} {r['nll']:>6.1f} {r['dcorr']:>5.2f} "
                  f"{r['mcorr']:>5.2f} {r['cov50_sci']:>6.2f} {r['cov90_sci']:>6.2f} "
                  f"{r['rail_naive']:>6.2f} {r['rail_spur']:>6.2f} "
                  f"{r['rail_dL_spur']:>6.2f} {r['rail_m1_spur']:>6.2f} "
                  f"{r['rail_a2_naive']:>6.2f} {r['rail_a2_spur']:>6.2f} {r['sbc_p']:>8.1e}")
        print(f"overall  rail_naive {out[domain]['overall']['rail_naive']:.3f}  "
              f"rail_spur {out[domain]['overall']['rail_spur']:.3f}  "
              f"nll {out[domain]['overall']['nll']:.3f}\n")

    json.dump(out, open(HERE / "step5_snr_bins.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
