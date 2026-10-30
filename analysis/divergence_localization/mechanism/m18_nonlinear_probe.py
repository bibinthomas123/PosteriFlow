"""M18 - Is q stored NON-LINEARLY in the encoder context?

m17 ran a ridge (linear) probe context -> q and found held-out R^2 rising with
mass (0.007 light -> 0.458 heavy & loud). A linear probe only bounds the
LINEARLY decodable information, so it cannot distinguish

  (a) the context carries little q information at all, from
  (b) the context carries q information in a form ridge cannot read.

This runs the same frozen context through progressively more expressive
regressors -- ridge, shallow MLP, random forest, gradient boosting -- and reads
off the held-out R^2(q).

  all saturate near the ridge value  -> extra expressivity buys nothing
  MLP/GBM markedly above ridge       -> the information is there, non-linearly

Protocol. Probes are FIT on events from the NPE's train split and TESTED on the
NPE's validation split, so there is no probe leakage and the non-linear models
get several thousand heavy systems to learn from (the val split alone holds only
~2k heavy events, thin for a 256-dim input). Every probe sees the identical
inputs; each is given a hyper-parameter search on an inner split so a weak
result cannot be blamed on tuning.

Controls, so the numbers are interpretable:
  * R^2(log Mc): chirp mass is known to be well represented -- an upper anchor
    showing each regressor works at all on this context.
  * shuffled q: labels permuted -> the R^2 noise floor for that model/sample
    size (a flexible model on 256 features can fake signal).

Nothing is trained or modified in the NPE; the encoder is frozen and only
forward passes are used.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from ahsd.inference.pipeline import load_model  # noqa: E402
from experiments.remix_data import RemixDataset  # noqa: E402

from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

HERE = Path(__file__).resolve().parent
N_FIT, N_TEST, K = 12000, 4702, 50    # fit on train split, test on val split
SEED = 0


def chirp(m1, m2):
    """Chirp mass from component masses."""
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


@torch.no_grad()
def collect(net, dev, split, n_max):
    """Frozen-encoder contexts + truths + SNR + the flow's own median q.

    Cached to .npz -- the encoder is frozen, so a given (split, n_max) always
    yields identical arrays and re-running the probe analysis is then instant.
    """
    cache = HERE / f"m18_cache_{split}_{n_max}.npz"
    if cache.exists():
        z = np.load(cache)
        print(f"  collect[{split}] loaded cache {cache.name}  n={len(z['C'])}")
        return z["C"], z["T"], z["snr"], z["qmed"], z["qmean"]
    ds = RemixDataset(f"data/dataset/memmap/{split}", remix=False, seed=1234,
                      return_asd_bands=net.psd_cond)   # design/Gaussian: cleanest
    n = min(n_max, len(ds))
    C = np.empty((n, net.context_dim), np.float32)
    T = np.empty((n, 11)); snr = np.empty(n)
    qmed = np.empty(n); qmean = np.empty(n)
    B, t0 = 128, time.time()
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
        ctx = net.encode(stt, abt)                       # <-- the probe input
        full = net._full_context(ctx, torch.zeros(len(idx), dtype=torch.long, device=dev))
        rep = full.unsqueeze(1).expand(len(idx), K, full.shape[1]).reshape(-1, full.shape[1])
        y, _ = net.flow.inverse(torch.randn(len(idx) * K, 11, device=dev), rep)
        smp = net.scaler.denormalize(net.scaler.wrap(y)).reshape(len(idx), K, 11).cpu().numpy()
        a_ = np.maximum(smp[:, :, 0], smp[:, :, 1]); b_ = np.minimum(smp[:, :, 0], smp[:, :, 1])
        cc = ctx.cpu().numpy()
        for jj, i in enumerate(idx):
            C[i] = cc[jj]; T[i] = tr[jj]; snr[i] = sv[jj]
            qq = b_[jj] / a_[jj]
            qmed[i] = float(np.median(qq))
            qmean[i] = float(np.mean(qq))      # the R^2-comparable functional
        if s % (B * 20) == 0:
            print(f"  collect[{split}] {s + len(idx)}/{n}  ({time.time() - t0:.0f}s)",
                  flush=True)
    np.savez(cache, C=C, T=T, snr=snr, qmed=qmed, qmean=qmean)
    return C, T, snr, qmed, qmean


def _r2(yt, yp):
    """Coefficient of determination against the test-set mean."""
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def _inner(n, seed=SEED, frac=0.85):
    """Inner split of the fit set, for hyper-parameter selection."""
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(frac * n)
    return idx[:cut], idx[cut:]


def ridge(Xtr, ytr, Xte, lams=(1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)):
    """Ridge (with intercept) whose penalty is chosen on an inner split."""
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1), Xtr.dtype)])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1), Xte.dtype)])
    a, b = _inner(len(Xtr))
    Xa, ya, Xb, yb = Xtr[a], ytr[a], Xtr[b], ytr[b]
    P = np.eye(Xa.shape[1]); P[-1, -1] = 0.0          # never penalize the intercept
    A0, b0 = Xa.T @ Xa, Xa.T @ ya
    best, blam = -np.inf, lams[0]
    for lam in lams:
        w = np.linalg.solve(A0 + lam * P, b0)
        r = _r2(yb, Xb @ w)
        if r > best:
            best, blam = r, lam
    w = np.linalg.solve(Xtr.T @ Xtr + blam * P, Xtr.T @ ytr)
    return Xte @ w


def mlp(Xtr, ytr, Xte, hidden, seed=SEED):
    """Shallow MLP probe; weight decay picked on an inner split."""
    a, b = _inner(len(Xtr))
    best, bm = -np.inf, None
    for alpha in (1e-4, 1e-2, 1.0):
        m = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", alpha=alpha,
                         learning_rate_init=1e-3, max_iter=800, early_stopping=True,
                         n_iter_no_change=25, validation_fraction=0.15,
                         random_state=seed)
        m.fit(Xtr[a], ytr[a])
        r = _r2(ytr[b], m.predict(Xtr[b]))
        if r > best:
            best, bm = r, m
    return bm.predict(Xte)


def forest(Xtr, ytr, Xte, seed=SEED):
    """Random-forest probe."""
    m = RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                              n_jobs=-1, random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def boost(Xtr, ytr, Xte, seed=SEED):
    """Gradient-boosted-tree probe (XGBoost if present, else sklearn HistGBM)."""
    try:
        from xgboost import XGBRegressor
        mk = lambda d, e: XGBRegressor(  # noqa: E731
            n_estimators=e, max_depth=d, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.5, reg_lambda=1.0, n_jobs=-1,
            random_state=seed, verbosity=0)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        mk = lambda d, e: HistGradientBoostingRegressor(  # noqa: E731
            max_iter=e, max_depth=d, learning_rate=0.05, random_state=seed)
    a, b = _inner(len(Xtr))
    best, bm = -np.inf, None
    for depth in (4, 6, 8):
        m = mk(depth, 600); m.fit(Xtr[a], ytr[a])
        r = _r2(ytr[b], m.predict(Xtr[b]))
        if r > best:
            best, bm = r, m
    return bm.predict(Xte)


MODELS = [("ridge", lambda a, b, c: ridge(a, b, c)),
          ("mlp-256", lambda a, b, c: mlp(a, b, c, (256,))),
          ("mlp-512-256", lambda a, b, c: mlp(a, b, c, (512, 256))),
          ("rand-forest", lambda a, b, c: forest(a, b, c)),
          ("grad-boost", lambda a, b, c: boost(a, b, c))]


def standardize(Xtr, Xte):
    """Z-score using fit-set statistics only."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


def main():
    net, meta = load_model("model/lean_npe/best_model.pth"); dev = meta["device"]
    print(f"model epoch {meta['model_epoch']}  (Gaussian domain)")
    print(f"probe FIT on NPE-train split (<= {N_FIT}), TESTED on NPE-val split\n")
    Cf, Tf, sf, _, _ = collect(net, dev, "train", N_FIT)
    Ce, Te, se, qe_med, qe_mean = collect(net, dev, "validation", N_TEST)
    print(f"\nfit n={len(Cf)}   test n={len(Ce)}   context_dim {Cf.shape[1]}\n")

    qf, qt = Tf[:, 1] / Tf[:, 0], Te[:, 1] / Te[:, 0]
    mcf, mct = np.log(chirp(Tf[:, 0], Tf[:, 1])), np.log(chirp(Te[:, 0], Te[:, 1]))
    rng = np.random.default_rng(SEED)

    groups = [("light  m1<20", Tf[:, 0] < 20, Te[:, 0] < 20),
              ("mid    20<=m1<29", (Tf[:, 0] >= 20) & (Tf[:, 0] < 29),
                                   (Te[:, 0] >= 20) & (Te[:, 0] < 29)),
              ("HEAVY  m1>=29", Tf[:, 0] >= 29, Te[:, 0] >= 29),
              ("HEAVY & loud (SNR>18)", (Tf[:, 0] >= 29) & (sf > 18),
                                        (Te[:, 0] >= 29) & (se > 18))]
    targets = [("q", qf, qt), ("log Mc [ctrl]", mcf, mct),
               ("q shuffled", rng.permutation(qf), rng.permutation(qt))]

    hdr = "".join(f"{n:>13}" for n, _ in MODELS)
    results = {}
    for lab, mf, me in groups:
        if mf.sum() < 200 or me.sum() < 100:
            continue
        Xf, Xe = standardize(Cf[mf], Ce[me])
        print(f"\n{lab}   fit n={int(mf.sum())}  test n={int(me.sum())}   "
              f"flow's own q err {np.median(np.abs(qe_med[me] - qt[me])):.3f}  "
              f"flow q {np.median(qe_med[me]):.3f}  true q {np.median(qt[me]):.3f}")
        print(f"  {'target':14}{hdr}")
        best_q = -np.inf
        for tname, yf, ye in targets:
            row = {}
            for name, fn in MODELS:
                row[name] = _r2(ye[me], fn(Xf, yf[mf], Xe))
            results.setdefault(lab, {})[tname] = row
            if tname == "q":
                best_q = max(row.values())
            print(f"  {tname:14}" + "".join(f"{row[n]:>13.3f}" for n, _ in MODELS),
                  flush=True)
        # The flow reads the SAME context. Score its own q estimate on the SAME
        # events, so probe and flow are directly comparable: a large gap means
        # decodable q information that the flow does not use.
        #
        # The MEAN is the fair comparison -- the probes minimize squared error,
        # so they estimate E[q|context], which is the R^2-maximizing predictor
        # by construction. A calibrated posterior's mean is that same functional,
        # so it should match the probe if the flow uses the context fully. The
        # median is reported too, but it is a different functional and would
        # lose R^2 on a skewed posterior for reasons unrelated to information.
        r2_mean, r2_med = _r2(qt[me], qe_mean[me]), _r2(qt[me], qe_med[me])
        results[lab].update(flow_mean_q_r2=r2_mean, flow_median_q_r2=r2_med,
                            best_probe_q_r2=best_q)
        print(f"  {'FLOW mean q':14}{r2_mean:>13.3f}   <- NSF's own E[q|ctx] "
              f"(best probe {best_q:.3f}, gap {best_q - r2_mean:+.3f})")
        print(f"  {'FLOW med q':14}{r2_med:>13.3f}   (different functional; "
              f"reported for reference only)")

    print("\nreading guide:")
    print("  compare each 'q' row against 'q shuffled' (that model's R^2 noise floor)")
    print("  and against 'log Mc [ctrl]' (what a well-represented target looks like).")
    print("  ridge ~= mlp ~= boost  -> extra expressivity buys nothing")
    print("  mlp/boost >> ridge     -> q is present but non-linearly encoded")
    json.dump(results, open(HERE / "m18_nonlinear_probe.json", "w"),
              indent=2, default=float)
    print(f"\nwrote {HERE / 'm18_nonlinear_probe.json'}")


if __name__ == "__main__":
    main()
