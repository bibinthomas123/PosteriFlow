"""M20 - WHERE is the information about q lost?

Established: the frozen context supports R^2(q) ~ 0.47-0.50 on heavy systems for
every probe family (m18: ridge == MLP == forest == boosting, so expressivity is
not the limiter), while the flow's own E[q|ctx] from the same context reaches
only 0.19-0.32 even after optimal affine recalibration (m19). The attractor
reproduces on pure simulation (m14/m15); PSD/asd_bands, real noise,
preprocessing, detector config and late-training overconfidence are all ruled
out. Encoder-vs-flow is NOT assumed here.

The decisive comparison is DERIVED vs DIRECT q, for the probe and the flow:

  q is a RATIO. Anything that predicts m1 and m2 can form m2/m1 without ever
  representing q. So for each predictor, measure q two ways --

    derived : build q from that predictor's own m1, m2
    direct  : ask that predictor for q itself

  PROBE  derived ~= direct  -> the context's q content is just the m1/m2 content
  PROBE  direct  >> derived -> the context carries q beyond its m1/m2 marginals

  FLOW   E[m2]/E[m1] (marginal means, ignores the joint)  vs
         E[m2/m1]    (per-sample ratio, uses the joint)
    the first >> the second means the WITHIN-POSTERIOR m1-m2 correlation is
    destroying q even though the marginals are fine. That is the signature of a
    joint-modelling failure, and it is measured, not inferred from R^2 alone.

Scenarios this discriminates:
  A joint failure      : flow m1/m2 ~= probe m1/m2, but flow q << probe q,
                         and flow E[m2]/E[m1] >> flow E[m2/m1]
  B conditioning       : probe recovers m1/m2/q, flow recovers none of them
  C encoder limitation : probe q ~= flow q ~= 0.5 and probe derived ~= direct
                         (nothing beyond the marginals is there to lose)

Reuses m18's cached contexts; the flow is re-sampled straight from them (no
encoder pass, no data loading). Nothing is trained or modified in the NPE.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from ahsd.inference.pipeline import load_model  # noqa: E402

from sklearn.ensemble import RandomForestRegressor  # noqa: E402

HERE = Path(__file__).resolve().parent
FIT = HERE / "m18_cache_train_12000.npz"
TEST = HERE / "m18_cache_validation_4702.npz"
K = 400          # posterior draws/event -> MC error on the mean is negligible


def chirp(m1, m2):
    """Chirp mass from component masses."""
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def _r2(yt, yp):
    """Coefficient of determination against the mean of yt."""
    return 1.0 - float(((yt - yp) ** 2).sum()) / float(((yt - yt.mean()) ** 2).sum())


@torch.no_grad()
def flow_stats(net, dev, C):
    """Flow posterior summaries, sampled from cached contexts.

    Returns per-event posterior means of m1, m2, log Mc; both q estimators
    (per-sample E[m2/m1], which uses the joint, and E[m2]/E[m1], which uses only
    the marginals); and the within-event Pearson corr(m1, m2) across draws.
    """
    keys = ("m1", "m2", "logMc", "q_direct", "q_from_marginals", "corr")
    out = {k: np.empty(len(C)) for k in keys}
    B = 256
    for s in range(0, len(C), B):
        ctx = torch.from_numpy(C[s:s + B]).to(dev).float()
        n = len(ctx)
        full = net._full_context(ctx, torch.zeros(n, dtype=torch.long, device=dev))
        rep = full.unsqueeze(1).expand(n, K, full.shape[1]).reshape(-1, full.shape[1])
        y, _ = net.flow.inverse(torch.randn(n * K, 11, device=dev), rep)
        smp = net.scaler.denormalize(net.scaler.wrap(y)).reshape(n, K, 11).cpu().numpy()
        a_ = np.maximum(smp[:, :, 0], smp[:, :, 1])
        b_ = np.minimum(smp[:, :, 0], smp[:, :, 1])
        out["m1"][s:s + n] = a_.mean(1)
        out["m2"][s:s + n] = b_.mean(1)
        out["logMc"][s:s + n] = np.log(chirp(a_, b_)).mean(1)
        out["q_direct"][s:s + n] = (b_ / a_).mean(1)
        out["q_from_marginals"][s:s + n] = b_.mean(1) / a_.mean(1)
        az, bz = a_ - a_.mean(1, keepdims=True), b_ - b_.mean(1, keepdims=True)
        out["corr"][s:s + n] = (az * bz).mean(1) / (a_.std(1) * b_.std(1) + 1e-9)
    return out


def _inner(n, seed=0, frac=0.85):
    """Inner split of the fit set, for hyper-parameter selection."""
    idx = np.random.default_rng(seed).permutation(n)
    return idx[:int(frac * n)], idx[int(frac * n):]


def ridge(Xtr, ytr, Xte, lams=(1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)):
    """Ridge (with unpenalized intercept); penalty chosen on an inner split."""
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1), Xtr.dtype)])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1), Xte.dtype)])
    a, b = _inner(len(Xtr))
    P = np.eye(Xtr.shape[1]); P[-1, -1] = 0.0
    A0, b0 = Xtr[a].T @ Xtr[a], Xtr[a].T @ ytr[a]
    best, blam = -np.inf, lams[0]
    for lam in lams:
        w = np.linalg.solve(A0 + lam * P, b0)
        r = _r2(ytr[b], Xtr[b] @ w)
        if r > best:
            best, blam = r, lam
    w = np.linalg.solve(Xtr.T @ Xtr + blam * P, Xtr.T @ ytr)
    return Xte @ w


def forest(Xtr, ytr, Xte, seed=0):
    """Random-forest probe (m18: all families tied, so one non-linear check)."""
    m = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                              n_jobs=-1, random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def best_probe(Xf, yf, Xe, ye):
    """Best held-out prediction over {ridge, forest}: probe = the best reader."""
    pr, pf = ridge(Xf, yf, Xe), forest(Xf, yf, Xe)
    return (pr, "ridge") if _r2(ye, pr) >= _r2(ye, pf) else (pf, "forest")


def main():
    net, meta = load_model("model/lean_npe/best_model.pth"); dev = meta["device"]
    zf, ze = np.load(FIT), np.load(TEST)
    Cf, Tf, snrf = zf["C"], zf["T"], zf["snr"]
    Ce, Te, snre = ze["C"], ze["T"], ze["snr"]
    print(f"model epoch {meta['model_epoch']}   probe fit n={len(Cf)} (train split) "
          f"-> test n={len(Ce)} (val split)   flow K={K} draws/event\n")
    print("sampling flow from cached contexts ...", flush=True)
    F = flow_stats(net, dev, Ce)

    def tgt(T):
        m1, m2 = T[:, 0], T[:, 1]
        return {"m1": m1, "m2": m2, "logMc": np.log(chirp(m1, m2)),
                "q": m2 / m1, "logm1": np.log(m1), "logm2": np.log(m2)}
    Yf, Ye = tgt(Tf), tgt(Te)

    # ALL is the ONLY unbiased probe-vs-flow comparison. The m1-based regimes
    # select on a TRUTH variable that correlates with both q and the context, so
    # a probe fit inside a regime estimates E[q | ctx, m1>=29] while the flow can
    # only ever estimate E[q | ctx]. Those differ, which hands the probe an
    # advantage and inflates every probe-minus-flow gap by an unknown amount.
    # (It also deflates R^2(m1) inside m1-selected regimes: selection truncates
    # Var(m1), shrinking the R^2 denominator.) Regimes are kept for continuity
    # with m17/m18/m19 but must not carry the verdict.
    allf, alle = np.ones(len(Tf), bool), np.ones(len(Te), bool)
    groups = [("ALL (no truth selection)", allf, alle),
              ("light  m1<20", Tf[:, 0] < 20, Te[:, 0] < 20),
              ("mid    20<=m1<29", (Tf[:, 0] >= 20) & (Tf[:, 0] < 29),
                                   (Te[:, 0] >= 20) & (Te[:, 0] < 29)),
              ("HEAVY  m1>=29", Tf[:, 0] >= 29, Te[:, 0] >= 29),
              ("HEAVY & loud (SNR>18)", (Tf[:, 0] >= 29) & (snrf > 18),
                                        (Te[:, 0] >= 29) & (snre > 18))]
    out = {}
    for lab, mf, me in groups:
        if mf.sum() < 200 or me.sum() < 100:
            continue
        mu, sd = Cf[mf].mean(0), Cf[mf].std(0) + 1e-8
        Xf, Xe = (Cf[mf] - mu) / sd, (Ce[me] - mu) / sd
        qtrue = Ye["q"][me]
        row = {"n_fit": int(mf.sum()), "n_test": int(me.sum())}

        # ── probe: marginals, direct q, and q DERIVED from its own m1/m2 ──────
        p, which = {}, {}
        for t in ("m1", "m2", "logMc", "q", "logm1", "logm2"):
            p[t], which[t] = best_probe(Xf, Yf[t][mf], Xe, Ye[t][me])
        # derive in log space: q = exp(log m2 - log m1), no division blow-ups
        q_der = np.exp(np.minimum(p["logm2"] - p["logm1"], 0.0))

        print(f"\n{'='*78}\n{lab}   fit n={int(mf.sum())}  test n={int(me.sum())}")
        print(f"{'='*78}")
        print(f"  {'quantity':34}{'PROBE R2':>10}{'FLOW R2':>10}{'gap':>9}")
        for t, fk in (("m1", "m1"), ("m2", "m2"), ("logMc", "logMc")):
            pr, fl = _r2(Ye[t][me], p[t]), _r2(Ye[t][me], F[fk][me])
            row[t] = dict(probe_r2=pr, flow_r2=fl, probe_model=which[t])
            print(f"  {t + '  (marginal)':34}{pr:>10.3f}{fl:>10.3f}{pr - fl:>+9.3f}")
        pq, fq = _r2(qtrue, p["q"]), _r2(qtrue, F["q_direct"][me])
        row["q_direct"] = dict(probe_r2=pq, flow_r2=fq, probe_model=which["q"])
        print(f"  {'q  DIRECT (asked for q)':34}{pq:>10.3f}{fq:>10.3f}{pq - fq:>+9.3f}")
        pqd = _r2(qtrue, q_der)
        fqd = _r2(qtrue, F["q_from_marginals"][me])
        row["q_derived"] = dict(probe_r2=pqd, flow_r2=fqd)
        print(f"  {'q  DERIVED from own m1,m2':34}{pqd:>10.3f}{fqd:>10.3f}"
              f"{pqd - fqd:>+9.3f}")
        row["flow_corr_m1m2_median"] = float(np.median(F["corr"][me]))
        row["true_corr_m1m2_population"] = float(np.corrcoef(Ye["m1"][me],
                                                            Ye["m2"][me])[0, 1])
        print(f"\n  probe: direct q {pq:.3f} vs derived-from-m1m2 {pqd:.3f}  "
              f"(direct - derived = {pq - pqd:+.3f})")
        print(f"  flow : E[m2/m1] {fq:.3f} (uses joint) vs E[m2]/E[m1] {fqd:.3f} "
              f"(marginals only)  (diff = {fqd - fq:+.3f})")
        print(f"  flow WITHIN-posterior corr(m1,m2) median = "
              f"{row['flow_corr_m1m2_median']:+.3f}   "
              f"(population corr across events = "
              f"{row['true_corr_m1m2_population']:+.3f}; the true PER-EVENT "
              f"posterior corr is NOT measured here -- it needs a reference sampler)")
        out[lab] = row

    print(f"\n{'='*78}\nreading guide")
    print("  probe direct >> derived      -> context holds q beyond m1/m2 marginals")
    print("  probe direct ~= derived      -> the context's q content IS its m1/m2")
    print("  flow E[m2]/E[m1] >> E[m2/m1] -> within-posterior m1-m2 correlation")
    print("                                  destroys q despite good marginals (A)")
    print("  flow m1/m2 ~= probe m1/m2 but flow q << probe q  -> scenario A")
    print("  flow << probe on m1/m2 too                       -> scenario B")
    print("  probe q ~= flow q ~= 0.5 and direct ~= derived    -> scenario C")
    json.dump(out, open(HERE / "m20_joint_structure.json", "w"), indent=2, default=float)
    print(f"\nwrote {HERE / 'm20_joint_structure.json'}")


if __name__ == "__main__":
    main()
