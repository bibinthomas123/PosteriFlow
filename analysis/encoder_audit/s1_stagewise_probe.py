#!/usr/bin/env python3
"""S1: WHERE does the encoder lose the information dynesty keeps?

Probes log Mc and q after every stage of the trained CoherentEncoder. The point
is localization: the audit knows the context is information-poor (probe R^2 0.71
for log Mc, 0.11 for q) and that dynesty achieves ~155x sharper Mc on the same
strain. This says at which stage that is lost.

PROBE POWER IS MATCHED ACROSS STAGES. Stage dimensions run from 64 to ~35k, and
an earlier step of this audit reached a wrong conclusion ("MF features carry no
q") precisely by comparing a ~10^4-feature probe at n=300 against a 256-feature
probe at n=1200. Every stage here is reduced to the SAME dimension by a fixed
Gaussian random projection (Johnson-Lindenstrauss preserves inner products, so
linear-probe performance is approximately preserved) and probed with the SAME
estimator at the SAME n. Absolute R^2 is therefore a lower bound; the STAGE-TO-
STAGE DELTA is the quantity of interest.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch
from experiments.remix_data import RemixDataset
from ahsd.models.lean_npe import LeanNPE

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_EVENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
PROJ_DIM = 384
SEED = 0


def main():
    ck = torch.load(ROOT / "model/lean_npe_mf_scratch/best_model.pth",
                    map_location="cpu", weights_only=False)
    m = LeanNPE(psd_cond=True, psd_bands=16, encoder_type="coherent",
                mass_param="mc_q", mf_tokens=True)
    m.load_state_dict(ck["model_state_dict"]); m.to(DEV).eval()
    E = m.encoder

    ds = RemixDataset(str(ROOT / "data/dataset/memmap/validation"), remix=True, seed=3,
                      real_noise_prob=0.0, return_asd_bands=True, psd_bands=16)
    ds.set_epoch(0)

    rng = np.random.default_rng(SEED)
    projs = {}

    def proj(name, x):
        """Fixed random projection to PROJ_DIM (identity if already smaller)."""
        x = x.reshape(x.shape[0], -1).float().cpu().numpy()
        if x.shape[1] <= PROJ_DIM:
            return x
        if name not in projs:
            projs[name] = rng.standard_normal((x.shape[1], PROJ_DIM)).astype(np.float32) / np.sqrt(x.shape[1])
        return x @ projs[name]

    STAGES = ["0_raw_strain", "1_asinh_strain", "2_stem_tokens", "3_tokens_embedded",
              "4_post_fusion", "5_pooled", "6_energy_branch", "7_geom_mf_tokens",
              "8_context"]
    acc = {s: [] for s in STAGES}
    MC, Q, SNR = [], [], []

    B = 16
    for i0 in range(0, N_EVENTS, B):
        items = [ds[i] for i in range(i0, min(i0 + B, N_EVENTS))]
        strain = torch.stack([x[0] for x in items]).to(DEV).float()
        ab = torch.stack([x[4] for x in items]).to(DEV).float()
        pv = torch.stack([x[1] for x in items]).numpy()
        snr = torch.stack([x[3] for x in items]).numpy()

        with torch.no_grad():
            s = torch.nan_to_num(strain, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100, 100)
            acc["0_raw_strain"].append(proj("raw", s))
            xa = torch.asinh(s)
            acc["1_asinh_strain"].append(proj("asinh", xa))

            Bn, D, T = s.shape
            tok = E.stem(xa.reshape(Bn * D, 1, T)).transpose(1, 2)      # [B*D, L, d]
            acc["2_stem_tokens"].append(proj("stem", tok.reshape(Bn, -1)))
            L = tok.shape[1]
            tok = tok + E.pos(L).unsqueeze(0)
            tok = tok.reshape(Bn, D, L, -1)
            tok = tok + E.detector_embed(torch.arange(D, device=DEV))[None, :, None, :]
            tok = tok.reshape(Bn, D * L, -1)
            acc["3_tokens_embedded"].append(proj("emb", tok))

            gtok = E._geometry_rel(s) if hasattr(E, "_geometry_rel") else None
            # rebuild the geometry/MF token block exactly as forward() does
            g = E.geom_to_tokens(E.geom_mlp(gtok)).reshape(Bn, E.n_geom_tokens, E.d_model)
            if E.use_mf_tokens:
                g = g + E.mf_gate * E.mf_to_tokens(E.mf_mlp(E._mf_features(s))
                                                   ).reshape(Bn, E.n_geom_tokens, E.d_model)
            acc["7_geom_mf_tokens"].append(proj("gmf", g))

            fused = E.fusion(torch.cat([g, tok], dim=1))
            acc["4_post_fusion"].append(proj("fus", fused))
            q_ = E.pool_queries.unsqueeze(0).expand(Bn, -1, -1)
            pooled, _ = E.pool_attn(q_, fused, fused)
            acc["5_pooled"].append(proj("pool", pooled))

            w = E.n_energy_windows
            win = s[:, :, : (T // w) * w].reshape(Bn, D, w, -1)
            le = torch.log((win ** 2).mean(dim=-1) + 1e-8)
            acc["6_energy_branch"].append(proj("en", E.energy_mlp(le.reshape(Bn, -1))))

            acc["8_context"].append(proj("ctx", m.encode(strain, ab)))

        for k in range(len(items)):
            m1, m2 = pv[k, 0, 0], pv[k, 0, 1]
            MC.append((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2); Q.append(m2 / m1); SNR.append(snr[k])
        if i0 % 320 == 0:
            print(f"  {i0}/{N_EVENTS}", flush=True)

    X = {s: np.concatenate(v) for s, v in acc.items()}
    MC, Q, SNR = np.array(MC), np.array(Q), np.array(SNR)
    print(f"\nn={len(MC)}  median SNR {np.median(SNR):.1f}  proj_dim={PROJ_DIM}")

    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler

    def r2(Xs, y):
        Xs = StandardScaler().fit_transform(np.nan_to_num(Xs))
        p = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 5, 30)), Xs, y, cv=5)
        return 1 - ((y - p) ** 2).mean() / y.var()

    print("\n" + "=" * 82)
    print(f"{'stage':22s} {'dim':>7s} {'logMc R2':>9s} {'delta':>7s} {'q R2':>8s} {'delta':>7s}")
    print("=" * 82)
    prev_mc = prev_q = None
    for s in STAGES:
        rm, rq = r2(X[s], np.log(MC)), r2(X[s], Q)
        dm = "" if prev_mc is None else f"{rm - prev_mc:+.3f}"
        dq = "" if prev_q is None else f"{rq - prev_q:+.3f}"
        raw_dim = X[s].shape[1]
        print(f"{s:22s} {raw_dim:7d} {rm:9.3f} {dm:>7s} {rq:8.3f} {dq:>7s}")
        if s not in ("6_energy_branch", "7_geom_mf_tokens"):
            prev_mc, prev_q = rm, rq
    print("=" * 82)
    np.savez(Path(__file__).parent / "s1_raw.npz", MC=MC, Q=Q, SNR=SNR,
             **{k: v for k, v in X.items()})


if __name__ == "__main__":
    main()
