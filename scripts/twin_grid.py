"""Bias map: twin injections over (chirp mass x mass ratio x inclination),
fixed SNR ~ 24, gaussian training noise. Output: median pred/true per cell."""
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import numpy as np, torch, json
from remix_data import RemixDataset, T_LEN
from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES
from ahsd.data.bilby_pipeline import (BilbySignalInjector, BilbyPreprocessor,
                                      get_default_psd, _GPS_REF)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "model/lean_npe_v3/best_model.pth"
OUT = sys.argv[2] if len(sys.argv) > 2 else "analysis/twin_grid.json"
ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
model = LeanNPE(premerger=ckpt["args"].get("premerger", False), psd_cond=ckpt["args"].get("psd_cond", False) or False, psd_bands=ckpt["args"].get("psd_bands", 16), encoder_type=ckpt["args"].get("encoder_type", "conv"))
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
print(f"grid on {MODEL} (epoch {ckpt['epoch']})", flush=True)

inj = BilbySignalInjector(4096, 4.0); prep = BilbyPreprocessor(4096, 4.0)
psds = {d: get_default_psd(d) for d in ("H1","L1","V1")}
ds = RemixDataset("data/dataset/memmap/validation", remix=False, seed=1234)
rng = np.random.default_rng(11)
j1, j2, jd = [PARAM_NAMES.index(k) for k in ("mass_1","mass_2","luminosity_distance")]

MC = [10, 20, 30, 45]        # chirp mass
Q  = [0.4, 0.7, 0.9]         # mass ratio
TH = [0.5, 1.57, 2.9]        # inclination
R, K, TARGET_SNR = 12, 300, 24.0
rows = []
for mc in MC:
    for q in Q:
        # component masses from (Mc, q)
        m1 = mc * (1+q)**0.2 / q**0.6
        m2 = q * m1
        if not (5 <= m2 <= m1 <= 100):
            continue
        for th in TH:
            p = dict(mass_1=m1, mass_2=m2, luminosity_distance=800.0,
                     ra=float(rng.uniform(0,2*np.pi)), dec=float(np.arcsin(rng.uniform(-1,1))),
                     theta_jn=th, psi=float(rng.uniform(0,np.pi)),
                     phase=float(rng.uniform(0,2*np.pi)), geocent_time=0.0,
                     geocent_time_gps=_GPS_REF, a1=0.1, a2=0.1, event_type="BBH")
            wsig, snrs = [], []
            for d in ("H1","L1","V1"):
                s, snr = inj.inject(np.zeros(T_LEN), p, d, psds[d])
                wsig.append(prep.preprocess(s, psds[d], detector=d)); snrs.append(snr)
            sig3 = np.stack(wsig); snr0 = float(np.sqrt(sum(s**2 for s in snrs)))
            if snr0 < 1e-3:
                continue
            sc = TARGET_SNR / snr0
            d_true = 800.0 / sc
            if not (50 < d_true < 2100):
                sc = np.clip(sc, 800/2100, 800/50); d_true = 800.0/sc
            S = [ds.noise[int(rng.integers(ds.n_noise))].astype(np.float32) + sc*sig3
                 for _ in range(R)]
            st = torch.from_numpy(np.stack(S)).float()
            samp = model.sample_posterior(st, rank=0, n_samples=K).reshape(R*K, 11).numpy()
            med = np.median(samp, axis=0)
            mc_pred = (med[j1]*med[j2])**0.6 / (med[j1]+med[j2])**0.2
            rows.append({"mc": mc, "q": q, "theta_jn": th, "snr": TARGET_SNR,
                         "d_true": round(d_true,0),
                         "m1_ratio": round(float(med[j1]/m1),3),
                         "mc_ratio": round(float(mc_pred/mc),3),
                         "dl_ratio": round(float(med[jd]/d_true),3)})
            print(rows[-1], flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
print(f"-> {OUT}", flush=True)
