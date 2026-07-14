"""
Proof-of-concept: does the CoherentEncoder let the flow convert amplitude ->
distance where the conv encoder is flat?

Short pure-NLL train (rank-0, design-domain memmap data, asd_bands=0), then the
fixed-heavy-BBH amplitude sweep. Success = flow posterior-median dL tracks true
dL (slope>0 / positive Spearman), vs the current conv model's ~0.

Not full training — a directional POC. Full launch command is in the report.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES
from remix_data import RemixDataset
from ahsd.data.bilby_pipeline import BilbySignalInjector, BilbyPreprocessor, get_default_psd

JD = PARAM_NAMES.index("luminosity_distance")
SR, T = 4096, 16384
GW = dict(mass_1=38.8, mass_2=33.4, luminosity_distance=440.0, theta_jn=2.9,
          ra=1.95, dec=-1.27, psi=1.75, phase=1.3, geocent_time=0.0, a1=0.0, a2=0.0)


def amp_sweep(model, device):
    prep = BilbyPreprocessor(SR, 4.0); inj = BilbySignalInjector(SR, 4.0)
    psds = {d: get_default_psd(d) for d in ["H1", "L1", "V1"]}
    zab = torch.zeros(1, 3, 16, device=device)
    grid = [300, 450, 650, 950, 1400, 1900]
    meds, snrs = [], []
    for dL in grid:
        P = dict(GW); P["luminosity_distance"] = float(dL)
        dm = []
        for seed in range(6):
            white, ss = {}, []
            for det in ["H1", "L1"]:
                st, s = inj.inject(np.zeros(T), P, det, psds[det])   # signal-only
                # add design noise
                from ahsd.data.bilby_pipeline import BilbyNoiseGenerator
                white[det] = prep.preprocess(st, psds[det], detector=det); ss.append(s)
            rng = np.random.default_rng(seed)
            s3 = torch.from_numpy(np.stack([white["H1"], white["L1"],
                                 rng.standard_normal(T).astype(np.float32)]))[None].float().to(device)
            with torch.no_grad():
                samp = model.sample_posterior(s3, rank=0, n_samples=800, asd_bands=zab)[0].cpu().numpy()
            dm.append(float(np.median(samp[:, JD])))
        meds.append(float(np.mean(dm))); snrs.append(float(np.sqrt(sum(x**2 for x in ss))))
    slope = float(np.polyfit(np.log(grid), np.log(meds), 1)[0])
    rho = float(spearmanr(grid, meds).correlation)
    return dict(grid=grid, dL_med=[round(m) for m in meds], slope=round(slope, 3),
                spearman=round(rho, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_type", default="coherent")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--subset", type=int, default=8000)
    ap.add_argument("--tag", default="coherent_poc")
    ap.add_argument("--device", default="cpu")   # CPU faster than MPS here (FFT fallback)
    ap.add_argument("--data", default=str(ROOT / "data/dataset/memmap/train"))
    args = ap.parse_args()
    device = args.device

    # direct memmap access (design domain): strain = signal + fresh noise crop
    base = Path(args.data)
    sig = np.load(base / "signals.npy", mmap_mode="r")   # [N,3,T] f16
    noise = np.load(base / "noise.npy", mmap_mode="r")    # [Nn,3,T] f16
    params = np.load(base / "params.npy")                 # [N,11]
    m1, m2 = params[:, 0], params[:, 1]
    bbh = np.where((m1 >= 5) & (m2 >= 5))[0][: args.subset]
    Sig = torch.from_numpy(np.asarray(sig[bbh], dtype=np.float32))   # cache BBH signals
    Pr = torch.from_numpy(params[bbh].astype(np.float32))
    Nn = noise.shape[0]
    print(f"[{args.tag}] train pool={len(bbh)} BBH events, noise pool={Nn}, device={device}", flush=True)

    model = LeanNPE(encoder_type=args.encoder_type, psd_bands=16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    warm = 300; zab = torch.zeros(args.batch, 3, 16, device=device)
    model.train(); t0 = time.time(); rng = np.random.default_rng(0)
    for step in range(1, args.steps + 1):
        b = rng.integers(0, len(Sig), args.batch)
        nb = rng.integers(0, Nn, args.batch)
        strain = (Sig[b] + torch.from_numpy(np.asarray(noise[nb], dtype=np.float32))).to(device)
        params = Pr[b].to(device); asd = zab
        rank = torch.zeros(args.batch, dtype=torch.long, device=device)
        lr = 3e-4 * min(1.0, step / warm)
        for g in opt.param_groups: g["lr"] = lr
        loss = model.nll(strain, params, rank, asd_bands=asd).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % 250 == 0 or step == 1:
            print(f"  step {step:5d}  nll={float(loss):+.3f}  ({(time.time()-t0)/step*1000:.0f} ms/step)", flush=True)

    model.eval()
    sw = amp_sweep(model, device)
    print(f"\n[{args.tag}] AMPLITUDE SWEEP (fixed heavy BBH, design noise):", flush=True)
    for dL, m in zip(sw["grid"], sw["dL_med"]):
        print(f"   true dL={dL:5d} -> flow dL_med={m:5d}", flush=True)
    print(f"   response slope d(log dL_med)/d(log dL) = {sw['slope']:+.2f}  (1=ideal, 0=flat)", flush=True)
    print(f"   Spearman(true dL, pred dL_med)         = {sw['spearman']:+.2f}", flush=True)
    print(f"   [current conv model baseline: slope ~0, Spearman ~0 / slightly negative]", flush=True)

    out = ROOT / "model" / args.tag; out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "args": {"premerger": False, "psd_cond": True, "psd_bands": 16,
                         "encoder_type": args.encoder_type},
                "amp_sweep": sw}, out / "poc_model.pth")
    import json
    json.dump(sw, open(Path(__file__).resolve().parent / f"poc_{args.tag}.json", "w"), indent=2)
    print(f"\nwrote {out/'poc_model.pth'}", flush=True)


if __name__ == "__main__":
    main()
