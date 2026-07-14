"""Run the current long_run (ep19) checkpoint on the 6 GWTC events, saving the
FULL intermediate state (context vector, normalized flow samples, per-param
railing, OOD score) for reuse in Steps 3/4/6. PreparedData is cached to disk so
GWOSC is fetched once.

Faithful to pipeline.infer (same preprocessing, same sampling), but exposes the
internals that infer() hides.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "src")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _events import EVENTS, chirp_mass  # noqa: E402
from ahsd.inference.preprocessing import fetch_gwosc, prepare_real, compute_asd_bands  # noqa: E402
from ahsd.inference.pipeline import load_model  # noqa: E402
from ahsd.inference.ood import load_context_stats, score_context  # noqa: E402
from ahsd.models.lean_npe import PARAM_NAMES  # noqa: E402

MODEL = "model/long_run/best_model.pth"
OUTDIR = HERE / "events_ep19"
NSAMP = 3000
SEED = 0


def prepared_for(event, gps, cachedir):
    """Cache prepared.strain + asd_bands per event to avoid refetch."""
    cachedir.mkdir(parents=True, exist_ok=True)
    sf = cachedir / "strain.npy"
    af = cachedir / "asd_bands.npy"
    mf = cachedir / "prep_meta.json"
    if sf.exists() and af.exists() and mf.exists():
        return (np.load(sf), np.load(af), json.load(open(mf)))
    gpsv, raw, starts, srs, found = fetch_gwosc(event)
    prepared = prepare_real(raw, gpsv, starts, srs, seed=SEED)
    bands = compute_asd_bands(prepared, psd_bands=16)
    meta = {"source": prepared.source, "warnings": list(prepared.warnings),
            "detectors_present": found}
    np.save(sf, prepared.strain.astype(np.float32))
    np.save(af, bands.astype(np.float32))
    json.dump(meta, open(mf, "w"))
    return prepared.strain.astype(np.float32), bands.astype(np.float32), meta


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    net, meta = load_model(MODEL)
    dev = meta["device"]
    stats = load_context_stats(meta["model_path"])
    print(f"model epoch {meta['model_epoch']} val_nll {meta['model_val_nll']:.4f} "
          f"dev {dev}\n")

    rows = {}
    for name, ev in EVENTS.items():
        t = time.time()
        strain, bands, pmeta = prepared_for(name, ev["gps"], OUTDIR / name)
        st = torch.from_numpy(strain).unsqueeze(0).to(dev).float()
        ab = torch.from_numpy(bands).unsqueeze(0).to(dev).float() if net.psd_cond else None
        with torch.no_grad():
            ctx = net.encode(st, ab)                       # [1, 256]
            r = torch.zeros(1, dtype=torch.long, device=dev)
            full = net._full_context(ctx, r)
            P = len(PARAM_NAMES)
            fc = full.expand(NSAMP, -1)
            z = torch.randn(NSAMP, P, device=dev)
            y_raw, _ = net.flow.inverse(z, fc)             # normalized, pre-wrap
            y = net.scaler.wrap(y_raw)                     # circular wrap / clamp
            samples = net.scaler.denormalize(y).cpu().numpy()
        y_np = y.cpu().numpy()
        yraw_np = y_raw.cpu().numpy()
        ctx_np = ctx[0].cpu().numpy()

        # enforce mass ordering
        j1, j2 = PARAM_NAMES.index("mass_1"), PARAM_NAMES.index("mass_2")
        a, b = samples[:, j1].copy(), samples[:, j2].copy()
        samples[:, j1], samples[:, j2] = np.maximum(a, b), np.minimum(a, b)

        med = np.median(samples, axis=0)
        rail_any = ((np.abs(y_np) > 0.999) & ~net.scaler.circ_mask.cpu().numpy()
                    ).any(axis=1).mean()
        ood = score_context(ctx_np, stats) if stats is not None else {}

        np.save(OUTDIR / name / "context.npy", ctx_np)
        np.save(OUTDIR / name / "samples_phys.npy", samples)
        np.save(OUTDIR / name / "y_norm.npy", y_np)       # wrapped/clamped
        np.save(OUTDIR / name / "y_raw.npy", yraw_np)     # pre-wrap flow output

        m1, m2 = med[j1], med[j2]
        rows[name] = {
            "m1": float(m1), "m2": float(m2), "Mc": float(chirp_mass(m1, m2)),
            "q": float(m2 / m1), "dL": float(med[PARAM_NAMES.index("luminosity_distance")]),
            "rail": round(float(rail_any), 4),
            "ood_pct": ood.get("percentile_vs_validation"),
            "ood_maha": ood.get("mahalanobis"),
            "source_warnings": pmeta["warnings"],
        }
        print(f"{name:10s} m1 {m1:5.1f}(pub {ev['m1']:.1f}) m2 {m2:5.1f} "
              f"Mc {rows[name]['Mc']:5.1f} dL {rows[name]['dL']:6.0f} | "
              f"rail {rail_any:.2f} ood {rows[name]['ood_pct']} "
              f"maha {rows[name]['ood_maha']}  [{time.time()-t:.1f}s]")

    rows["_meta"] = {"epoch": meta["model_epoch"], "val_nll": meta["model_val_nll"],
                     "n_samples": NSAMP}
    json.dump(rows, open(HERE / "events_ep19.json", "w"), indent=2)


if __name__ == "__main__":
    main()
