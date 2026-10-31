"""Offline teacher distillation, stage 1: compute refined "teacher" contexts
for a batch of hard training events against a FROZEN checkpoint snapshot,
and cache them to disk. Stage 2 is train_lean_npe.py's
--offline_distill_cache, which trains against this fixed cache with a cheap
lookup-and-compare loss -- see experiments/semi_amortized.py's module
docstring for why this replaced the live/online semi-amortized trainer
(feedback-loop divergence + per-batch cost).

Because the reference model is frozen for the ENTIRE precompute pass, there
is no feedback loop possible here regardless of how many events are
refined or how long refinement runs -- the risk that made the live version
unsafe simply doesn't exist in this design.

Usage:
  conda run -n ahsd python experiments/precompute_distill_targets.py \
      --checkpoint model/lean_npe_mf_scratch/best_model.pth \
      --data data/dataset --n_events 2000 --steps 20 --manifold_lambda 0.1 \
      --out model/lean_npe_mf_scratch/distill_cache.pt

Re-run periodically against a fresher checkpoint (e.g. every 5-10 epochs of
a training run using the cache) to keep targets from going stale -- doing
this in bulk, occasionally, is still far cheaper than the live version's
every-batch refinement.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from ahsd.models.lean_npe import LeanNPE  # noqa: E402
from semi_amortized import ContextRefiner, ManifoldStats, select_hard_examples  # noqa: E402


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = LeanNPE(psd_cond=a.get("psd_cond", True), psd_bands=a.get("psd_bands", 16),
                    encoder_type=a.get("encoder_type", "coherent"),
                    mass_param=a.get("mass_param", "mc_q"),
                    geom_dropout=a.get("geom_dropout", 0.0),
                    geom_phase_residual=a.get("geom_phase_residual", False),
                    mf_tokens=a.get("mf_tokens", False),
                    mf_shape_norm=a.get("mf_shape_norm", False),
                    mf_token_fusion=a.get("mf_token_fusion", "add"), mf_bank=a.get("mf_bank", "mf_bank.npz"), pol_features=a.get("pol_features", False),
                    q_embed_dim=a.get("q_embed_dim", 0))
    model.load_state_dict(ck["model_state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"loaded {Path(path).name}: epoch {ck.get('epoch')} val_nll {ck.get('val_nll'):.4f}",
          flush=True)
    return model, a


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--n_events", type=int, default=2000,
                    help="how many hard events to refine (basin-cell first, "
                         "then lowest-q fallback to fill the budget)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--manifold_lambda", type=float, default=0.1,
                    help="whitened-PCA manifold penalty during refinement -- "
                         "0.1 keeps results near the population's own spread "
                         "(D~2-4) per analysis/attractor_basin/"
                         "oracle_manifold_tradeoff.py; 0 = unconstrained "
                         "(reaches D~50-100, still recovers q well as a "
                         "ONE-OFF measurement, but a needlessly extreme "
                         "distillation target for a student encoder to chase).")
    ap.add_argument("--manifold_pca_components", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                         else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}", flush=True)

    model, ckpt_args = load_model(args.checkpoint, device)

    from remix_data import RemixDataset, build_memmap_cache
    cache_dir = Path(args.data) / "memmap" / "train"
    if not (cache_dir / "events.json").exists():
        print("building memmap cache (one-time)...", flush=True)
        build_memmap_cache(args.data, "train", str(cache_dir))
    train_ds = RemixDataset(str(cache_dir), remix=False,
                            return_asd_bands=ckpt_args.get("psd_cond", True),
                            psd_bands=ckpt_args.get("psd_bands", 16),
                            return_event_idx=True)
    train_ds.set_epoch(0) if hasattr(train_ds, "set_epoch") else None

    basin_idx = train_ds.basin_cell_event_indices()
    print(f"basin-cell pool: {len(basin_idx)} events", flush=True)
    n_events = min(args.n_events, len(train_ds.events))
    if len(basin_idx) >= n_events:
        rng = np.random.default_rng(0)
        chosen = rng.choice(basin_idx, n_events, replace=False)
    else:
        # fill remaining budget with lowest-q events across the whole dataset
        starts = np.array([e[0] for e in train_ds.events])
        m1 = np.asarray(train_ds.params[starts, 0], dtype=np.float64)
        m2 = np.asarray(train_ds.params[starts, 1], dtype=np.float64)
        q = np.minimum(m1, m2) / np.maximum(m1, m2)
        order = np.argsort(q)
        extra_needed = n_events - len(basin_idx)
        extra = [i for i in order if i not in set(basin_idx.tolist())][:extra_needed]
        chosen = np.concatenate([basin_idx, np.array(extra, dtype=basin_idx.dtype)])
    print(f"refining {len(chosen)} events "
          f"({min(len(basin_idx), n_events)} basin-cell + "
          f"{max(0, n_events - len(basin_idx))} lowest-q fallback)", flush=True)

    manifold_stats = None
    if args.manifold_lambda > 0.0:
        print("fitting manifold reference from a sample of contexts...", flush=True)
        sample_idx = np.random.default_rng(1).choice(len(train_ds.events),
                                                      min(500, len(train_ds.events)), replace=False)
        sample_strains, sample_asd = [], []
        for i in sample_idx:
            item = train_ds[int(i)]
            sample_strains.append(item[0])
            if ckpt_args.get("psd_cond", True):
                sample_asd.append(item[4])
        xs = torch.stack(sample_strains).to(device)
        ab = torch.stack(sample_asd).to(device) if sample_asd else None
        with torch.no_grad():
            sample_ctx = model.encode(xs, ab)
        manifold_stats = ManifoldStats(sample_ctx, n_components=args.manifold_pca_components).to(device)

    refiner = ContextRefiner(model, steps=args.steps, lr=args.lr,
                             patience=args.patience, tol=args.tol,
                             manifold_lambda=args.manifold_lambda,
                             manifold_stats=manifold_stats, diag_samples=32)

    cache = {}
    t0 = time.time()
    for start in range(0, len(chosen), args.batch):
        batch_event_idx = chosen[start:start + args.batch]
        strains, params_list, asd_list = [], [], []
        for i in batch_event_idx:
            item = train_ds[int(i)]
            strains.append(item[0])
            params_list.append(item[1])
            if ckpt_args.get("psd_cond", True):
                asd_list.append(item[4])
        xs = torch.stack(strains).to(device).float()
        ab = torch.stack(asd_list).to(device).float() if asd_list else None
        params_true = torch.stack(params_list).to(device)[:, 0, :]  # rank-0 truth, physical
        rank0 = torch.zeros(xs.shape[0], dtype=torch.long, device=device)

        with torch.no_grad():
            context0 = model.encode(xs, ab)
        result = refiner.refine(context0, rank0, params_true)

        for j, i in enumerate(batch_event_idx):
            cache[int(i)] = result.context_star[j].cpu()

        elapsed = time.time() - t0
        done = start + len(batch_event_idx)
        print(f"  {done}/{len(chosen)}  ({elapsed:.0f}s)  "
              f"nll {result.nll_initial.mean():.2f}->{result.nll_final.mean():.2f}  "
              f"success {result.success.float().mean():.2f}  "
              f"refine_dist {result.refine_distance.mean():.2f}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"targets": cache, "source_checkpoint": args.checkpoint,
               "source_epoch": None, "n_events": len(cache),
               "manifold_lambda": args.manifold_lambda,
               "context_dim": model.context_dim}, out_path)
    print(f"\nsaved {len(cache)} refined targets -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
