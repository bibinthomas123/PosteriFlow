#!/usr/bin/env python3
"""
Train LeanNPE with a pure-NLL objective and per-epoch conditional-inference
diagnostics.

The diagnostics answer, every epoch, the questions that matter (and that the
previous 22.7M-param model failed: see analysis/context_conditioning_test.json):

  - shuffle_delta_nll : val NLL with contexts shuffled across events minus
                        matched-context NLL. ~0 => marginal fit. Should grow.
  - dist_corr         : corr(posterior median distance, true distance).
  - cov50/cov90       : empirical credible-interval coverage for distance
                        (should approach 0.50 / 0.90, not collapse below).

Usage:
    conda run -n ahsd python experiments/train_lean_npe.py \
        --data data/dataset --outdir model/lean_npe --epochs 60 --batch 128
"""

import argparse, glob, json, logging, math, pickle, sys, time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_lean_npe")

MAX_SIGNALS = 5
T_LEN = 16384


def load_split(data_dir: str, split: str, limit: int | None = None):
    """Load a split into compact tensors: strain fp16, params, n_signals."""
    files = sorted(glob.glob(f"{data_dir}/{split}/batch_*.pkl"))
    strains, params, nsigs = [], [], []
    for fp in files:
        with open(fp, "rb") as fh:
            b = pickle.load(fh)
        for s in (b["samples"] if isinstance(b, dict) else b):
            if str(s.get("event_type", "")) == "noise":
                continue
            plist = s.get("parameters", [])
            if not plist:
                continue
            dd = s["detector_data"]
            st = np.stack([np.asarray(dd[d]["strain"], dtype=np.float32) for d in ("H1", "L1", "V1")])
            if st.shape[1] != T_LEN:
                continue
            # Rank-conditioning requires a CONSISTENT signal order, but the
            # generator stores overlapping signals in random order (measured:
            # first-listed is loudest only 49% of the time). Sort by a
            # deterministic loudness proxy, amplitude ~ Mc^(5/6)/d_L, so.  
            # "rank r" means the same thing for every event.
            def _loudness(p):
                m1, m2 = p.get("mass_1", 1.0), p.get("mass_2", 1.0)
                mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
                return mc ** (5.0 / 6.0) / max(p.get("luminosity_distance", 1e9), 1.0)
            plist = sorted(plist, key=_loudness, reverse=True)
            pv = np.zeros((MAX_SIGNALS, len(PARAM_NAMES)), dtype=np.float32)
            n = min(len(plist), MAX_SIGNALS)
            ok = True
            for i in range(n):
                for j, k in enumerate(PARAM_NAMES):
                    v = plist[i].get(k)
                    if v is None:
                        ok = False
                        break
                    pv[i, j] = v
                if not ok:
                    break
            if not ok:
                continue
            strains.append(torch.from_numpy(st).to(torch.float16))
            params.append(torch.from_numpy(pv))
            nsigs.append(n)
            if limit and len(strains) >= limit:
                break
        if limit and len(strains) >= limit:
            break
    return torch.stack(strains), torch.stack(params), torch.tensor(nsigs, dtype=torch.long)


class EventDataset(Dataset):
    def __init__(self, strain, params, nsig):
        self.strain, self.params, self.nsig = strain, params, nsig

    def __len__(self):
        return self.strain.shape[0]

    def __getitem__(self, i):
        return self.strain[i], self.params[i], self.nsig[i]


def batch_nll(model, strain, params, nsig):
    """Mean per-signal NLL for a batch of events. Encoder runs ONCE per event;
    the flow is queried once per signal rank present in the batch."""
    context = model.encoder(strain)
    total, count = 0.0, 0
    losses = []
    max_n = int(nsig.max().item())
    for r in range(max_n):
        mask = nsig > r
        if not mask.any():
            continue
        idx = mask.nonzero(as_tuple=True)[0]
        rank = torch.full((idx.numel(),), r, dtype=torch.long, device=strain.device)
        nll = model.nll(None, params[idx, r, :], rank, context=context[idx])
        losses.append(nll.sum())
        count += idx.numel()
    return torch.stack(losses).sum() / count


@torch.no_grad()
def run_diagnostics(model, strain, params, nsig, device, n_events=256, n_post=128):
    """Conditional-inference metrics on validation events (rank 0 / primary)."""
    model.eval()
    n = min(n_events, strain.shape[0])
    sub = strain[:n].to(device).float()
    p0 = params[:n, 0, :].to(device)
    rank0 = torch.zeros(n, dtype=torch.long, device=device)

    ctx = []
    for i in range(0, n, 64):
        ctx.append(model.encoder(sub[i:i + 64]))
    ctx = torch.cat(ctx)

    nll_true = model.nll(None, p0, rank0, context=ctx)
    perm = torch.randperm(n, device=device)
    nll_shuf = model.nll(None, p0, rank0, context=ctx[perm])
    shuffle_delta = (nll_shuf.mean() - nll_true.mean()).item()

    # posterior samples for distance metrics (all params' coverage at 50/90)
    full_ctx = model._full_context(ctx, rank0)
    ctx_rep = full_ctx.unsqueeze(1).expand(n, n_post, full_ctx.shape[1]).reshape(n * n_post, -1)
    z = torch.randn(n * n_post, len(PARAM_NAMES), device=device)
    y, _ = model.flow.inverse(z, ctx_rep)
    samp = model.scaler.denormalize(y.reshape(n, n_post, -1))  # [n, n_post, 11]

    d_med = samp[:, :, 2].median(dim=1).values
    d_true = p0[:, 2]
    dist_corr = torch.corrcoef(torch.stack([torch.log(d_med), torch.log(d_true)]))[0, 1].item()

    lo50, hi50 = samp.quantile(0.25, dim=1), samp.quantile(0.75, dim=1)
    lo90, hi90 = samp.quantile(0.05, dim=1), samp.quantile(0.95, dim=1)
    cov50 = ((p0 >= lo50) & (p0 <= hi50)).float().mean(dim=0)
    cov90 = ((p0 >= lo90) & (p0 <= hi90)).float().mean(dim=0)

    j = PARAM_NAMES.index("luminosity_distance")
    return {
        "val_nll_diag": nll_true.mean().item(),
        "shuffle_delta_nll": shuffle_delta,
        "dist_corr": dist_corr,
        "dist_cov50": cov50[j].item(),
        "dist_cov90": cov90[j].item(),
        "cov50_all": {k: round(cov50[i].item(), 3) for i, k in enumerate(PARAM_NAMES)},
        "cov90_all": {k: round(cov90[i].item(), 3) for i, k in enumerate(PARAM_NAMES)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--outdir", default="model/lean_npe")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int, default=300)
    ap.add_argument("--limit_train", type=int, default=None, help="cap train events (smoke tests)")
    ap.add_argument("--premerger", action="store_true",
                    help="widen geocent_time range for pre-merger (early-warning) datasets")
    ap.add_argument("--remix", action="store_true",
                    help="v2 component dataset: remix noise/shift/distance on the fly "
                         "(builds a memmap cache under <data>/memmap on first use)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--noise_bank", default=None,
                    help="real-noise bank dir (e.g. data/noise_bank); enables real-noise mixing")
    ap.add_argument("--real_noise_prob", type=float, default=0.0,
                    help="per-event probability of real noise instead of Gaussian (needs --noise_bank)")
    ap.add_argument("--init_from", default=None,
                    help="checkpoint to fine-tune from (loads weights; fresh optimizer/schedule)")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log.info("Loading data...")
    if args.remix:
        from v2_remix_data import RemixDataset, build_memmap_cache
        cache = Path(args.data) / "memmap" / "train"
        if not (cache / "events.json").exists():
            log.info("Building memmap cache (one-time)...")
            build_memmap_cache(args.data, "train", str(cache))
        train_ds = RemixDataset(str(cache), remix=True,
                                real_noise_dir=args.noise_bank,
                                real_noise_prob=args.real_noise_prob)
        log.info(f"remix train events={len(train_ds)} (noise pool={train_ds.n_noise}, "
                 f"p_real={args.real_noise_prob})")
    else:
        tr_s, tr_p, tr_n = load_split(args.data, "train", args.limit_train)
        train_ds = EventDataset(tr_s, tr_p, tr_n)
        log.info(f"train events={len(train_ds)}")
    # validation is always FIXED (stored strain, no remixing) for comparability
    va_s, va_p, va_n = load_split(args.data, "validation")
    log.info(f"val events={va_s.shape[0]}")

    # Fixed REAL-NOISE validation variant: same validation events, noise
    # replaced by deterministic (seeded, epoch-independent) real crops with
    # signals re-colored to each crop's whitening. Materialized once so the
    # per-epoch real-noise metrics are exactly comparable across epochs.
    va_real = None
    if args.noise_bank:
        from v2_remix_data import RemixDataset as _RD, build_memmap_cache as _bc
        vcache = Path(args.data) / "memmap" / "validation"
        if not (vcache / "events.json").exists():
            _bc(args.data, "validation", str(vcache))
        vds = _RD(str(vcache), remix=False, seed=1234,
                  real_noise_dir=args.noise_bank, real_noise_prob=1.0)
        vs, vp, vn = [], [], []
        for i in range(len(vds)):
            s, p, n = vds[i]
            vs.append(s.to(torch.float16)); vp.append(p); vn.append(n)
        va_real = (torch.stack(vs), torch.stack(vp), torch.stack(vn))
        log.info(f"real-noise validation materialized: {len(vds)} events")

    model = LeanNPE(premerger=args.premerger).to(device)
    if args.init_from:
        init = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(init["model_state_dict"])
        log.info(f"fine-tuning from {args.init_from} "
                 f"(epoch {init.get('epoch')}, val_nll {init.get('val_nll'):.4f})")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"LeanNPE parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps_per_epoch = math.ceil(len(train_ds) / args.batch)
    total_steps = steps_per_epoch * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # persistent_workers must stay False: workers snapshot the dataset, and
    # set_epoch() (fresh noise pairings each epoch) must reach them — workers
    # are respawned per epoch and pick up the bumped epoch counter.
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=(args.workers if args.remix else 0))
    val_loader = DataLoader(EventDataset(va_s, va_p, va_n), batch_size=args.batch,
                            shuffle=False, num_workers=0)

    use_wandb = False
    if args.wandb:
        try:
            import wandb
            wandb.init(project="posteriflow-lean-npe", config=vars(args))
            use_wandb = True
        except Exception as e:
            log.warning(f"wandb disabled: {e}")

    best_val = float("inf")
    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        model.train()
        t0 = time.time()
        tr_losses, gnorms = [], []
        try:
            from tqdm import tqdm
            train_iter = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        except ImportError:
            train_iter = train_loader
        for strain, params, nsig in train_iter:
            strain = strain.to(device).float()
            params = params.to(device)
            nsig = nsig.to(device)
            loss = batch_nll(model, strain, params, nsig)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            global_step += 1
            tr_losses.append(loss.item())
            gnorms.append(float(gn))
            if hasattr(train_iter, "set_postfix"):
                train_iter.set_postfix(nll=f"{tr_losses[-1]:.3f}", gnorm=f"{gnorms[-1]:.1f}")

        model.eval()
        va_losses = []
        with torch.no_grad():
            for strain, params, nsig in val_loader:
                strain = strain.to(device).float()
                va_losses.append(batch_nll(model, strain, params.to(device), nsig.to(device)).item())
        val_nll = float(np.mean(va_losses))

        diag = run_diagnostics(model, va_s, va_p, va_n, device)
        diag_real = {}
        if va_real is not None:
            dr = run_diagnostics(model, *va_real, device)
            diag_real = {f"real_{k}": v for k, v in dr.items() if not isinstance(v, dict)}
        dt = time.time() - t0
        rec = {
            "epoch": epoch, "train_nll": float(np.mean(tr_losses)), "val_nll": val_nll,
            "grad_norm": float(np.mean(gnorms)), "lr": sched.get_last_lr()[0],
            "epoch_seconds": round(dt, 1), **{k: v for k, v in diag.items() if not isinstance(v, dict)},
            **diag_real,
        }
        history.append({**rec, "cov50_all": diag["cov50_all"], "cov90_all": diag["cov90_all"]})
        real_str = ""
        if diag_real:
            real_str = (f" | REAL val {rec['real_val_nll_diag']:.2f} "
                        f"dcorr {rec['real_dist_corr']:+.2f} "
                        f"dcov {rec['real_dist_cov50']:.2f}/{rec['real_dist_cov90']:.2f}")
        log.info(
            f"epoch {epoch:3d} | train {rec['train_nll']:.3f} | val {val_nll:.3f} | "
            f"shufΔ {rec['shuffle_delta_nll']:+.3f} | dcorr {rec['dist_corr']:+.3f} | "
            f"dcov50/90 {rec['dist_cov50']:.2f}/{rec['dist_cov90']:.2f} | "
            f"gnorm {rec['grad_norm']:.1f} | {dt:.0f}s{real_str}"
        )
        if use_wandb:
            import wandb
            wandb.log(rec)

        # selection metric: Gaussian val NLL alone (legacy), or the mean of
        # Gaussian and real-noise val NLL when a noise bank is configured —
        # the deployment goal is robustness on real data WITHOUT losing
        # simulated-data performance, so both count equally.
        select = val_nll if va_real is None else 0.5 * (val_nll + rec["real_val_nll_diag"])
        if select < best_val:
            best_val = select
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "val_nll": val_nll, "diagnostics": diag, "args": vars(args),
            }, outdir / "best_model.pth")
        with open(outdir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    log.info(f"done. best val NLL {best_val:.4f} -> {outdir/'best_model.pth'}")


if __name__ == "__main__":
    main()
