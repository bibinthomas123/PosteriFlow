#!/usr/bin/env python3
"""
Train LeanNPE with a pure-NLL objective, per-epoch conditional-inference
diagnostics, and CALIBRATION-GATED checkpoint selection.

Every epoch the diagnostics answer the questions that matter (and that the
previous 22.7M-param model failed: see analysis/context_conditioning_test.json):

  - shuffle_delta_nll : val NLL with contexts shuffled across events minus
                        matched-context NLL. ~0 => marginal fit. Should grow.
  - dist_corr         : corr(posterior median distance, true distance).
  - dist_cov50/90     : credible-interval coverage for distance (-> 0.50 / 0.90).

They also report the calibration metrics that val NLL is BLIND to — because MLE
keeps lowering NLL by over-sharpening the flow long after the model stops
improving (see analysis/divergence_localization):

  - spurious_railing  : fraction of samples pinned at a prior bound whose truth
                        is not near it (the overconfidence signature).
  - base_conc         : E||z||^2/D of the truth in base space; 1.0 = calibrated,
                        >1 = overconfident.
  - sci_cov90[_highsnr]: mass/distance 90% coverage, all events / high-SNR subset.
  - sbc_pass_frac     : fraction of params passing an SBC uniformity test.

`best_model.pth` is the lowest-val-NLL epoch that still passes the calibration
gate (`spurious_railing <= --max_spurious_railing`), so continued training
cannot drift the saved checkpoint into the overconfident regime.

Usage:
    conda run -n ahsd python experiments/train_lean_npe.py \
        --data data/dataset --remix --outdir model/lean_npe --epochs 40 --batch 128 \
        --encoder_type coherent --psd_cond --noise_bank data/noise_bank \
        --real_noise_prob 0.9 --ckpt_every 2
"""

import argparse, glob, json, logging, math, pickle, sys, time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES, ParamScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_lean_npe")
# bilby logs each WaveformGenerator instantiation (basin probe, delay aug) at
# INFO via its own handler, flooding the per-epoch progress output.
logging.getLogger("bilby").setLevel(logging.WARNING)

MAX_SIGNALS = 5
T_LEN = 16384


def load_split(data_dir: str, split: str, limit: int | None = None):
    """Load a split into compact tensors: strain fp16, params, n_signals,
    network SNR (for the fixed non-remix path the stored SNR is exact; the
    remix path recomputes SNR on the fly since it rescales/re-colors)."""
    files = sorted(glob.glob(f"{data_dir}/{split}/batch_*.pkl"))
    strains, params, nsigs, snrs = [], [], [], []
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
                """Deterministic loudness proxy (amplitude ~ Mc^(5/6)/d_L) used
                to order overlapping signals so rank labels are consistent."""
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
            snrs.append(float(s.get("network_snr", 0.0)))
            if limit and len(strains) >= limit:
                break
        if limit and len(strains) >= limit:
            break
    return (torch.stack(strains), torch.stack(params),
            torch.tensor(nsigs, dtype=torch.long),
            torch.tensor(snrs, dtype=torch.float32))


def audit_scaler_support(train_ds, args, log):
    """Report training targets that fall OUTSIDE ParamScaler's declared ranges.

    `ParamScaler.normalize()` ends in `.clamp(-1, 1)`, so an out-of-range target
    is silently relabelled to the boundary and the flow is trained to put density
    there. Nothing else in the pipeline surfaces this: the NLL stays finite, SBC
    and coverage stay green (a boundary spike is still *calibrated*), so the bias
    is invisible to every existing gate.

    Measured on the shipped 50k dataset (2026-07-21 audit): luminosity_distance
    has 2.28% of targets below the scaler's 40 Mpc floor, because the generator
    emits down to ~10 Mpc while ParamScaler.RANGES declares [40, 2200]. Every one
    of those events teaches the flow that a 10 Mpc source sits at exactly 40 Mpc.

    This is a WARNING, not an error: the fix is to widen ParamScaler.RANGES, but
    that changes the physical<->normalized map and therefore silently invalidates
    every existing checkpoint, so it must be a deliberate, coordinated change.
    """
    from ahsd.models.lean_npe import ParamScaler

    params = getattr(train_ds, "params", None)
    if params is None:
        return
    p = np.asarray(params[:200000], dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < len(PARAM_NAMES):
        return

    scaler = ParamScaler(PARAM_NAMES, premerger=args.premerger,
                         mass_param=args.mass_param, dist_range=args.dist_range)
    # Audit in the PHYSICAL parameters the dataset actually stores. The mc_q
    # internal coordinates are a bijection of (m1, m2), so checking m1/m2 covers
    # the mass pair under either mass_param.
    offenders = []
    for i, name in enumerate(PARAM_NAMES):
        lo, hi, _ = ParamScaler.RANGES[name]
        if name == "geocent_time" and args.premerger:
            lo, hi = -1.6, 5.2
        if name == "luminosity_distance":
            lo, hi = scaler.dist_range
        v = p[:, i]
        frac_lo = float((v < lo).mean())
        frac_hi = float((v > hi).mean())
        if frac_lo + frac_hi > 1e-4:
            offenders.append((name, lo, hi, frac_lo, frac_hi, float(v.min()), float(v.max())))

    if not offenders:
        log.info(f"scaler support audit: all {len(PARAM_NAMES)} params within "
                 f"ParamScaler ranges (n={len(p)}) — no targets clamped")
        return

    log.warning("=" * 78)
    log.warning("SCALER SUPPORT AUDIT: training targets outside ParamScaler.RANGES")
    log.warning("these are CLAMPED to the boundary by normalize() and train the")
    log.warning("flow to place density at the bound — a silent, calibrated bias.")
    for name, lo, hi, f_lo, f_hi, vmin, vmax in offenders:
        log.warning(f"  {name:20s} range=[{lo:g}, {hi:g}]  "
                    f"below={f_lo*100:.2f}%  above={f_hi*100:.2f}%  "
                    f"data=[{vmin:g}, {vmax:g}]")
    log.warning("=" * 78)
    _ = scaler  # constructed to validate mass_param/premerger consistency


class EventDataset(Dataset):
    """In-memory dataset of pre-loaded events (the non-remix path). Yields
    (strain, params, n_signals, net_snr) tuples, matching RemixDataset's shape
    so the two are drop-in interchangeable in the training/validation loop."""

    def __init__(self, strain, params, nsig, snr=None):
        self.strain, self.params, self.nsig = strain, params, nsig
        # 4th element kept for tuple-shape parity with RemixDataset (net_snr);
        # unused by the pure-NLL loss.
        self.snr = snr if snr is not None else torch.zeros(len(nsig))

    def __len__(self):
        return self.strain.shape[0]

    def __getitem__(self, i):
        return self.strain[i], self.params[i], self.nsig[i], self.snr[i]


def multiview_consistency_loss(model, views_batch, device, psd_cond):
    """views_batch: G groups x K noise-views of the same underlying signal
    (RemixDataset.get_multiview). Encodes all G*K views and penalizes
    within-group context variance around the group mean -- "same BBH,
    different noise draw -> same encoder output". Targets the mechanism
    found by token grafting on GW150914 (analysis/attractor_basin addendum,
    2026-07-20): the MF-token branch's learned function is sensitive to the
    fine SNR-landscape shape of a specific noise realization, and that
    sensitivity, not the astrophysical content, carried the collapse.
    Encoder-only (no flow query) -- cheap relative to the NLL pass."""
    strains, asds = [], []
    for views in views_batch:
        for v in views:
            strains.append(v[0])
            if psd_cond:
                asds.append(v[4])
    strain = torch.stack(strains).to(device).float()
    ab = torch.stack(asds).to(device).float() if psd_cond else None
    ctx = model.encode(strain, ab)                              # [G*K, C]
    G, K = len(views_batch), len(views_batch[0])
    ctx = ctx.reshape(G, K, -1)
    mean = ctx.mean(dim=1, keepdim=True)
    return ((ctx - mean) ** 2).mean()


def batch_nll(model, strain, params, nsig, asd_bands=None,
              aux_head=None, aux_w=0.0, aux_q_embed_w=0.0, return_context=False):
    """Mean per-signal NLL for a batch of events. Encoder runs ONCE per event;
    the flow is queried once per signal rank present in the batch.

    asd_bands: [B, n_det, psd_bands] sensitivity summary, required when the
    model is PSD-conditioned (ignored otherwise).

    aux_head/aux_w (train only): auxiliary mass regression ON THE CONTEXT —
    SmoothL1 between aux_head(context) and the rank-0 signal's normalized
    mass coordinates (Mc,q for mc_q models). The (30,30) attractor is
    LOCALIZED CONTEXT COLLAPSE (analysis/attractor_basin round 6/8: every
    event in the cell gets one context vector, so the flow can only return
    the cell average); an 11-dim NLL barely notices one flattened coordinate
    on a thin cell, but a direct regression through the context cannot be
    satisfied unless contexts SEPARATE by mass within the cell. Near-zero on
    already-resolved events, so the gradient self-targets the collapsed
    ones."""
    context = model.encode(strain, asd_bands)
    count = 0
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
    loss = torch.stack(losses).sum() / count
    if aux_head is not None and aux_w > 0.0:
        # .contiguous(): the column slice is strided and MPS smooth_l1_loss
        # rejects non-contiguous inputs (CPU tolerates them)
        y0 = model.scaler.normalize(params[:, 0, :])[:, :2].contiguous()
        loss = loss + aux_w * torch.nn.functional.smooth_l1_loss(aux_head(context), y0)
    if model.q_embed_dim > 0 and aux_q_embed_w > 0.0:
        # Same target/loss form as --aux_mass, but through the DEDICATED
        # embedding that is actually concatenated onto the flow's context
        # (see LeanNPE._full_context) -- this head is saved in the
        # checkpoint and live at inference, unlike aux_head above.
        y0q = model.scaler.normalize(params[:, 0, :])[:, :2].contiguous()
        q_pred = model.q_pred_head(model.q_embed_head(context))
        loss = loss + aux_q_embed_w * torch.nn.functional.smooth_l1_loss(q_pred, y0q)
    if return_context:
        return loss, context
    return loss


@torch.no_grad()
def run_diagnostics(model, strain, params, nsig, device, n_events=256, n_post=128,
                    asd_bands=None, snr=None):
    """Conditional-inference metrics on validation events (rank 0 / primary).
    asd_bands: [N, n_det, psd_bands] sensitivity summary for PSD-conditioned
    models (None otherwise).
    snr: [N] network SNR per event; enables the high-SNR coverage split that
    exposes overconfidence (see analysis/divergence_localization)."""
    model.eval()
    n = min(n_events, strain.shape[0])
    sub = strain[:n].to(device).float()
    ab = asd_bands[:n].to(device).float() if asd_bands is not None else None
    p0 = params[:n, 0, :].to(device)
    rank0 = torch.zeros(n, dtype=torch.long, device=device)

    ctx = []
    for i in range(0, n, 64):
        ctx.append(model.encode(sub[i:i + 64], ab[i:i + 64] if ab is not None else None))
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

    # --- calibration metrics that val NLL is BLIND to (analysis/
    #     divergence_localization): MLE keeps lowering NLL by over-sharpening the
    #     flow (boundary railing + high-SNR under-coverage) long after the model
    #     stops improving. Gate/early-stop on these, not on NLL. ---
    noncirc = ~model.scaler.circ_mask
    y_true = model.scaler.normalize(p0)                            # [n, 11]
    yw = model.scaler.wrap(y).reshape(n, n_post, -1)              # normalized, wrapped
    yt = y_true.unsqueeze(1)
    # SPURIOUS railing: pinned at a bound whose truth is NOT near that bound
    spurious = (((yw > 0.999) & (yt < 0.9)) | ((yw < -0.999) & (yt > -0.9)))
    spurious_railing = spurious[:, :, noncirc].any(-1).float().mean().item()
    # base-space concentration: calibrated flow maps truths to N(0,I) -> 1.0;
    # >1 == overconfident (truths pushed into the base tails)
    z_true, _ = model.flow.forward(y_true, full_ctx)
    base_conc = (z_true.pow(2).sum(1).mean() / len(PARAM_NAMES)).item()
    # science-param (m1,m2,dL) 90% coverage, all events + high-SNR subset
    sci = [PARAM_NAMES.index(x) for x in ("mass_1", "mass_2", "luminosity_distance")]
    in90 = ((p0 >= lo90) & (p0 <= hi90)).float()                  # [n, 11]
    sci_cov90 = in90[:, sci].mean().item()
    if snr is not None:
        hs = snr[:n].to(device).float() > 20.0
        sci_cov90_highsnr = in90[hs][:, sci].mean().item() if hs.any() else sci_cov90
    else:
        sci_cov90_highsnr = sci_cov90
    # SBC: fraction of params whose truth-quantile is ~uniform (KS p > 1e-3)
    q = (samp < p0.unsqueeze(1)).float().mean(1).cpu().numpy()    # [n, 11]
    try:
        from scipy.stats import kstest
        sbc_pass_frac = float(np.mean([
            kstest(np.clip(q[:, k], 1e-6, 1 - 1e-6), "uniform").pvalue > 1e-3
            for k in range(len(PARAM_NAMES))]))
    except Exception:
        sbc_pass_frac = float("nan")

    # --- SUPPORT VIOLATIONS: the training data satisfies mass_1 >= mass_2 in
    #     100% of events, so any posterior mass at m1 < m2 sits where the target
    #     density is exactly ZERO. Free to compute and it was the loudest signal
    #     in the whole post-mortem (analysis/.../mechanism/m22: 27.7% on
    #     GW150914 vs 2.1% in-distribution) yet nothing ever looked at it.
    #     Structurally 0 for mass_param="mc_q"; keep reporting it as the check
    #     that the guarantee holds. ---
    m1s, m2s = samp[:, :, 0], samp[:, :, 1]
    support_viol = (m1s < m2s).float().mean().item()

    # --- CORNER BENCHMARK: population averages are prior-weighted and were ALL
    #     green (SBC 11/11, cov90 0.893, base_conc 1.01) while heavy+near-equal
    #     +loud events were catastrophically wrong. That corner is a thin slice
    #     of the prior but carries the science, so report it explicitly, always.
    #     Nested regions, each a subset of the last. ---
    q_pred = (torch.minimum(m1s, m2s) / torch.maximum(m1s, m2s)).median(dim=1).values
    q_true = torch.minimum(p0[:, 0], p0[:, 1]) / torch.maximum(p0[:, 0], p0[:, 1])
    frac_q95 = ((torch.minimum(m1s, m2s) / torch.maximum(m1s, m2s)) > 0.95).float().mean(1)
    viol_ev = (m1s < m2s).float().mean(1)
    rail_ev = spurious[:, :, noncirc].any(-1).float().mean(1)
    in90_ev = in90[:, sci].mean(1)
    snr_t = snr[:n].to(device).float() if snr is not None else torch.full((n,), 99.0, device=device)

    regions = [
        ("light  m1<20",             p0[:, 0] < 20),
        ("heavy  m1>=29",            p0[:, 0] >= 29),
        ("heavy + q>0.8",            (p0[:, 0] >= 29) & (q_true > 0.8)),
        ("heavy + q>0.8 + SNR>18",   (p0[:, 0] >= 29) & (q_true > 0.8) & (snr_t > 18)),
    ]
    corner = {}
    for lab, m in regions:
        if int(m.sum()) < 5:                       # too few to mean anything
            corner[lab] = {"n": int(m.sum())}
            continue
        # q R^2 is reported for continuity but is NOT meaningful inside a
        # truth-selected region: selecting on q shrinks Var(q), so the R^2
        # denominator collapses (q>0.8 leaves almost no spread). q_bias and
        # frac_q_gt95 are the honest attractor readouts here.
        qt_, qp_ = q_true[m], q_pred[m]
        ss_tot = ((qt_ - qt_.mean()) ** 2).sum()
        r2 = (1 - ((qt_ - qp_) ** 2).sum() / ss_tot).item() if ss_tot > 1e-8 else float("nan")
        corner[lab] = {
            "n": int(m.sum()),
            "cov90_sci": round(in90_ev[m].mean().item(), 3),
            "q_bias": round((qp_ - qt_).median().item(), 3),
            "q_abs_err": round((qp_ - qt_).abs().median().item(), 3),
            "q_r2": round(r2, 3),
            "frac_q_gt95": round(frac_q95[m].mean().item(), 3),
            "support_viol": round(viol_ev[m].mean().item(), 4),
            "rail": round(rail_ev[m].mean().item(), 3),
        }

    return {
        "val_nll_diag": nll_true.mean().item(),
        "shuffle_delta_nll": shuffle_delta,
        "dist_corr": dist_corr,
        "dist_cov50": cov50[j].item(),
        "dist_cov90": cov90[j].item(),
        "spurious_railing": round(spurious_railing, 4),
        "base_conc": round(base_conc, 3),
        "sci_cov90": round(sci_cov90, 3),
        "sci_cov90_highsnr": round(sci_cov90_highsnr, 3),
        "sbc_pass_frac": round(sbc_pass_frac, 3),
        "support_viol": round(support_viol, 4),
        "corner": corner,
        "cov50_all": {k: round(cov50[i].item(), 3) for i, k in enumerate(PARAM_NAMES)},
        "cov90_all": {k: round(cov90[i].item(), 3) for i, k in enumerate(PARAM_NAMES)},
    }


def log_corner(diag, log):
    """Print the corner benchmark table alongside the population metrics.

    The failure this exists to catch (analysis/divergence_localization) was
    invisible in every population average, so the table is printed every epoch
    rather than kept behind a flag.
    """
    c = diag.get("corner") or {}
    if not c:
        return
    log.info(f"  {'region':26}{'n':>5}{'cov90':>7}{'q_bias':>8}{'q_err':>7}"
             f"{'q>0.95':>8}{'viol':>8}{'rail':>7}")
    for lab, r in c.items():
        if r.get("n", 0) < 5:
            log.info(f"  {lab:26}{r.get('n', 0):>5}      (too few events)")
            continue
        log.info(f"  {lab:26}{r['n']:>5}{r['cov90_sci']:>7.3f}{r['q_bias']:>+8.3f}"
                 f"{r['q_abs_err']:>7.3f}{r['frac_q_gt95']:>8.3f}"
                 f"{r['support_viol']:>8.4f}{r['rail']:>7.3f}")


def main():
    """Parse args, build the model + (remix or in-memory) data loaders, and run
    the training loop with per-epoch diagnostics and calibration-gated
    checkpointing. See the module docstring for the config and gate semantics."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--outdir", default="model/lean_npe")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int, default=500)
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
    ap.add_argument("--det_dropout", type=float, default=0.0,
                    help="per-event probability of keeping only a random detector "
                         "subset (dropped -> unit white noise, matching the "
                         "inference-time fill for missing detectors)")
    ap.add_argument("--mc_oversample", type=float, default=0.0,
                    help="oversample events by (chirp mass)^alpha via a weighted "
                         "sampler — rebalances the log-uniform mass population "
                         "toward the heavy corner where twin tests show NPE "
                         "prior-fallback bias (0 = off; try 1.0)")
    ap.add_argument("--psd_cond", action="store_true",
                    help="condition the encoder on a per-detector log-ASD-vs-"
                         "design summary so whitened amplitude can be converted "
                         "to physical SNR->distance. Fixes the distance bias where "
                         "quieter (less-sensitive, e.g. O1/O2) real events are read "
                         "as farther. Requires --noise_bank (design-vs-real "
                         "sensitivity variation is what the branch learns from).")
    ap.add_argument("--psd_bands", type=int, default=16,
                    help="number of log-spaced ASD bands per detector for --psd_cond")
    ap.add_argument("--encoder_type", default="conv",
                    help="'conv' (LeanStrainEncoder) or 'coherent' (conv stem + "
                         "inter-detector geometry tokens: per-band coherence, GCC "
                         "arrival delays, amplitude ratios; optionally MF mass tokens). "
                         "NOTE: 'coherent' does NOT use an SVD basis -- the experimental "
                         "SVD-only encoder was removed in 6bb2ccd; assets/svd_basis.npz "
                         "is currently unused by any model.")
    # Flow size. These used to be LeanNPE-only defaults that never reached the
    # checkpoint, so a checkpoint's loadability depended on the working tree's
    # defaults at load time -- a strict=True size mismatch that has bitten this
    # repo before (see CLAUDE.md). Recorded in args now, so every checkpoint
    # carries the geometry it was trained with.
    ap.add_argument("--flow_layers", type=int, default=10,
                    help="NSF autoregressive RQS layers (default matches LeanNPE)")
    ap.add_argument("--flow_hidden", type=int, default=256,
                    help="NSF conditioner hidden width (default matches LeanNPE)")
    ap.add_argument("--flow_bins", type=int, default=16,
                    help="RQS bins per layer (default matches LeanNPE)")
    ap.add_argument("--dist_range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="luminosity-distance support in Mpc (default: legacy 40 2200). "
                         "Targets outside it are clamped, so it must cover the dataset's "
                         "distance_ranges (e.g. 10 5000 for configs/data_config_fixed.yaml). "
                         "Stored in the checkpoint state dict; load sites need no change.")
    ap.add_argument("--mass_param", default="m1m2", choices=["m1m2", "mc_q"],
                    help="flow's mass coordinates. 'm1m2' (legacy) models "
                         "(log m1, log m2), whose mass_1>=mass_2 convention makes the "
                         "support a triangle the flow must learn; measured cost is "
                         "27.7%% of GW150914's posterior landing at m1<m2 where the "
                         "training density is zero (analysis/.../mechanism/m22). "
                         "'mc_q' models (log Mc, q): support becomes an axis-aligned "
                         "box, m1>=m2 holds by construction, and Mc -- the "
                         "best-measured mass combination -- gets its own coordinate. "
                         "Changes ONLY the flow's target; weights/shapes/physical "
                         "interfaces are identical.")
    ap.add_argument("--geom_dropout", type=float, default=0.0,
                    help="coherent encoder only: per-sample probability of zeroing the "
                         "geometry tokens during training. The tokens were measured to "
                         "carry a degenerate mapping in the tau_HL~+7ms x loud x heavy "
                         "cell -- the GW150914/GW170814 (30,30) attractor "
                         "(analysis/attractor_basin, round 8 token grafting); dropout "
                         "keeps the conv path self-sufficient so the tokens cannot "
                         "own a cell outright. No effect at inference (train-only).")
    ap.add_argument("--basin_oversample", type=int, default=0,
                    help="extra epoch copies of every {tau_HL in [+4,+10] ms, m1>=29} "
                         "training event (each copy remixes independently). The basin "
                         "cell collapses on converged models because prior-weighted "
                         "NLL never pays to resolve it — feature-side fixes were "
                         "measured insufficient (analysis/attractor_basin addendum); "
                         "this makes the collapsed answer cost NLL. 3 adds ~12%% "
                         "epoch length on the current train set.")
    ap.add_argument("--delay_aug", type=float, default=0.0,
                    help="per-signal probability of remapping the event to a new sky: "
                         "per-detector arrival delays re-rolled EXACTLY and ra/dec/psi "
                         "relabeled honestly (antenna-matched candidate sky, closed-form "
                         "psi rotation; experiments/delay_aug.py). The ONLY augmentation "
                         "that varies inter-detector delay — it decorrelates measured "
                         "tau from mass ratio (the training data itself carries the "
                         "polarization-fold tau<->q correlation behind the (30,30) "
                         "attractor, analysis/attractor_basin addendum), so the encoder "
                         "must source q from the conv/chirp pathway instead of the "
                         "geometry-cell shortcut. Try 0.5.")
    ap.add_argument("--aux_mass", type=float, default=0.0,
                    help="weight of an auxiliary mass-regression loss on the encoder "
                         "context (SmoothL1 to the rank-0 normalized mass coords; "
                         "0=off, try 1.0). Directly penalizes the localized context "
                         "collapse behind the (30,30) attractor: the flow's NLL can "
                         "average a thin cell away, a per-event regression through "
                         "the context cannot. The head is a TRAINING SCAFFOLD "
                         "(separate module, not saved in the checkpoint); inference "
                         "is byte-identical.")
    ap.add_argument("--q_embed_dim", type=int, default=0,
                    help="width of a dedicated mass-ratio conditioning embedding, "
                         "concatenated onto the FLOW's own context input (unlike "
                         "--aux_mass, which only regularizes the encoder and never "
                         "reaches the flow). 0=off (default, exact prior behavior/"
                         "checkpoint shape). Motivated by flow_context_sensitivity.py "
                         "(2026-07-22): the flow's per-layer dense context->hidden map "
                         "was measured equally (in)sensitive to the q-informative "
                         "direction as to a random direction of the same size -- the "
                         "flow isn't discovering/routing through q on its own. This "
                         "hands it a small, explicitly-supervised (--aux_q_embed) "
                         "channel instead of asking it to find q amid 256 competing "
                         "dims. Try 8-16. Changes flow context_features, so a "
                         "checkpoint trained with q_embed_dim>0 will NOT load into a "
                         "model built with a different value.")
    ap.add_argument("--aux_q_embed", type=float, default=0.0,
                    help="weight of the auxiliary SmoothL1 regression loss that "
                         "supervises --q_embed_dim's embedding (q_pred_head -> "
                         "normalized rank-0 mass coords, same target as --aux_mass). "
                         "Requires --q_embed_dim>0. Unlike --aux_mass's head, "
                         "q_pred_head/q_embed_head ARE saved in the checkpoint since "
                         "the embedding is a live part of the flow's conditioning at "
                         "inference too. Try 1.0.")
    ap.add_argument("--offline_distill_cache", type=str, default=None,
                    help="path to a cache produced by "
                         "experiments/precompute_distill_targets.py (dict of "
                         "event_idx -> refined 'teacher' context, computed OFFLINE "
                         "against a FROZEN checkpoint snapshot). Replaces the earlier "
                         "live/online semi-amortized mechanism (removed 2026-07-23 "
                         "after two rounds of feedback-loop divergence in training -- "
                         "see experiments/semi_amortized.py's module docstring): here "
                         "nothing the targets depend on changes during this run, so "
                         "there is no feedback loop to diverge, and there is no "
                         "per-batch inner optimization loop to pay for. Default None: "
                         "baseline training is completely unchanged when absent. "
                         "Motivated by the same oracle-context evidence in "
                         "analysis/attractor_basin/REPORT.md -- good contexts exist "
                         "close to the encoder's own output distribution, this is a "
                         "training-signal problem, not a capacity one.")
    ap.add_argument("--offline_distill_flow_lambda", type=float, default=1.0,
                    help="weight of the flow-side loss computed against the FIXED "
                         "cached teacher context for events present in the cache "
                         "(added on top of, not instead of, that example's normal "
                         "NLL term).")
    ap.add_argument("--offline_distill_lambda", type=float, default=0.1,
                    help="weight of the distillation loss pulling the live encoder "
                         "context toward the (detached) fixed cached teacher context, "
                         "for events present in the cache.")
    ap.add_argument("--offline_distill_kind", default="mse", choices=["mse", "cosine", "huber"],
                    help="distillation loss form between the live encoder context "
                         "and the fixed cached teacher context.")
    ap.add_argument("--delay_aug_candidates", type=int, default=64,
                    help="sky grid candidates scored per delay-aug draw; the best "
                         "antenna-response match becomes the new sky (higher = more "
                         "honest amplitude labels, slightly less tau diversity).")
    ap.add_argument("--noise_consistency_w", type=float, default=0.0,
                    help="weight of the noise-consistency loss: each step, "
                         "--noise_consistency_groups signals (drawn from the "
                         "basin cell by default) each get --noise_consistency_k "
                         "independently-noised design-noise views (same masses/"
                         "geometry/distance, different noise draw), and the "
                         "encoder context is penalized for varying across a "
                         "group's views (MSE to the group mean). Implements "
                         "'more diverse noise per signal' + 'noise-invariant "
                         "context' directly targeting the MF-token branch's "
                         "noise-realization sensitivity found by grafting on "
                         "GW150914 (analysis/attractor_basin addendum, "
                         "2026-07-20). 0=off (default); try 0.3-1.0.")
    ap.add_argument("--noise_consistency_k", type=int, default=6,
                    help="noise views per signal for the consistency loss.")
    ap.add_argument("--noise_consistency_groups", type=int, default=12,
                    help="distinct signals per training step for the "
                         "consistency loss (12x6=72 extra encoder-only forward "
                         "passes per step by default).")
    ap.add_argument("--noise_consistency_pool", choices=["basin", "all"],
                    default="basin",
                    help="draw consistency-loss signals from the basin cell "
                         "(tau_HL in [+4,+10]ms, m1>=29 -- the diagnosed "
                         "population) or uniformly from all training events.")
    ap.add_argument("--noise_consistency_mode", choices=["design", "real"],
                    default="design",
                    help="'design': views are Gaussian design noise (no "
                         "sensitivity-context confound, but no exposure to real "
                         "noise's fine structure -- measured 2026-07-20 to NOT "
                         "transfer to real GW150914, rail 74.0%%->81.6%%). "
                         "'real': views are real noise from the SAME fixed "
                         "segment per detector (only crop position/sign vary), "
                         "keeping asd_bands identical across views while "
                         "targeting the ACTUAL mechanism the grafting test "
                         "isolated (real noise-realization sensitivity in the "
                         "MF branch). Requires --noise_bank. Try 'real'.")
    ap.add_argument("--mf_shape_norm", action="store_true",
                    help="requires --mf_tokens: add a 5th MF channel holding the "
                         "peak-overlap profile divided by its own per-detector max. "
                         "The existing |z| channel is ABSOLUTE (dominated by "
                         "distance/amplitude); mass-ratio information lives in the "
                         "profile's relative SHAPE. Measured (analysis/"
                         "posterior_bias_audit, E-K, 0.30%%-spaced bank, n=344): a "
                         "probe of the raw profile recovers q at R^2 0.005-0.048, "
                         "the peak-normalized profile at 0.178 (0.219 at SNR>15). "
                         "Changes the MF feature width, so a checkpoint trained with "
                         "it will NOT load into a model built without it.")
    ap.add_argument("--mf_tokens", action="store_true",
                    help="coherent encoder only: add matched-filter mass tokens — "
                         "per-detector complex overlaps against a fixed (Mc,q) "
                         "template grid (assets/mf_bank.npz), injected additively "
                         "into the geometry-token slots behind a zero-init gate "
                         "(init_from function preserved exactly). The overlap "
                         "pattern is the chirp-phase mass statistic the conv stem "
                         "cannot extract at low SNR: in-cell q AUC 0.989 vs 0.532 "
                         "for geometry features (analysis/attractor_basin/"
                         "mf_probe.py, 2026-07-19).")
    ap.add_argument("--mf_token_fusion", default="add", choices=["add", "concat"],
                    help="requires --mf_tokens. 'add' (default): legacy gated-"
                         "additive merge into the geometry-token slots. 'concat': "
                         "MF tokens are appended as independent tokens alongside "
                         "(not summed into) the geometry tokens, doubling the "
                         "extra-token count (4->8) fed to the fusion transformer. "
                         "Motivated by mf_embedding_tap_v2.py (2026-07-22): the "
                         "additive merge was measured as the single largest "
                         "information-loss stage in the whole MF pathway (in-cell "
                         "q AUC 0.883->0.738, bigger than the 1440->128 mf_mlp "
                         "embedding compression) — a near-uninformative geometry "
                         "signal (AUC 0.532 alone) summed into a clean informative "
                         "one appears to corrupt it. Changes encoder token count, "
                         "so a checkpoint trained with 'concat' will NOT load into "
                         "a model built with 'add' (or vice versa).")
    ap.add_argument("--geom_phase_residual", action="store_true",
                    help="coherent encoder only: append the residual polarization "
                         "phase after delay removal (cos,sin per pair) to the "
                         "geometry vector. The phase-slope delay alone fixed the "
                         "loud basin rows but left quiet heavy tau~+7ms events "
                         "collapsed; the residual is the quantity the slope "
                         "estimator cancels and exactly what disambiguates that "
                         "cell (analysis/attractor_basin addendum). Widens "
                         "geom_mlp input; --init_from checkpoints without it are "
                         "zero-padded so the pretrained function is preserved.")
    ap.add_argument("--era_weights", default=None,
                    help="relative real-noise sampling weights per observing run, e.g. "
                         "'O1:3,O2:3,O3b:1,O4:1'. The bank covers O1..O4 but is dominated "
                         "by the far more sensitive O3b/O4, so training asd_bands skew "
                         "'sensitive' while real O1/O2 events sit in the insensitive tail "
                         "(H1 train mean -0.47 vs GW150914 -0.88) — a measured driver of "
                         "their PSD-OOD context. Weight the eras to match deployment. "
                         "Default: uniform (legacy).")
    ap.add_argument("--det_keep_weights", default=None,
                    help="relative weights over the 6 detector keep-configs, order "
                         "[(H1),(L1),(V1),(H1,L1),(H1,V1),(L1,V1)] — e.g. '1,1,1,6,1,1' to "
                         "upweight the H1+L1-only config that every real 2-detector O1/O2 "
                         "event has (uniform makes it only ~1/6 of dropout events). "
                         "Requires --det_dropout>0. Default: uniform (legacy).")
    ap.add_argument("--ckpt_every", type=int, default=0,
                    help="also save epoch_XXXX.pth every N epochs (0=off). The trainer "
                         "otherwise keeps only best_model.pth, which makes a late "
                         "calibration regression unrecoverable.")
    ap.add_argument("--diag_events", type=int, default=1024,
                    help="validation events used for per-epoch diagnostics. Also "
                         "caps how many real-noise val events are materialized in "
                         "RAM, since that set feeds diagnostics ONLY. Drives the "
                         "corner benchmark's slice sizes: the heavy+q>0.8 region is "
                         "~5%% of events, so 256 (the old hardcoded value) left only "
                         "n~12-20 there -- too few to read q_bias off. 1024 gives "
                         "n~50. Raising it costs diagnostics time (~4x at 1024 vs "
                         "256, roughly +2 min on a 10 min epoch) and buys a "
                         "readable corner table.")
    ap.add_argument("--max_spurious_railing", type=float, default=0.10,
                    help="CALIBRATION GATE for best_model.pth: reject an epoch whose "
                         "Gaussian-val spurious railing exceeds this, even if its val "
                         "NLL is lower. MLE keeps lowering NLL by going overconfident "
                         "(boundary railing) and NLL is blind to it — in the long_run "
                         "study ep18 (good)=0.06 vs ep19 (collapsed)=0.9, so ~0.10 "
                         "cleanly separates them. Set very high (e.g. 1.0) to disable.")
    ap.add_argument("--min_highsnr_cov90", type=float, default=0.0,
                    help="optional extra gate: require high-SNR (>20) science-param "
                         "90%% coverage >= this (0=disabled; try 0.7).")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    if args.encoder_type == "coherent":
        # the coherent encoder always ingests asd_bands; activate the plumbing.
        args.psd_cond = True
    if args.psd_cond and not args.noise_bank:
        log.warning("--psd_cond without --noise_bank: all training events are "
                    "design-whitened (asd_bands=0), so the noise branch sees no "
                    "sensitivity variation and cannot learn. Add --noise_bank.")

    # ---- mixture-rebalance knobs (analysis/divergence_localization): match the
    # training sensitivity / detector-configuration distribution to deployment ----
    era_weights = None
    if args.era_weights:
        try:
            era_weights = {k.strip(): float(v)
                           for k, v in (tok.split(":") for tok in args.era_weights.split(","))}
        except ValueError:
            ap.error("--era_weights must look like 'O1:3,O2:3,O3b:1,O4:1'")
        if not args.noise_bank:
            ap.error("--era_weights requires --noise_bank (it weights real-noise segments)")
    det_keep_weights = None
    if args.det_keep_weights:
        try:
            det_keep_weights = [float(x) for x in args.det_keep_weights.split(",")]
        except ValueError:
            ap.error("--det_keep_weights must be comma-separated numbers")
        if len(det_keep_weights) != 6:
            ap.error(f"--det_keep_weights needs 6 values (order [(H1),(L1),(V1),"
                     f"(H1,L1),(H1,V1),(L1,V1)]), got {len(det_keep_weights)}")
        if args.det_dropout <= 0:
            ap.error("--det_keep_weights requires --det_dropout > 0 (it reweights "
                     "which subset dropout keeps)")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    if outdir.exists() and any(outdir.iterdir()):
        log.warning(f"outdir {outdir} exists and is not empty; will overwrite files")
    else:
        outdir.mkdir(parents=True, exist_ok=True)

    log.info("Loading data...")
    # Record the generator's distance prior in the checkpoint args, so
    # PosteriorResult.reweight_distance_prior() can convert from the prior the
    # model actually learned. None = unknown (dataset predates provenance).
    args.dataset_distance_prior, args.dataset_distance_ranges = None, None
    args.dataset_event_types = None
    dcfg_path = Path(args.data) / "dataset_config.yaml"
    if dcfg_path.exists():
        import yaml
        dcfg = yaml.safe_load(open(dcfg_path)) or {}
        args.dataset_distance_prior = str(dcfg.get("distance_prior") or "comoving_d2").lower()
        args.dataset_distance_ranges = dcfg.get("distance_ranges")
        # source classes present (inference warns when an event falls outside)
        mix = dcfg.get("event_type_distribution") or {}
        args.dataset_event_types = sorted(t.upper() for t, p in mix.items()
                                          if t != "noise" and float(p) > 0) or None
        log.info(f"dataset distance prior: {args.dataset_distance_prior} "
                 f"ranges={args.dataset_distance_ranges}")
    else:
        log.warning(f"no {dcfg_path} — training distance prior unrecorded; "
                    "distance-prior reweighting will need it passed explicitly")
    if args.remix:
        from remix_data import RemixDataset, build_memmap_cache
        cache = Path(args.data) / "memmap" / "train"
        if not (cache / "events.json").exists():
            log.info("Building memmap cache (one-time)...")
            build_memmap_cache(args.data, "train", str(cache))
        train_ds = RemixDataset(str(cache), remix=True,
                                real_noise_dir=args.noise_bank,
                                real_noise_prob=args.real_noise_prob,
                                det_dropout=args.det_dropout,
                                return_asd_bands=args.psd_cond,
                                psd_bands=args.psd_bands,
                                real_noise_era_weights=era_weights,
                                det_keep_weights=det_keep_weights,
                                basin_oversample=args.basin_oversample,
                                delay_aug=args.delay_aug,
                                delay_aug_candidates=args.delay_aug_candidates,
                                return_event_idx=bool(args.offline_distill_cache),
                                dist_range=args.dist_range)
        log.info(f"remix train events={len(train_ds)} (noise pool={train_ds.n_noise}, "
                 f"p_real={args.real_noise_prob}, det_dropout={args.det_dropout}, "
                 f"era_weights={era_weights}, det_keep_weights={det_keep_weights}, "
                 f"delay_aug={args.delay_aug})")
    else:
        tr_s, tr_p, tr_n, tr_snr = load_split(args.data, "train", args.limit_train)
        train_ds = EventDataset(tr_s, tr_p, tr_n, tr_snr)
        log.info(f"train events={len(train_ds)}")
    audit_scaler_support(train_ds, args, log)

    # validation is always FIXED (stored strain, no remixing) for comparability
    va_s, va_p, va_n, va_snr = load_split(args.data, "validation")
    log.info(f"val events={va_s.shape[0]}")

    # Fixed REAL-NOISE validation variant: same validation events, noise
    # replaced by deterministic (seeded, epoch-independent) real crops with
    # signals re-colored to each crop's whitening. Materialized once so the
    # per-epoch real-noise metrics are exactly comparable across epochs.
    va_real = None
    va_real_asd = None
    if args.noise_bank:
        from remix_data import RemixDataset as _RD, build_memmap_cache as _bc
        vcache = Path(args.data) / "memmap" / "validation"
        if not (vcache / "events.json").exists():
            _bc(args.data, "validation", str(vcache))
        vds = _RD(str(vcache), remix=False, seed=1234,
                  real_noise_dir=args.noise_bank, real_noise_prob=1.0,
                  return_asd_bands=args.psd_cond, psd_bands=args.psd_bands)
        # Materialize only what the diagnostics actually read. This set feeds
        # run_diagnostics ONLY (the full val NLL comes from the Gaussian
        # val_loader), and run_diagnostics looks at its first n_events, so
        # holding all 4702 in RAM was pure waste: ~460 MB of fp16 strain to use
        # ~50 MB of it. Capping is behaviour-identical, not an approximation.
        n_keep = min(args.diag_events, len(vds))
        vs, vp, vn, vab = [], [], [], []
        for i in range(n_keep):
            item = vds[i]
            s, p, n = item[:3]  # (strain, params, nsig, snr[, asd_bands])
            vs.append(s.to(torch.float16)); vp.append(p); vn.append(n)
            if args.psd_cond:
                vab.append(item[4])
        va_real = (torch.stack(vs), torch.stack(vp), torch.stack(vn))
        va_real_asd = torch.stack(vab) if args.psd_cond else None
        log.info(f"real-noise validation materialized: {n_keep} of {len(vds)} events "
                 f"(diagnostics read {args.diag_events})")

    model = LeanNPE(premerger=args.premerger,
                    psd_cond=args.psd_cond, psd_bands=args.psd_bands,
                    encoder_type=args.encoder_type,
                    mass_param=args.mass_param,
                    geom_dropout=args.geom_dropout,
                    geom_phase_residual=args.geom_phase_residual,
                    mf_tokens=args.mf_tokens,
                    mf_shape_norm=args.mf_shape_norm,
                    mf_token_fusion=args.mf_token_fusion,
                    q_embed_dim=args.q_embed_dim,
                    dist_range=args.dist_range,
                    flow_layers=args.flow_layers,
                    flow_hidden=args.flow_hidden,
                    flow_bins=args.flow_bins).to(device)
    if args.init_from:
        init = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd = init["model_state_dict"]
        # Feature-widening compatibility: the phase-residual features are
        # appended at the END of the geometry vector, so a pre-residual
        # geom_mlp input weight can be zero-padded — the pretrained function
        # is preserved exactly and the new features start with zero influence.
        wk = "encoder.geom_mlp.0.weight"
        if wk in sd:
            cur = model.state_dict()[wk]
            if sd[wk].shape[0] == cur.shape[0] and sd[wk].shape[1] < cur.shape[1]:
                pad = torch.zeros(cur.shape[0], cur.shape[1] - sd[wk].shape[1],
                                  dtype=sd[wk].dtype)
                log.info(f"init_from: zero-padding {wk} "
                         f"{tuple(sd[wk].shape)} -> {tuple(cur.shape)}")
                sd[wk] = torch.cat([sd[wk], pad], dim=1)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys loading {args.init_from}: {unexpected}"
        if missing:
            log.info(f"freshly initialized (absent from checkpoint): {missing}")
        # load_state_dict adopts the INIT checkpoint's distance support; an
        # explicit --dist_range must win (the distance coordinate is then
        # re-learned during fine-tuning, since its normalized map changed).
        if args.dist_range is not None:
            old = tuple(model.dist_range.tolist())
            model.dist_range.copy_(torch.tensor(args.dist_range, dtype=torch.float32))
            model.scaler = ParamScaler(PARAM_NAMES, premerger=args.premerger,
                                       mass_param=args.mass_param,
                                       dist_range=tuple(args.dist_range)).to(device)
            if old != tuple(args.dist_range):
                log.warning(f"init_from: distance support {old} -> {tuple(args.dist_range)} Mpc; "
                            "the flow's distance coordinate must re-learn its map")
        log.info(f"fine-tuning from {args.init_from} "
                 f"(epoch {init.get('epoch')}, val_nll {init.get('val_nll'):.4f})")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"LeanNPE parameters: {n_params:,}")

    # Training-scaffold aux head (see batch_nll docstring); never saved, so
    # every checkpoint consumer (validate_checkpoint, infer, pipeline) is
    # untouched.
    aux_head = None
    if args.aux_mass > 0.0:
        aux_head = torch.nn.Sequential(
            torch.nn.Linear(model.context_dim, 128), torch.nn.GELU(),
            torch.nn.Linear(128, 2)).to(device)
        log.info(f"aux mass-regression head active (weight {args.aux_mass})")
    if args.q_embed_dim > 0:
        assert args.aux_q_embed > 0.0, "--q_embed_dim>0 with --aux_q_embed=0 trains an unsupervised embedding"
        log.info(f"dedicated q-conditioning embedding active: dim {args.q_embed_dim}, "
                 f"aux weight {args.aux_q_embed} (flow context_features "
                 f"{model.context_dim + args.q_embed_dim + model.rank_embed.embedding_dim})")

    # Frozen basin regression probe (analysis/attractor_basin): the GWTC
    # (30,30) attractor cell {tau_HL~+7ms, loud, heavy} is invisible to every
    # population metric AND to the sky-aggregated corner table, so it gets its
    # own fixed injections, printed every epoch like the corner.
    basin_bank = None
    try:
        from ahsd.evaluation.basin_probe import (build_bank, evaluate_bank,
                                                 log_basin, basin_gate)
        basin_bank = build_bank()
        log.info(f"basin probe bank ready: {len(basin_bank['labels'])} injections")
    except Exception as e:
        log.warning(f"basin probe unavailable ({type(e).__name__}: {e})")

    # Noise-consistency loss pool (see multiview_consistency_loss docstring).
    cons_pool = None
    cons_rng = None
    if args.noise_consistency_w > 0.0:
        assert args.remix, "--noise_consistency_w requires --remix"
        if args.noise_consistency_mode == "real":
            assert args.noise_bank, "--noise_consistency_mode real requires --noise_bank"
        cons_pool = (train_ds.basin_cell_event_indices() if args.noise_consistency_pool == "basin"
                    else np.arange(len(train_ds.events)))
        cons_rng = np.random.default_rng(20260720)
        log.info(f"noise-consistency loss: weight {args.noise_consistency_w}, "
                 f"{args.noise_consistency_groups} groups x {args.noise_consistency_k} views/step, "
                 f"pool={args.noise_consistency_pool} ({len(cons_pool)} events), "
                 f"mode={args.noise_consistency_mode}")

    # Offline teacher distillation (replaces the earlier live/online
    # semi-amortized mechanism, removed 2026-07-23 -- see
    # experiments/semi_amortized.py's module docstring and
    # experiments/precompute_distill_targets.py).
    distill_targets = None
    distill_loss_fn = None
    if args.offline_distill_cache:
        from semi_amortized import DistillationLoss
        payload = torch.load(args.offline_distill_cache, map_location="cpu", weights_only=False)
        raw_targets = payload["targets"]
        distill_targets = torch.full((len(train_ds.events), model.context_dim), float("nan"))
        for idx, ctx in raw_targets.items():
            distill_targets[idx] = ctx
        distill_targets = distill_targets.to(device)
        distill_loss_fn = DistillationLoss(kind=args.offline_distill_kind)
        log.info(f"offline distillation ENABLED: cache={args.offline_distill_cache} "
                 f"({payload['n_events']} cached targets from "
                 f"{payload['source_checkpoint']}), "
                 f"flow_lambda={args.offline_distill_flow_lambda}, "
                 f"distill_lambda={args.offline_distill_lambda} "
                 f"({args.offline_distill_kind})")

    opt = torch.optim.AdamW(
        list(model.parameters()) + (list(aux_head.parameters()) if aux_head else []),
        lr=args.lr, weight_decay=1e-5)
    steps_per_epoch = math.ceil(len(train_ds) / args.batch)
    total_steps = steps_per_epoch * args.epochs

    def lr_lambda(step):
        """LR multiplier: linear warmup for `warmup_steps`, then cosine decay to
        1% of the peak over the remaining steps."""
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # persistent_workers must stay False: workers snapshot the dataset, and
    # set_epoch() (fresh noise pairings each epoch) must reach them — workers
    # are respawned per epoch and pick up the bumped epoch counter.
    train_sampler = None
    if args.mc_oversample > 0:
        assert args.remix, "--mc_oversample requires --remix"
        starts = [e[0] for e in train_ds.events]
        m1 = np.asarray(train_ds.params[starts, 0], dtype=np.float64)
        m2 = np.asarray(train_ds.params[starts, 1], dtype=np.float64)
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        w = torch.from_numpy(mc ** args.mc_oversample)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            w, num_samples=len(train_ds), replacement=True)
        log.info(f"mc_oversample alpha={args.mc_oversample}: Mc-45 events drawn "
                 f"{(45/10)**args.mc_oversample:.1f}x more often than Mc-10")
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=(args.workers if args.remix else 0))
    val_loader = DataLoader(EventDataset(va_s, va_p, va_n, va_snr), batch_size=args.batch,
                            shuffle=False, num_workers=0)

    use_wandb = False
    if args.wandb:
        try:
            import wandb
            wandb.init(project="posteriflow-lean-npe", config=vars(args))
            use_wandb = True
        except Exception as e:
            log.warning(f"wandb disabled: {e}")

    best_select_cal = float("inf")   # best selection score among CALIBRATED epochs
    best_select_any = float("inf")   # best selection score among all epochs (fallback)
    best_tier_cal = -1               # 1 = passed the basin gate, 0 = calibrated only.
    # A basin-passing epoch outranks ANY non-passing one regardless of NLL:
    # the pslope run visited basin-clean states (snr24 q95 0.33) that pure
    # NLL selection then discarded for later basin-collapsed epochs.
    have_calibrated = False
    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        model.train()
        t0 = time.time()
        tr_losses, gnorms = [], []
        distill_diag_accum = {"n_cached": [], "flow_loss": [], "distill_loss": []}
        try:
            from tqdm import tqdm
            train_iter = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        except ImportError:
            train_iter = train_loader
        for batch in train_iter:
            strain, params, nsig = batch[:3]
            asd = batch[4].to(device).float() if args.psd_cond else None
            strain = strain.to(device).float()
            params = params.to(device)
            nsig = nsig.to(device)
            loss, context = batch_nll(model, strain, params, nsig, asd_bands=asd,
                                      aux_head=aux_head, aux_w=args.aux_mass,
                                      aux_q_embed_w=args.aux_q_embed, return_context=True)
            cons_loss_val = None
            if cons_pool is not None:
                group_events = cons_rng.choice(
                    cons_pool, size=args.noise_consistency_groups,
                    replace=len(cons_pool) < args.noise_consistency_groups)
                views_batch = [train_ds.get_multiview(int(ev), args.noise_consistency_k,
                                                      epoch, gi,
                                                      noise_mode=args.noise_consistency_mode)
                              for gi, ev in enumerate(group_events)]
                cons_loss = multiview_consistency_loss(model, views_batch, device,
                                                        args.psd_cond)
                loss = loss + args.noise_consistency_w * cons_loss
                cons_loss_val = float(cons_loss.item())
            if distill_targets is not None:
                event_idx_batch = batch[-1].to(device)
                cached = distill_targets[event_idx_batch]
                has_cache = ~torch.isnan(cached[:, 0])
                n_cached = int(has_cache.sum().item())
                if n_cached > 0:
                    idx = has_cache.nonzero(as_tuple=True)[0]
                    cached_ctx = cached[idx]
                    rank0 = torch.zeros(idx.numel(), dtype=torch.long, device=device)
                    flow_loss = model.nll(None, params[idx, 0, :], rank0,
                                          context=cached_ctx).mean()
                    distill_loss = distill_loss_fn(context[idx], cached_ctx.detach())
                    loss = loss + (args.offline_distill_flow_lambda * flow_loss
                                   + args.offline_distill_lambda * distill_loss)
                    distill_diag_accum["n_cached"].append(n_cached)
                    distill_diag_accum["flow_loss"].append(float(flow_loss.item()))
                    distill_diag_accum["distill_loss"].append(float(distill_loss.item()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            global_step += 1
            tr_losses.append(loss.item())
            gnorms.append(float(gn))
            if hasattr(train_iter, "set_postfix"):
                pf = dict(nll=f"{tr_losses[-1]:.3f}", gnorm=f"{gnorms[-1]:.1f}")
                if cons_loss_val is not None:
                    pf["cons"] = f"{cons_loss_val:.4f}"
                train_iter.set_postfix(**pf)

        model.eval()
        va_losses = []
        with torch.no_grad():
            for strain, params, nsig, _snr in val_loader:
                strain = strain.to(device).float()
                # Gaussian val is design-whitened -> sensitivity summary is 0.
                asd = (torch.zeros(strain.shape[0], 3, args.psd_bands, device=device)
                       if args.psd_cond else None)
                va_losses.append(batch_nll(model, strain, params.to(device),
                                           nsig.to(device), asd_bands=asd).item())
        val_nll = float(np.mean(va_losses))

        va_asd_diag = (torch.zeros(min(args.diag_events, va_s.shape[0]), 3, args.psd_bands)
                       if args.psd_cond else None)
        diag = run_diagnostics(model, va_s, va_p, va_n, device, asd_bands=va_asd_diag,
                               snr=va_snr, n_events=args.diag_events)
        diag_real = {}
        if va_real is not None:
            dr = run_diagnostics(model, *va_real, device, asd_bands=va_real_asd,
                                 n_events=args.diag_events)
            diag_real = {f"real_{k}": v for k, v in dr.items() if not isinstance(v, dict)}
        dt = time.time() - t0
        rec = {
            "epoch": epoch, "train_nll": float(np.mean(tr_losses)), "val_nll": val_nll,
            "grad_norm": float(np.mean(gnorms)), "lr": sched.get_last_lr()[0],
            "epoch_seconds": round(dt, 1), **{k: v for k, v in diag.items() if not isinstance(v, dict)},
            **diag_real,
        }
        if distill_targets is not None and distill_diag_accum["n_cached"]:
            # per-batch; average over the epoch, persisted into history.json
            # like every other diagnostic.
            rec["distill_n_cached"] = float(np.mean(distill_diag_accum["n_cached"]))
            rec["distill_flow_loss"] = float(np.mean(distill_diag_accum["flow_loss"]))
            rec["distill_loss"] = float(np.mean(distill_diag_accum["distill_loss"]))
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
            f"rail {rec['spurious_railing']:.3f} zc {rec['base_conc']:.2f} "
            f"covHi {rec['sci_cov90_highsnr']:.2f} sbc {rec['sbc_pass_frac']:.2f} "
            f"viol {rec.get('support_viol', float('nan')):.3f} | "
            f"gnorm {rec['grad_norm']:.1f} | {dt:.0f}s{real_str}"
        )
        if distill_targets is not None and "distill_n_cached" in rec:
            log.info(
                f"  offline distill | n_cached {rec['distill_n_cached']:.1f}/batch | "
                f"flow_loss {rec['distill_flow_loss']:.3f} | "
                f"distill_loss {rec['distill_loss']:.4f}"
            )
        log_corner(diag, log)
        basin_ok = False
        if basin_bank is not None:
            try:
                basin_rows = evaluate_bank(model, basin_bank, device,
                                           psd_bands=args.psd_bands)
                log_basin(basin_rows, log)
                basin_ok = basin_gate(basin_rows)[0]
                diag["basin"] = basin_rows   # persisted into the checkpoint
            except Exception as e:
                log.warning(f"basin probe failed ({type(e).__name__}: {e})")
        if use_wandb:
            import wandb
            wandb.log(rec)

        # selection metric: Gaussian val NLL alone (legacy), or the mean of
        # Gaussian and real-noise val NLL when a noise bank is configured —
        # the deployment goal is robustness on real data WITHOUT losing
        # simulated-data performance, so both count equally.
        select = val_nll if va_real is None else 0.5 * (val_nll + rec["real_val_nll_diag"])

        def _save(path):
            """Write the current checkpoint (weights, epoch, val_nll,
            diagnostics, args) to `path`."""
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "val_nll": val_nll, "diagnostics": diag, "args": vars(args),
            }, path)

        # CALIBRATION GATE. A lower val NLL does NOT mean a better model: late
        # training goes overconfident (boundary railing + high-SNR under-
        # coverage) while NLL keeps falling, and NLL is blind to it (see
        # analysis/divergence_localization). Only a CALIBRATED epoch may claim
        # best_model.pth; until the first calibrated epoch appears we fall back
        # to best-NLL so a run never ends with no checkpoint.
        rail = diag.get("spurious_railing", 0.0)
        cov_hi = diag.get("sci_cov90_highsnr", 1.0)
        calibrated = (rail <= args.max_spurious_railing and
                      (args.min_highsnr_cov90 <= 0.0 or cov_hi >= args.min_highsnr_cov90))
        if calibrated:
            tier = 1 if basin_ok else 0
            if tier > best_tier_cal or (tier == best_tier_cal and select < best_select_cal):
                best_tier_cal = tier
                best_select_cal = select
                have_calibrated = True
                _save(outdir / "best_model.pth")
                log.info(f"  ✓ calibrated (rail {rail:.3f}, basin "
                         f"{'PASS' if basin_ok else 'fail'}) -> best_model.pth "
                         f"(select {select:.4f}, tier {tier})")
        else:
            log.info(f"  ⚠ epoch {epoch} REJECTED for best_model "
                     f"(spurious railing {rail:.3f} > {args.max_spurious_railing})")
            if not have_calibrated and select < best_select_any:
                best_select_any = select
                _save(outdir / "best_model.pth")   # provisional (uncalibrated) fallback
        _save(outdir / "last_model.pth")
        if args.ckpt_every and epoch % args.ckpt_every == 0:
            _save(outdir / f"epoch_{epoch:04d}.pth")
        with open(outdir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    if have_calibrated:
        log.info(f"done. best_model.pth = best CALIBRATED epoch "
                 f"(select {best_select_cal:.4f}) -> {outdir/'best_model.pth'}")
    else:
        log.info(f"done. NO epoch passed the calibration gate "
                 f"(spurious railing <= {args.max_spurious_railing}); best_model.pth is "
                 f"the best-NLL FALLBACK (select {best_select_any:.4f}). Inspect "
                 f"history.json / loosen --max_spurious_railing.")


if __name__ == "__main__":
    main()
