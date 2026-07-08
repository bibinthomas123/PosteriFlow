#!/usr/bin/env python3
"""
Controlled experiment: encoder representation vs posterior approximation.

The encoder is FROZEN (v3 final, eval mode, no grads). Contexts are extracted
once for 4 remix epochs of the training set (noise diversity preserved) and
one fixed pass of validation. Three posterior heads then train on EXACTLY the
same (context, rank, theta) rows — the only variable is the estimator:

  A  current NSF   (8 layers, hidden 192, 16 bins)   — protocol control
  B  larger NSF    (12 layers, hidden 384, 24 bins)  — flow capacity
  C  MDN           (16 diagonal Gaussians, MLP head) — different family

Decision logic: same real-event bias across A/B/C => encoder bottleneck;
B or C markedly better => posterior-approximation bottleneck.

    python experiments/frozen_context_heads.py --extract
    python experiments/frozen_context_heads.py --train A   (then B, C)
    python experiments/frozen_context_heads.py --eval
"""
import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _ROOT)
sys.path.insert(0, f"{_ROOT}/src")

import numpy as np
import torch
import torch.nn as nn

from experiments.remix_data import RemixDataset
from ahsd.models.lean_npe import LeanNPE, ParamScaler, PARAM_NAMES

ENCODER_CKPT = "model/lean_npe_v3/best_model.pth"
CACHE = Path("data/frozen_ctx")
OUTDIR = Path("model/frozen_heads")
N_REMIX_EPOCHS = 4
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

REAL_EVENTS = [("GW150914", {"m1": 38.8, "m2": 33.4, "dL": 440}),
               ("GW151226", {"m1": 13.7, "m2": 7.7, "dL": 440}),
               ("GW170104", {"m1": 31.0, "m2": 20.1, "dL": 960}),
               ("GW170608", {"m1": 12.0, "m2": 7.0, "dL": 320}),
               ("GW170729", {"m1": 50.6, "m2": 34.3, "dL": 2750}),
               ("GW170814", {"m1": 30.7, "m2": 25.3, "dL": 600})]


def load_frozen():
    ckpt = torch.load(ENCODER_CKPT, map_location="cpu", weights_only=False)
    model = LeanNPE(premerger=ckpt["args"].get("premerger", False))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ── stage 1: context extraction ──────────────────────────────────────────────

def extract():
    CACHE.mkdir(parents=True, exist_ok=True)
    model = load_frozen()

    def encode_pass(ds, tag, epochs):
        rows_ctx, rows_rank, rows_theta = [], [], []
        for ep in epochs:
            ds.set_epoch(ep)
            for i0 in range(0, len(ds), 64):
                items = [ds[i] for i in range(i0, min(i0 + 64, len(ds)))]
                strain = torch.stack([x[0] for x in items]).to(DEV).float()
                with torch.no_grad():
                    ctx = model.encoder(strain).cpu().numpy()
                for k, (s, pv, nsig) in enumerate(items):
                    for r in range(int(nsig)):
                        rows_ctx.append(ctx[k])
                        rows_rank.append(r)
                        rows_theta.append(pv[r].numpy())
            print(f"[{tag}] epoch {ep} done ({len(rows_ctx)} rows)", flush=True)
        np.save(CACHE / f"{tag}_ctx.npy", np.stack(rows_ctx).astype(np.float32))
        np.save(CACHE / f"{tag}_rank.npy", np.array(rows_rank, dtype=np.int64))
        np.save(CACHE / f"{tag}_theta.npy", np.stack(rows_theta).astype(np.float32))

    tr = RemixDataset("data/dataset/memmap/train", remix=True, seed=0,
                      real_noise_dir="data/noise_bank", real_noise_prob=0.5,
                      det_dropout=0.2)
    encode_pass(tr, "train", range(N_REMIX_EPOCHS))
    va = RemixDataset("data/dataset/memmap/validation", remix=False, seed=1234)
    encode_pass(va, "val", [0])
    print("extraction complete ->", CACHE)


# ── heads ────────────────────────────────────────────────────────────────────

class MDNHead(nn.Module):
    """Conditional Gaussian mixture in the scaler's normalized space."""

    def __init__(self, ctx_dim=288, n_comp=16, n_par=11, hidden=512):
        super().__init__()
        self.n_comp, self.n_par = n_comp, n_par
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_comp * (1 + 2 * n_par)))

    def _split(self, ctx):
        o = self.net(ctx).view(-1, self.n_comp, 1 + 2 * self.n_par)
        logit = o[..., 0]
        mu = o[..., 1:1 + self.n_par]
        log_sig = o[..., 1 + self.n_par:].clamp(-7.0, 1.0)
        return logit, mu, log_sig

    def nll(self, y, ctx):
        logit, mu, log_sig = self._split(ctx)
        lw = torch.log_softmax(logit, dim=1)
        z = (y.unsqueeze(1) - mu) / torch.exp(log_sig)
        comp = -0.5 * (z ** 2).sum(-1) - log_sig.sum(-1) \
               - 0.5 * self.n_par * np.log(2 * np.pi)
        return -torch.logsumexp(lw + comp, dim=1)

    @torch.no_grad()
    def sample(self, ctx, n):
        logit, mu, log_sig = self._split(ctx)
        B = ctx.shape[0]
        w = torch.softmax(logit, dim=1)
        comp = torch.multinomial(w, n, replacement=True)          # [B, n]
        idx = comp.unsqueeze(-1).expand(B, n, self.n_par)
        m = torch.gather(mu, 1, idx)
        s = torch.gather(torch.exp(log_sig), 1, idx)
        return m + s * torch.randn_like(m)


class FlowHead(nn.Module):
    def __init__(self, layers, hidden, bins):
        super().__init__()
        from ahsd.models.flows import NSFPosteriorFlow
        self.flow = NSFPosteriorFlow(
            features=len(PARAM_NAMES), context_features=288,
            hidden_features=hidden, num_layers=layers, num_bins=bins,
            tail_bound=3.0, dropout=0.0, temperature_scale=1.0,
            use_masked_context=False)
        self.flow.temperature.requires_grad_(False)

    def nll(self, y, ctx):
        return self.flow.compute_psd_aware_nll(y, ctx, torch.zeros_like(y))

    @torch.no_grad()
    def sample(self, ctx, n):
        B, C = ctx.shape
        cr = ctx.unsqueeze(1).expand(B, n, C).reshape(B * n, C)
        z = torch.randn(B * n, len(PARAM_NAMES), device=ctx.device)
        y, _ = self.flow.inverse(z, cr)
        return y.reshape(B, n, -1)


def make_head(which):
    if which == "A":
        return FlowHead(8, 192, 16)
    if which == "B":
        return FlowHead(12, 384, 24)
    if which == "C":
        return MDNHead()
    raise ValueError(which)


class Rows(torch.utils.data.Dataset):
    def __init__(self, tag):
        self.ctx = np.load(CACHE / f"{tag}_ctx.npy", mmap_mode="r")
        self.rank = np.load(CACHE / f"{tag}_rank.npy")
        self.theta = np.load(CACHE / f"{tag}_theta.npy")

    def __len__(self):
        return len(self.rank)

    def __getitem__(self, i):
        return (torch.from_numpy(self.ctx[i].copy()),
                int(self.rank[i]), torch.from_numpy(self.theta[i]))


def train(which, epochs=25, batch=512, lr=3e-4):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    scaler = ParamScaler().to(DEV)
    head = make_head(which).to(DEV)
    # each head owns its rank embedding (part of the posterior estimator)
    rank_embed = nn.Embedding(5, 32).to(DEV)
    params = list(head.parameters()) + list(rank_embed.parameters())
    n_par = sum(p.numel() for p in params if p.requires_grad)
    print(f"head {which}: {n_par:,} trainable params", flush=True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    tr = torch.utils.data.DataLoader(Rows("train"), batch_size=batch, shuffle=True)
    va = torch.utils.data.DataLoader(Rows("val"), batch_size=1024, shuffle=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tr))
    best = float("inf")
    hist = []
    for ep in range(1, epochs + 1):
        head.train()
        tl = []
        for ctx, rank, theta in tr:
            ctx, theta = ctx.to(DEV), theta.to(DEV)
            fc = torch.cat([ctx, rank_embed(rank.to(DEV))], dim=1)
            y = scaler.normalize(theta)
            loss = head.nll(y, fc).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            tl.append(loss.item())
        head.eval()
        vl = []
        with torch.no_grad():
            for ctx, rank, theta in va:
                ctx, theta = ctx.to(DEV), theta.to(DEV)
                fc = torch.cat([ctx, rank_embed(rank.to(DEV))], dim=1)
                vl.append(head.nll(scaler.normalize(theta), fc).mean().item())
        v = float(np.mean(vl))
        hist.append({"epoch": ep, "train": float(np.mean(tl)), "val": v})
        print(f"[{which}] epoch {ep:2d} | train {np.mean(tl):.3f} | val {v:.3f}",
              flush=True)
        if v < best:
            best = v
            torch.save({"head": head.state_dict(),
                        "rank_embed": rank_embed.state_dict(),
                        "which": which, "val_nll": v, "epoch": ep},
                       OUTDIR / f"head_{which}.pth")
    json.dump(hist, open(OUTDIR / f"head_{which}_history.json", "w"))
    print(f"[{which}] best val {best:.3f}", flush=True)


# ── stage 3: shared evaluation ───────────────────────────────────────────────

def load_head(which):
    ck = torch.load(OUTDIR / f"head_{which}.pth", map_location="cpu",
                    weights_only=False)
    head = make_head(which)
    head.load_state_dict(ck["head"])
    rank_embed = nn.Embedding(5, 32)
    rank_embed.load_state_dict(ck["rank_embed"])
    return head.to(DEV).eval(), rank_embed.to(DEV), ck


def evaluate():
    from scipy import stats as sps
    from ahsd.inference.preprocessing import fetch_gwosc, prepare_real
    model = load_frozen()
    scaler = ParamScaler().to(DEV)
    rows = Rows("val")
    n_ev = 1500
    ctx = torch.from_numpy(rows.ctx[:n_ev].copy()).to(DEV)
    rank = torch.from_numpy(rows.rank[:n_ev]).to(DEV)
    theta = torch.from_numpy(rows.theta[:n_ev]).to(DEV)
    report = {}
    for which in ("A", "B", "C"):
        if not (OUTDIR / f"head_{which}.pth").exists():
            continue
        head, rank_embed, ck = load_head(which)
        fc = torch.cat([ctx, rank_embed(rank)], dim=1)
        with torch.no_grad():
            nll = head.nll(scaler.normalize(theta), fc).mean().item()
            y = head.sample(fc, 300)
            y = model.scaler.wrap(y.reshape(-1, 11)).reshape(n_ev, 300, 11)
            samp = scaler.denormalize(y.reshape(-1, 11)).reshape(n_ev, 300, 11)
            samp = samp.cpu().numpy()
        tru = theta.cpu().numpy()
        lo = np.quantile(samp, 0.05, axis=1)
        hi = np.quantile(samp, 0.95, axis=1)
        cov90 = float(((tru >= lo) & (tru <= hi)).mean())
        ranks = (samp < tru[:, None, :]).mean(axis=1)
        ks_bad = sum(sps.kstest(ranks[:, j], "uniform").pvalue < 1e-3
                     for j in range(11))
        ev = {}
        for name, pub in REAL_EVENTS:
            try:
                gps, raw, starts, srs, found = fetch_gwosc(name)
                prep = prepare_real(raw, gps, starts, srs, seed=0)
                st = torch.from_numpy(prep.strain).unsqueeze(0).to(DEV).float()
                with torch.no_grad():
                    c = model.encoder(st)
                    f1 = torch.cat([c, rank_embed(torch.zeros(1, dtype=torch.long,
                                                              device=DEV))], dim=1)
                    ys = head.sample(f1, 4000)
                    ys = model.scaler.wrap(ys.reshape(-1, 11))
                    s = scaler.denormalize(ys).cpu().numpy()
                m1 = np.maximum(s[:, 0], s[:, 1])
                m2 = np.minimum(s[:, 0], s[:, 1])
                ev[name] = {"m1": round(float(np.median(m1)), 1),
                            "m2": round(float(np.median(m2)), 1),
                            "dL": round(float(np.median(s[:, 2])), 0),
                            "pub": pub}
                print(f"[{which}] {name}: m1 {ev[name]['m1']} (pub {pub['m1']}) "
                      f"dL {ev[name]['dL']} (pub {pub['dL']})", flush=True)
            except Exception as e:
                ev[name] = {"error": str(e)[:80]}
        report[which] = {"val_nll": round(ck["val_nll"], 3),
                         "val_nll_eval": round(nll, 3),
                         "cov90_all_params": round(cov90, 3),
                         "sbc_ks_failures": int(ks_bad),
                         "real_events": ev}
    json.dump(report, open(OUTDIR / "comparison.json", "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "real_events"}
                      for k, v in report.items()}, indent=1))
    print(f"-> {OUTDIR}/comparison.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--train", choices=["A", "B", "C"])
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    if args.extract:
        extract()
    if args.train:
        train(args.train, epochs=args.epochs)
    if args.eval:
        evaluate()
