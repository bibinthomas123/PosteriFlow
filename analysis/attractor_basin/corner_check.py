"""Corner-table check for saved mcq checkpoints (epochs 20/30/40).

Reuses the trainer's own load_split / run_diagnostics / log_corner so the
numbers are directly comparable to the per-epoch console tables.
"""
import logging
import sys

sys.path.insert(0, "/Users/bibinthomas/Developer/PosteriFlow")
sys.path.insert(0, "/Users/bibinthomas/Developer/PosteriFlow/experiments")
sys.path.insert(0, "/Users/bibinthomas/Developer/PosteriFlow/src")

import torch

import train_lean_npe as T
from ahsd.models.lean_npe import LeanNPE

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("corner")

CKPTS = [
    ("epoch 20", "model/lean_npe_mcq/epoch_0020.pth"),
    ("epoch 30", "model/lean_npe_mcq/epoch_0030.pth"),
    ("epoch 40 (best)", "model/lean_npe_mcq/best_model.pth"),
]

device = "mps" if torch.backends.mps.is_available() else "cpu"
root = "/Users/bibinthomas/Developer/PosteriFlow/"

first = torch.load(root + CKPTS[0][1], map_location="cpu", weights_only=False)
data_dir = first["args"].get("data", "data/dataset")
diag_events = first["args"].get("diag_events", 1024)
psd_bands = first["args"].get("psd_bands", 16)

log.info(f"loading validation split from {data_dir} ...")
va_s, va_p, va_n, va_snr = T.load_split(root + data_dir, "validation")
log.info(f"val events: {va_s.shape[0]} | diag_events={diag_events}")
va_asd = torch.zeros(min(diag_events, va_s.shape[0]), 3, psd_bands)

for label, path in CKPTS:
    ck = torch.load(root + path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = LeanNPE(premerger=a.get("premerger", False),
                    psd_cond=a.get("psd_cond", False) or False,
                    psd_bands=a.get("psd_bands", 16),
                    encoder_type=a.get("encoder_type", "conv"),
                    mass_param=a.get("mass_param", "m1m2"),
                    geom_phase_residual=a.get("geom_phase_residual", False),
                    mf_tokens=a.get("mf_tokens", False),
                    mf_shape_norm=a.get("mf_shape_norm", False), mf_token_fusion=a.get("mf_token_fusion", "add"), mf_bank=a.get("mf_bank", "mf_bank.npz"), pol_features=a.get("pol_features", False), q_embed_dim=a.get("q_embed_dim", 0))
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    torch.manual_seed(0)
    diag = T.run_diagnostics(model, va_s, va_p, va_n, device,
                             asd_bands=va_asd, snr=va_snr, n_events=diag_events)
    log.info(f"\n=== {label} (ckpt epoch {ck['epoch']}, val_nll {ck['val_nll']:.4f}) ===")
    log.info(f"  shufD {diag['shuffle_delta_nll']:+.2f} | dcorr {diag['dist_corr']:+.3f} | "
             f"rail {diag['spurious_railing']:.4f} | zc {diag['base_conc']:.3f} | "
             f"cov90 {diag['sci_cov90']:.3f} covHi {diag['sci_cov90_highsnr']:.3f} | "
             f"sbc {diag['sbc_pass_frac']:.2f} | viol {diag['support_viol']:.4f}")
    T.log_corner(diag, log)
    del model
    if device == "mps":
        torch.mps.empty_cache()

log.info("\ndone")
