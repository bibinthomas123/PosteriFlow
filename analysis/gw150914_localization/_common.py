"""
Shared helpers for the GW150914 distance-loss localization study.

Everything downstream imports from here so we prepare the event and load the
model exactly once per process, with the CANONICAL checkpoint.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import torch

from ahsd.models.lean_npe import LeanNPE, PARAM_NAMES, ParamScaler
from ahsd.inference import preprocessing as pp

# ── canonical model + event constants ────────────────────────────────────────
CANON = str(ROOT / "model" / "lean_v8_psdcond_ft" / "best_model.pth")
GW150914_GPS = 1126259462.4
# LVC detector-frame pseudo-truth used throughout the repo's benchmarks
GW150914_TRUTH = dict(mass_1=38.8, mass_2=33.4, luminosity_distance=440.0,
                      theta_jn=2.9, ra=1.95, dec=-1.27, psi=1.75, phase=1.3,
                      geocent_time=0.0, a1=0.0, a2=0.0)
JD = PARAM_NAMES.index("luminosity_distance")
DEVICE = "cpu"
SR, T_LEN, DUR = pp.SAMPLE_RATE, pp.T_LEN, pp.DURATION


def load_model(path=CANON, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = LeanNPE(premerger=a.get("premerger", False),
                    psd_cond=a.get("psd_cond", False) or False,
                    psd_bands=a.get("psd_bands", 16),
                    encoder_type=a.get("encoder_type", "conv"))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    model._ckpt_meta = dict(epoch=ckpt.get("epoch"), val_nll=ckpt.get("val_nll"),
                            psd_cond=a.get("psd_cond"), psd_bands=a.get("psd_bands"))
    return model


def fetch_gw150914(fetch_seconds=64, detectors=("H1", "L1")):
    """Cached GWOSC fetch of the raw strain (V1 not operational in O1)."""
    gps, strain, starts, srs, found = pp.fetch_gwosc(
        GW150914_GPS, detectors=list(detectors), fetch_seconds=fetch_seconds)
    return gps, strain, starts, srs, found


def prepare_gw150914(fetch_seconds=64, detectors=("H1", "L1"), seed=0):
    gps, strain, starts, srs, found = fetch_gw150914(fetch_seconds, detectors)
    prep = pp.prepare_real(strain, gps, starts, srs, seed=seed)
    return prep


def asd_bands_of(prep, psd_bands=16):
    ab = pp.compute_asd_bands(prep, psd_bands=psd_bands,
                              design_asd_dir=str(ROOT / "data" / "noise_bank"))
    return ab  # [3, psd_bands] float32


def encode(model, prep, psd_bands=16):
    """Return the 256-D encoder context for a PreparedData."""
    strain = torch.from_numpy(prep.strain[None]).float().to(DEVICE)
    ab = None
    if model._ckpt_meta.get("psd_cond"):
        ab = torch.from_numpy(asd_bands_of(prep, psd_bands)[None]).float().to(DEVICE)
    with torch.no_grad():
        ctx = model.encode(strain, ab)
    return ctx[0].cpu().numpy()


def sample(model, prep, rank=0, n=4000, psd_bands=16, seed=0):
    """Posterior samples [n, 11] physical units, on a PreparedData."""
    torch.manual_seed(seed)
    strain = torch.from_numpy(prep.strain[None]).float().to(DEVICE)
    ab = None
    if model._ckpt_meta.get("psd_cond"):
        ab = torch.from_numpy(asd_bands_of(prep, psd_bands)[None]).float().to(DEVICE)
    with torch.no_grad():
        s = model.sample_posterior(strain, rank=rank, n_samples=n, asd_bands=ab)
    return s[0].cpu().numpy()


def sample_from_strain(model, strain_np, asd_bands_np=None, rank=0, n=4000, seed=0):
    """Posterior samples from a raw [3,T] whitened strain array + optional asd_bands."""
    torch.manual_seed(seed)
    strain = torch.from_numpy(np.asarray(strain_np)[None]).float().to(DEVICE)
    ab = None
    if model._ckpt_meta.get("psd_cond") and asd_bands_np is not None:
        ab = torch.from_numpy(np.asarray(asd_bands_np)[None]).float().to(DEVICE)
    with torch.no_grad():
        s = model.sample_posterior(strain, rank=rank, n_samples=n, asd_bands=ab)
    return s[0].cpu().numpy()


def summ(x):
    x = np.asarray(x)
    return dict(median=float(np.median(x)), mean=float(np.mean(x)),
                std=float(np.std(x)),
                ci90=[float(np.quantile(x, .05)), float(np.quantile(x, .95))],
                ci50=[float(np.quantile(x, .25)), float(np.quantile(x, .75))])


# ── matched-filter network SNR on the SAME whitened representation ────────────

def whiten_like_pipeline(raw_win, asd_win):
    """Whiten a raw 4 s window with the noise-bank recipe: irfft(rfft/ASD),
    zero sub-18 Hz. asd_win is on the 4 s rfft grid (as in PreparedData.asds)."""
    n = len(raw_win)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    wf = np.fft.rfft(raw_win) / np.maximum(asd_win, 1e-30)
    wf[freqs < pp.HIGHPASS_HZ + 3.0] = 0.0
    return np.fft.irfft(wf, n=n)


def raw_projected_signal(params, gps_epoch, approximant="IMRPhenomXP", f_min=20.0,
                         detectors=("H1", "L1")):
    """Raw (unwhitened) detector strain [T_LEN] per detector for an IMRPhenomXP
    source at `params`, projected at GPS epoch `gps_epoch`, merger at window
    centre. Reusable for injections and matched-filter templates."""
    import bilby
    wfg = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=SR,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={"waveform_approximant": approximant,
                            "reference_frequency": 50.0, "minimum_frequency": f_min})
    inj = dict(mass_1=params["mass_1"], mass_2=params["mass_2"],
               luminosity_distance=params["luminosity_distance"],
               theta_jn=params.get("theta_jn", 0.0), psi=params.get("psi", 0.0),
               phase=params.get("phase", 0.0), a_1=params.get("a1", 0.0),
               a_2=params.get("a2", 0.0), tilt_1=0.0, tilt_2=0.0, phi_12=0.0,
               phi_jl=0.0, ra=params.get("ra", 0.0), dec=params.get("dec", 0.0),
               geocent_time=gps_epoch + params.get("geocent_time", 0.0))
    pol = wfg.time_domain_strain(inj)
    sig = {}
    for det in detectors:
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        h = np.zeros(int(SR * DUR))
        for m in ("plus", "cross"):
            resp = ifo.antenna_response(inj["ra"], inj["dec"], inj["geocent_time"],
                                        inj["psi"], m)
            h = h + resp * pol[m]
        sig[det] = np.roll(h, T_LEN // 2)   # merger at window centre
    return sig


def inject_into_real_noise(off_gps, params, fetch_seconds=64,
                           detectors=("H1", "L1"), approximant="IMRPhenomXP",
                           seed=0):
    """Semi-real injection: raw IMRPhenomXP signal added into raw off-source
    real strain, then whitened by the SAME prepare_real path a real event uses.
    Returns (PreparedData, snr_dict)."""
    gps, strain, starts, srs, found = pp.fetch_gwosc(
        off_gps, detectors=list(detectors), fetch_seconds=fetch_seconds)
    sig = raw_projected_signal(params, off_gps, approximant, detectors=found)
    raw = {}
    for det in found:
        v = np.array(strain[det], dtype=np.float64)
        c = int(round((off_gps - starts[det]) * SR))     # trigger index in segment
        i0 = c - T_LEN // 2
        v[i0:i0 + T_LEN] = v[i0:i0 + T_LEN] + sig[det]
        raw[det] = v
    prep = pp.prepare_real(raw, off_gps, starts, srs, seed=seed)
    snr = network_mf_snr(prep, params, approximant)
    return prep, snr


def network_mf_snr(prep, params, approximant="IMRPhenomXP", f_min=20.0):
    """Matched-filter & optimal network SNR of an IMRPhenomXP template at
    `params` computed on the SAME whitened representation the model sees.

    Returns dict per detector + network, using whitened inner products
    (unit-variance white noise => rho_opt = ||h_w||, rho_mf = <d_w,h_w>/||h_w||).

    The pipeline reaches a unit white floor by dividing the whitened strain by
    the off-source std `norm`; the template MUST be scaled by the SAME norm to
    live in the same unit-variance convention as the data window.
    """
    from scipy.signal import correlate, hilbert
    epoch = prep.trigger_gps if prep.trigger_gps is not None else GW150914_GPS
    sig = raw_projected_signal(params, epoch, approximant, f_min,
                               detectors=tuple(prep.detectors_present))
    out = {}
    net_opt2 = net_mf2 = 0.0
    max_lag = int(1.0 * SR)                       # search merger time within +-1 s of center
    for i, det in enumerate(pp.DETECTORS):
        if det not in prep.detectors_present:
            continue
        h = sig[det]
        asd_win = np.asarray(prep.asds[det], dtype=np.float64)
        norm = float(prep.quality[det]["whitening_norm"])
        h_w = whiten_like_pipeline(h, asd_win) / (norm + 1e-30)
        d_w = prep.strain[i].astype(np.float64)
        opt = float(np.sqrt(np.sum(h_w ** 2)))          # optimal SNR at these params
        hq = np.imag(hilbert(h_w))                        # 90-deg quadrature template
        # phase- & time-maximised matched filter (plain-sum cross-correlations)
        zI = correlate(d_w, h_w, mode="full")
        zQ = correlate(d_w, hq, mode="full")
        rho = np.sqrt(zI ** 2 + zQ ** 2) / (opt + 1e-30)
        center = len(h_w) - 1                             # zero-lag index in 'full'
        lo, hi = center - max_lag, center + max_lag
        mf = float(rho[lo:hi].max())
        out[det] = dict(optimal_snr=opt, mf_snr=mf)
        net_opt2 += opt ** 2
        net_mf2 += mf ** 2
    out["network"] = dict(optimal_snr=float(np.sqrt(net_opt2)),
                          mf_snr=float(np.sqrt(net_mf2)))
    return out


if __name__ == "__main__":
    m = load_model()
    print("model", m._ckpt_meta)
    prep = prepare_gw150914()
    print("present", prep.detectors_present, "warnings", prep.warnings)
    print("asd_bands\n", np.round(asd_bands_of(prep), 3))
    s = sample(m, prep, n=4000)
    print("dL", summ(s[:, JD]))
