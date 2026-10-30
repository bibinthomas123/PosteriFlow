"""Frozen basin regression probe.

analysis/attractor_basin (2026-07-16): converged coherent-encoder models
collapsed every event in {tau_HL ~ +7 ms} x {SNR >~ 20} x {heavy} to the SAME
(30,30), q->1 posterior (GW150914/GW170814) while 11/11 SBC, coverage,
railing and the corner table stayed green — the cell is a thin sky slice
diluted ~50:1 in every aggregate, so no population metric can see it. This
module holds a tiny FROZEN bank of delay-correct design-noise injections
inside and beside the cell so that failure class is measured directly, every
epoch and in CI.

Bank rows (H1+L1 only; V1 white-filled, matching how a real two-detector
event reaches the model):
  basin rows — GW150914-like source at the real sky geometry (tau_HL ~ +7 ms):
               SNR ~85 and ~24 (distance-scaled), q_true 0.65, and edge-on.
  ctrl rows  — the same source with mirrored (~ -7 ms) and zeroed delays.
               Must stay clean; a red ctrl row means the failure has spread
               beyond the known cell.

Collapse signature per row: median q-hat, posterior fraction at q > 0.95,
spurious railing, relative chirp-mass error. An affected model reads
q-hat ~ 1.0 / q>0.95 ~ 0.6 / rail ~ 0.6 on basin rows; a healthy one keeps
q>0.95 < 0.30 and rail < 0.20 everywhere (`basin_gate`).
"""
from __future__ import annotations

import numpy as np
import torch

from ahsd.models.lean_npe import PARAM_NAMES

SR, DUR, T_LEN = 4096, 4.0, 16384
# GW150914 epoch: fixes the Earth orientation that puts tau_HL at ~ +7 ms
GPS_REF = 1126259462.4
BASE = dict(mass_1=38.8, mass_2=33.4, luminosity_distance=440.0,
            ra=1.95, dec=-1.27, theta_jn=2.9, psi=1.75, phase=1.3,
            geocent_time=0.0, a1=0.0, a2=0.0)
DETS = ("H1", "L1")

# label, params, delay sign (+1 physical, -1 mirrored, 0 none), noise seed
ROWS = [
    ("basin snr85",  dict(BASE), +1, 101),
    ("basin snr24",  dict(BASE, luminosity_distance=1564.0), +1, 102),
    ("basin q.65",   dict(BASE, mass_1=44.8, mass_2=29.1), +1, 103),
    ("basin edgeon", dict(BASE, theta_jn=float(np.pi / 2)), +1, 104),
    ("ctrl tau-7",   dict(BASE), -1, 105),
    ("ctrl tau0",    dict(BASE), 0, 106),
]

_BANK = None


def _projected(params, sign):
    """Raw per-detector signal: merger at window centre + sign * geocentric
    time delay (sign=-1 mirrors the geometry, 0 drops it)."""
    import bilby
    wfg = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=SR,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={"waveform_approximant": "IMRPhenomXP",
                            "reference_frequency": 50.0, "minimum_frequency": 20.0})
    inj = dict(mass_1=params["mass_1"], mass_2=params["mass_2"],
               luminosity_distance=params["luminosity_distance"],
               theta_jn=params["theta_jn"], psi=params["psi"], phase=params["phase"],
               a_1=params["a1"], a_2=params["a2"], tilt_1=0.0, tilt_2=0.0,
               phi_12=0.0, phi_jl=0.0, ra=params["ra"], dec=params["dec"],
               geocent_time=GPS_REF)
    pol = wfg.time_domain_strain(inj)
    sig = {}
    for det in DETS:
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        h = np.zeros(int(SR * DUR))
        for m in ("plus", "cross"):
            h = h + ifo.antenna_response(inj["ra"], inj["dec"], GPS_REF,
                                         inj["psi"], m) * pol[m]
        dt = ifo.time_delay_from_geocenter(inj["ra"], inj["dec"], GPS_REF)
        sig[det] = np.roll(h, T_LEN // 2 + sign * int(round(dt * SR)))
    return sig


def _design_noise(det, seed):
    import bilby
    from ahsd.data.bilby_pipeline import get_default_psd
    p = get_default_psd(det)
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=np.asarray(p["frequencies"], dtype=float),
        psd_array=np.asarray(p["psd"], dtype=float))
    np.random.seed(seed)
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=SR, duration=DUR, start_time=0.0)
    return np.asarray(ifo.strain_data.time_domain_strain, dtype=np.float64)


def build_bank():
    """Deterministic bank, cached per process: strain [N,3,T] fp32 (whitened
    by the same BilbyPreprocessor path as dataset generation), truths [N,11],
    labels."""
    global _BANK
    if _BANK is not None:
        return _BANK
    from ahsd.inference.preprocessing import prepare_simulated
    strains, truths, labels = [], [], []
    for label, p, sign, seed in ROWS:
        sig = _projected(p, sign)
        raw = {det: _design_noise(det, seed * 10 + i) + sig[det]
               for i, det in enumerate(DETS)}
        prep = prepare_simulated(raw, seed=seed)
        strains.append(prep.strain)
        truths.append([p[k] for k in PARAM_NAMES])
        labels.append(label)
    _BANK = dict(strain=torch.from_numpy(np.stack(strains)).float(),
                 truth=torch.tensor(truths, dtype=torch.float32),
                 labels=labels)
    return _BANK


@torch.no_grad()
def evaluate_bank(model, bank, device, n_samples=256, psd_bands=16):
    """Collapse signature per bank row. Restores the model's training mode."""
    was_training = model.training
    model.eval()
    strain = bank["strain"].to(device)
    ab = (torch.zeros(strain.shape[0], 3, psd_bands, device=device)
          if getattr(model, "psd_cond", False) else None)
    s = model.sample_posterior(strain, rank=0, n_samples=n_samples,
                               asd_bands=ab)                      # [N,K,11]
    truth = bank["truth"].to(s.device)
    m1, m2 = s[..., 0], s[..., 1]
    q = torch.minimum(m1, m2) / torch.maximum(m1, m2)
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    yt = model.scaler.normalize(truth)
    noncirc = ~model.scaler.circ_mask
    rows = []
    for i, label in enumerate(bank["labels"]):
        yn = model.scaler.normalize(s[i])
        spur = (((yn > 0.999) & (yt[i] < 0.9))
                | ((yn < -0.999) & (yt[i] > -0.9)))[:, noncirc]
        mct = float((truth[i, 0] * truth[i, 1]) ** 0.6
                    / (truth[i, 0] + truth[i, 1]) ** 0.2)
        rows.append(dict(label=label,
                         q_med=round(float(q[i].median()), 3),
                         frac_q_gt95=round(float((q[i] > 0.95).float().mean()), 3),
                         rail=round(float(spur.any(-1).float().mean()), 3),
                         mc_rel_err=round(float(mc[i].median()) / mct - 1.0, 3)))
    if was_training:
        model.train()
    return rows


def log_basin(rows, log):
    log.info(f"  {'BASIN probe':16}{'q_med':>7}{'q>0.95':>8}{'rail':>7}{'Mc_err':>8}")
    for r in rows:
        log.info(f"  {r['label']:16}{r['q_med']:>7.3f}{r['frac_q_gt95']:>8.3f}"
                 f"{r['rail']:>7.3f}{r['mc_rel_err']:>+8.3f}")


def basin_gate(rows, q95_max=0.30, rail_max=0.20):
    """CI gate over the basin rows only (ctrl rows are context, not gated)."""
    basin = [r for r in rows if r["label"].startswith("basin")]
    ok = all(r["frac_q_gt95"] < q95_max and r["rail"] < rail_max for r in basin)
    worst = max(basin, key=lambda r: max(r["frac_q_gt95"], r["rail"]))
    return ok, (f"worst {worst['label']}: q>0.95 {worst['frac_q_gt95']:.2f} "
                f"rail {worst['rail']:.2f}")
