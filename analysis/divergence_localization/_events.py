"""Shared event reference values for the divergence-localization study.

Published DETECTOR-frame medians are taken verbatim from the project's own CI
reference (scripts/validate_checkpoint.py REAL_EVENTS), so overlays are
self-consistent with the CI reports being investigated.

Network matched-filter SNRs are GWTC-1 catalog values (LVC 2019,
arXiv:1811.12907, Table). They are annotations only; the training-SNR audit is
computed from the actual stored whitened signals.
"""
import numpy as np

# name -> detector-frame published medians (m1,m2 solar, dL Mpc) + GWTC-1 net SNR
EVENTS = {
    "GW150914": dict(m1=38.8, m2=33.4, dL=440.0,  snr=24.4, gps=1126259462.4),
    "GW151226": dict(m1=13.7, m2=7.7,  dL=440.0,  snr=13.1, gps=1135136350.6),
    "GW170104": dict(m1=31.0, m2=20.1, dL=960.0,  snr=13.0, gps=1167559936.6),
    "GW170608": dict(m1=12.0, m2=7.0,  dL=320.0,  snr=15.4, gps=1180922494.5),
    "GW170729": dict(m1=50.6, m2=34.3, dL=2750.0, snr=10.8, gps=1185389807.3),  # OOD in distance
    "GW170814": dict(m1=30.7, m2=25.3, dL=600.0,  snr=15.9, gps=1186741861.5),
}

# Events the prompt flags as *worsening* with continued training.
PROBLEM = ("GW150914", "GW170814")

# Training-distribution priors (ParamScaler.RANGES) for edge reference.
PRIOR = dict(
    mass=(1.0, 105.0),          # both masses, log-space
    dL=(40.0, 2200.0),          # log-space
)


def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def derived(ev):
    m1, m2 = ev["m1"], ev["m2"]
    return dict(
        Mc=chirp_mass(m1, m2),
        Mtot=m1 + m2,
        q=m2 / m1,
        **ev,
    )


def event_table():
    rows = {}
    for name, ev in EVENTS.items():
        d = derived(ev)
        rows[name] = d
    return rows


if __name__ == "__main__":
    print(f"{'event':10s} {'m1':>6} {'m2':>6} {'Mtot':>6} {'Mc':>6} {'q':>5} "
          f"{'dL':>6} {'snr':>5}")
    for n, d in event_table().items():
        print(f"{n:10s} {d['m1']:6.1f} {d['m2']:6.1f} {d['Mtot']:6.1f} "
              f"{d['Mc']:6.1f} {d['q']:5.2f} {d['dL']:6.0f} {d['snr']:5.1f}")
