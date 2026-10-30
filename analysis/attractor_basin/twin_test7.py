"""Round 7: which encoder generations share the +7 ms basin?

Runs {real GW150914, design twin at Δt +7 / -7 / 0 ms} through every loadable
checkpoint generation, printing encoder_type/mass_param provenance. If the
pre-coherent (conv) generation lacks the +7 ms lobe, the coherent geometry
branch is indicted; if it shares it, the cause predates the coherent encoder.
"""
import sys
from pathlib import Path

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))

import numpy as np
import torch

from twin_test import analyze
from twin_test6 import prep_with_dt
import _common as C

CANDIDATES = {
    "mcq": "model/lean_npe_mcq/best_model.pth",
    "m1m2-old": "model/lean_npe/best_model.pth",
    "era": "model/lean_npe_era/best_model.pth",
    "v8-conv": "model/lean_v8_psdcond_ft/best_model.pth",
}


def main():
    models = {}
    for name, rel in CANDIDATES.items():
        path = C.ROOT / rel
        if not path.exists():
            print(f"model {name}: MISSING ({rel})")
            continue
        try:
            ck = torch.load(str(path), map_location="cpu", weights_only=False)
            a = ck["args"]
            m = C.load_model(str(path))
            models[name] = m
            print(f"model {name}: epoch {ck['epoch']} val_nll {ck['val_nll']:.3f} "
                  f"encoder={a.get('encoder_type', 'conv')} mass_param={a.get('mass_param', 'm1m2')} "
                  f"psd_cond={a.get('psd_cond')}")
        except Exception as e:
            print(f"model {name}: LOAD FAILED ({type(e).__name__}: {str(e)[:160]})")

    zeros_ab = np.zeros((3, 16), dtype=np.float32)
    truth = dict(C.GW150914_TRUTH)

    print("\n=== real GW150914 ===")
    prep_real = C.prepare_gw150914()
    ab_real = C.asd_bands_of(prep_real)
    for name, m in models.items():
        analyze(m, prep_real, truth, ab_real, f"real / {name}")

    for dms in (+7, -7, 0):
        print(f"\n=== design twin Δt {dms:+d} ms ===")
        prep = prep_with_dt(truth, dms)
        for name, m in models.items():
            analyze(m, prep, truth, zeros_ab, f"dt{dms:+d} / {name}")


if __name__ == "__main__":
    main()
