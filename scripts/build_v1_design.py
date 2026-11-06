#!/usr/bin/env python3
"""Write the Advanced Virgo design ASD into the noise bank, on the 4 s rfft grid.

design_asd_V1.npy stays as it is: it holds the flat curve older datasets were whitened with.
    python scripts/build_v1_design.py data/noise_bank
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ahsd.data.bilby_pipeline import V1_ADVIRGO, design_asd_file, get_default_psd

bank = Path(sys.argv[1])
ref = np.load(bank / "design_asd_H1.npy")
freqs = np.fft.rfftfreq(16384, 1.0 / 4096)
assert len(ref) == len(freqs), (len(ref), len(freqs))
p = get_default_psd("V1", V1_ADVIRGO)
asd = np.sqrt(np.interp(freqs, p["frequencies"], p["psd"])).astype(ref.dtype)
assert asd[np.argmin(abs(freqs - 20))] > 5 * asd[np.argmin(abs(freqs - 100))]     # not flat
out = bank / design_asd_file("V1", V1_ADVIRGO)
np.save(out, asd)
print(out, "ASD at 20, 100, 1000 Hz:", [float(asd[np.argmin(abs(freqs - f))]) for f in (20, 100, 1000)])
