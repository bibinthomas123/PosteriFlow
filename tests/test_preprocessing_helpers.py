"""Real-data preprocessing helpers (ahsd.inference.preprocessing)."""
import numpy as np


def test_finite_run_around_the_trigger():
    from ahsd.inference.preprocessing import _finite_run_around
    x = np.arange(100.0)
    x[10:20] = np.nan
    x[80:] = np.nan
    assert _finite_run_around(x, 50) == (20, 80)
    assert _finite_run_around(x, 5) == (0, 10)
    assert _finite_run_around(x, 15) is None


def test_likelihood_seconds_follows_the_chirp_time():
    from ahsd.inference.preprocessing import likelihood_seconds
    from ahsd.models.lean_npe import PARAM_NAMES
    def rows(m1, m2, tc):
        s = np.zeros((100, len(PARAM_NAMES)))
        s[:, 0], s[:, 1], s[:, PARAM_NAMES.index("geocent_time")] = m1, m2, tc
        return s
    assert likelihood_seconds(rows(36, 30, 0.0)) == 4.0          # GW150914-like, centred trigger
    assert likelihood_seconds(rows(36, 30, -1.5)) == 8.0         # early merger keeps its band
    assert likelihood_seconds(rows(1.5, 1.3, 0.0)) >= 128.0      # BNS
