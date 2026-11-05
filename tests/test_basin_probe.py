"""The basin probe must take its PSD band count from the model it evaluates."""
import pytest
import torch

from ahsd.evaluation.basin_probe import BASE, evaluate_bank
from ahsd.models.lean_npe import PARAM_NAMES, LeanNPE


def _bank():
    return dict(strain=torch.randn(2, 3, 16384),
                truth=torch.tensor([[BASE[k] for k in PARAM_NAMES]] * 2, dtype=torch.float32),
                labels=["basin a", "ctrl b"])


@pytest.mark.parametrize("bands", [16, 64])
def test_evaluate_bank_uses_model_psd_bands(bands):
    torch.manual_seed(0)
    model = LeanNPE(psd_cond=True, psd_bands=bands)
    rows = evaluate_bank(model, _bank(), "cpu", n_samples=4)
    assert [r["label"] for r in rows] == ["basin a", "ctrl b"]
