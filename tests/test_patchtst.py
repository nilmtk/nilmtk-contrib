import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("nilmtk") is None
    or importlib.util.find_spec("torch") is None,
    reason="PatchTST tests require the torch and nilm extras",
)


def test_patchtst_network_covers_window_with_overlapping_patches():
    import torch

    from nilmtk_contrib.torch.patchtst import PatchTSTNetwork

    network = PatchTSTNetwork(
        sequence_length=99,
        patch_length=16,
        patch_stride=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
    )
    prediction = network(torch.randn(3, 1, 99))

    assert network.num_patches == 12
    assert prediction.shape == (3, 1)
    assert torch.isfinite(prediction).all()


def test_patchtst_requires_odd_sequence_length():
    from nilmtk_contrib.torch.patchtst import PatchTST, SequenceLengthError

    with pytest.raises(SequenceLengthError, match="odd"):
        PatchTST({"sequence_length": 100, "device": "cpu"})


def test_patchtst_rejects_patch_gaps():
    from nilmtk_contrib.torch.patchtst import PatchTST

    with pytest.raises(ValueError, match="patch_stride"):
        PatchTST(
            {
                "sequence_length": 99,
                "patch_length": 8,
                "patch_stride": 9,
                "device": "cpu",
            }
        )
