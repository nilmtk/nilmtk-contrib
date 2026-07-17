import importlib.util

import numpy as np
import pandas as pd
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("nilmtk") is None
    or importlib.util.find_spec("torch") is None,
    reason="PatchTST tests require the torch and nilm extras",
)


def _small_params(**overrides):
    params = {
        "sequence_length": 9,
        "patch_length": 3,
        "patch_stride": 2,
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 16,
        "dropout": 0.0,
        "n_epochs": 1,
        "batch_size": 4,
        "mains_mean": 100.0,
        "mains_std": 50.0,
        "seed": 7,
        "device": "cpu",
    }
    params.update(overrides)
    return params


def _training_frames(samples=24):
    index = pd.date_range("2024-01-01", periods=samples, freq="1min")
    appliance = np.where(np.arange(samples) % 5 < 2, 80.0, 0.0).astype(np.float32)
    mains = appliance + np.linspace(20.0, 40.0, samples, dtype=np.float32)
    return (
        [pd.DataFrame({"power": mains}, index=index)],
        [("fridge", [pd.DataFrame({"power": appliance}, index=index)])],
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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_epochs": 0}, "n_epochs"),
        ({"mains_mean": float("nan")}, "mains_mean"),
        ({"mains_mean": float("inf")}, "mains_mean"),
        ({"mains_std": float("nan")}, "mains_std"),
        ({"mains_std": float("inf")}, "mains_std"),
        ({"mains_std": -1.0}, "mains_std"),
        ({"mains_std": True}, "mains_std"),
    ],
)
def test_patchtst_rejects_invalid_training_parameters(overrides, message):
    from nilmtk_contrib.torch.patchtst import PatchTST

    with pytest.raises(ValueError, match=message):
        PatchTST(_small_params(**overrides))


def test_patchtst_network_validates_geometry_when_used_directly():
    from nilmtk_contrib.torch.patchtst import PatchTSTNetwork

    with pytest.raises(ValueError, match="patch_length"):
        PatchTSTNetwork(
            sequence_length=7,
            patch_length=8,
            patch_stride=4,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
        )


def test_patchtst_rejects_nonfinite_training_data():
    from nilmtk_contrib.torch.patchtst import PatchTST

    model = PatchTST(_small_params())
    mains = [pd.DataFrame(np.zeros((3, 9), dtype=np.float32))]
    mains[0].iloc[0, 0] = np.nan
    appliances = [("fridge", [pd.DataFrame([0.0, 1.0, 0.0])])]

    with pytest.raises(ValueError, match="finite"):
        model.partial_fit(mains, appliances, do_preprocessing=False)


def test_patchtst_requires_a_trained_model_for_inference():
    from nilmtk_contrib.torch.patchtst import PatchTST

    model = PatchTST(_small_params())
    mains, _ = _training_frames()

    with pytest.raises(RuntimeError, match="trained|loaded"):
        model.disaggregate_chunk(mains)


def test_patchtst_checkpoint_round_trip_preserves_predictions(tmp_path):
    from nilmtk_contrib.torch.patchtst import PatchTST

    mains, appliances = _training_frames()
    save_params = _small_params(save_model_path=str(tmp_path))
    trained = PatchTST(save_params)
    trained.partial_fit(mains, appliances)
    expected = trained.disaggregate_chunk(mains)[0]
    trained.save_model()

    loaded = PatchTST(
        _small_params(
            seed=99,
            pretrained_model_path=str(tmp_path),
        )
    )
    observed = loaded.disaggregate_chunk(mains)[0]

    pd.testing.assert_frame_equal(expected, observed, rtol=1e-6, atol=1e-6)
