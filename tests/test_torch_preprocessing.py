import numpy as np
import pandas as pd
import pytest

from nilmtk_contrib.torch.preprocessing import ApplianceNotFoundError, preprocess


def test_preprocess_test_mode_returns_normalized_mains_windows_only():
    mains = [pd.DataFrame({"power": [10.0, 20.0, 30.0]})]

    processed = preprocess(
        sequence_length=3,
        mains_mean=10,
        mains_std=10,
        mains_lst=mains,
        method="test",
    )

    assert len(processed) == 1
    assert processed[0].shape == (3, 3)
    assert processed[0].to_numpy().tolist() == [
        [-1.0, 0.0, 1.0],
        [0.0, 1.0, 2.0],
        [1.0, 2.0, -1.0],
    ]


def test_preprocess_train_mode_normalizes_flat_appliance_targets():
    mains = [pd.DataFrame({"power": [10.0, 20.0, 30.0]})]
    appliances = [("fridge", [pd.DataFrame({"power": [5.0, 10.0, 15.0]})])]

    processed_mains, processed_apps = preprocess(
        sequence_length=3,
        mains_mean=10,
        mains_std=10,
        mains_lst=mains,
        submeters_lst=appliances,
        appliance_params={"fridge": {"mean": 5, "std": 5}},
    )

    assert processed_mains[0].shape == (3, 3)
    assert processed_apps[0][0] == "fridge"
    assert processed_apps[0][1][0].shape == (3, 1)
    assert processed_apps[0][1][0].to_numpy().flatten().tolist() == [0.0, 1.0, 2.0]


def test_preprocess_train_mode_can_window_appliance_targets():
    mains = [pd.DataFrame({"power": [10.0, 20.0]})]
    appliances = [("fridge", [pd.DataFrame({"power": [5.0, 15.0]})])]

    _, processed_apps = preprocess(
        sequence_length=3,
        mains_mean=10,
        mains_std=10,
        mains_lst=mains,
        submeters_lst=appliances,
        appliance_params={"fridge": {"mean": 5, "std": 5}},
        windowing=True,
    )

    np.testing.assert_allclose(
        processed_apps[0][1][0].to_numpy(),
        np.array([[-1.0, 0.0, 2.0], [0.0, 2.0, -1.0]], dtype=np.float32),
    )


def test_preprocess_rejects_missing_appliance_params():
    with pytest.raises(ApplianceNotFoundError, match="fridge"):
        preprocess(
            sequence_length=3,
            mains_mean=10,
            mains_std=10,
            mains_lst=[pd.DataFrame({"power": [10.0]})],
            submeters_lst=[("fridge", [pd.DataFrame({"power": [5.0]})])],
            appliance_params={},
        )


def test_preprocess_returns_only_mains_when_no_submeters_are_supplied():
    processed = preprocess(
        sequence_length=3,
        mains_mean=10,
        mains_std=10,
        mains_lst=[pd.DataFrame({"power": [10.0]})],
        submeters_lst=None,
    )

    assert len(processed) == 1
    assert processed[0].shape == (1, 3)
