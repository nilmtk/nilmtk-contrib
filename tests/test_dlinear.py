import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
metadata_module = pytest.importorskip("nilmtk_contrib.metadata")
engine_module = pytest.importorskip("nilmtk_contrib.torch._seq2point")
dlinear_module = pytest.importorskip("nilmtk_contrib.torch.dlinear")
model_catalog_by_module = metadata_module.model_catalog_by_module
SequenceToPointTorchDisaggregator = engine_module.SequenceToPointTorchDisaggregator
DLinear = dlinear_module.DLinear
DLinearNetwork = dlinear_module.DLinearNetwork
SeriesDecomposition = dlinear_module.SeriesDecomposition


def _params(**overrides):
    return {
        "device": "cpu",
        "sequence_length": 9,
        "moving_average": 3,
        "n_epochs": 1,
        "batch_size": 4,
        "seed": 17,
        "mains_mean": 100.0,
        "mains_std": 40.0,
    } | overrides


def _chunks(length=18):
    values = np.arange(length, dtype=np.float32)
    index = pd.date_range("2026-01-01", periods=length, freq="1min", tz="UTC")
    appliance = 25.0 + 30.0 * ((values // 3) % 2)
    mains = 70.0 + appliance + 3.0 * np.sin(values / 2)
    return (
        [pd.DataFrame({"power": mains}, index=index)],
        [("fridge", [pd.DataFrame({"power": appliance}, index=index)])],
    )


def test_dlinear_is_an_architecture_only_engine_consumer():
    source = Path(inspect.getfile(DLinear)).read_text()

    assert issubclass(DLinear, SequenceToPointTorchDisaggregator)
    assert len(source.splitlines()) < 180
    for method in ("partial_fit", "disaggregate_chunk", "save_model", "load_model"):
        assert f"def {method}(" not in source


def test_decomposition_reconstructs_input_and_preserves_constant_edges():
    decomposition = SeriesDecomposition(3)
    values = torch.tensor([[[1.0, 3.0, 2.0, 6.0, 4.0]]])
    seasonal, trend = decomposition(values)

    assert torch.allclose(seasonal + trend, values)
    constant = torch.full((2, 1, 5), 7.0)
    constant_seasonal, constant_trend = decomposition(constant)
    assert torch.allclose(constant_seasonal, torch.zeros_like(constant))
    assert torch.allclose(constant_trend, constant)


def test_network_has_separate_linear_trend_and_seasonal_paths():
    network = DLinearNetwork(sequence_length=9, moving_average=3)

    assert network.seasonal_linear.in_features == 9
    assert network.trend_linear.in_features == 9
    assert sum(parameter.numel() for parameter in network.parameters()) == 20
    assert network(torch.ones(4, 1, 9)).shape == (4, 1)
    assert torch.allclose(network(torch.ones(1, 1, 9)), torch.ones(1, 1))


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"sequence_length": 8}, "odd"),
        ({"moving_average": 4}, "must be odd"),
        ({"moving_average": 11}, "must not exceed"),
        ({"moving_average": True}, "positive integer"),
        ({"learning_rate": 0.0}, "greater than"),
        ({"validation_fraction": 1.0}, "less than 1"),
    ],
)
def test_invalid_configuration_fails_at_construction(params, message):
    with pytest.raises(ValueError, match=message):
        DLinear(_params(**params))


@pytest.mark.parametrize(
    "inputs",
    [
        torch.ones(2, 9),
        torch.ones(2, 1, 8),
        torch.ones(2, 1, 9, dtype=torch.int64),
        torch.full((2, 1, 9), float("nan")),
    ],
)
def test_network_rejects_invalid_inputs(inputs):
    network = DLinear(_params()).return_network()

    with pytest.raises((TypeError, ValueError)):
        network(inputs)


def test_one_epoch_train_infer_smoke_preserves_every_row():
    mains, targets = _chunks()
    model = DLinear(_params(appliance_params={}))

    model.partial_fit(mains, targets)
    result = model.disaggregate_chunk(mains)[0]

    assert result.index.equals(mains[0].index)
    assert result.columns.tolist() == ["fridge"]
    assert result["fridge"].dtype == np.float32
    assert np.isfinite(result["fridge"]).all()


def test_checkpoint_roundtrip_restores_architecture_and_predictions(tmp_path):
    mains, targets = _chunks()
    model = DLinear(
        _params(appliance_params={}, save_model_path=str(tmp_path), moving_average=5)
    )
    model.partial_fit(mains, targets)
    expected = model.disaggregate_chunk(mains)[0]

    loaded = DLinear(
        _params(
            appliance_params={},
            pretrained_model_path=str(tmp_path),
            moving_average=3,
        )
    )
    actual = loaded.disaggregate_chunk(mains)[0]

    assert loaded.moving_average == 5
    assert np.allclose(actual, expected)


def test_public_export_catalog_and_defaults_are_complete():
    import nilmtk_contrib.torch as torch_models

    default = DLinear({"device": "cpu"})
    entry = model_catalog_by_module()["nilmtk_contrib.torch.dlinear"]
    assert torch_models.DLinear is DLinear
    assert default.sequence_length == 299
    assert default.batch_size == 128
    assert default.moving_average == 25
    assert entry.class_name == "DLinear"
    assert entry.exported_from == "nilmtk_contrib.torch"
