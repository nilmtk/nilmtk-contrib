import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
metadata_module = pytest.importorskip("nilmtk_contrib.metadata")
engine_module = pytest.importorskip("nilmtk_contrib.torch._seq2point")
timesnet_module = pytest.importorskip("nilmtk_contrib.torch.timesnet")
model_catalog_by_module = metadata_module.model_catalog_by_module
SequenceToPointTorchDisaggregator = engine_module.SequenceToPointTorchDisaggregator
TimesNet = timesnet_module.TimesNet
TimesNetNetwork = timesnet_module.TimesNetNetwork
InceptionBlock2D = timesnet_module.InceptionBlock2D
dominant_periods = timesnet_module.dominant_periods


def _params(**overrides):
    return {
        "device": "cpu",
        "sequence_length": 9,
        "n_epochs": 1,
        "batch_size": 4,
        "seed": 17,
        "mains_mean": 100.0,
        "mains_std": 40.0,
        "d_model": 4,
        "d_ff": 8,
        "n_blocks": 1,
        "top_k": 2,
        "num_kernels": 2,
        "dropout": 0.0,
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


def test_timesnet_is_an_architecture_only_engine_consumer():
    source = Path(inspect.getfile(TimesNet)).read_text()

    assert issubclass(TimesNet, SequenceToPointTorchDisaggregator)
    assert len(source.splitlines()) < 300
    for method in ("partial_fit", "disaggregate_chunk", "save_model", "load_model"):
        assert f"def {method}(" not in source


def test_fft_period_selection_finds_a_clean_three_sample_cycle():
    time = torch.arange(9, dtype=torch.float32)
    signal = torch.sin(2 * math.pi * time / 3).reshape(1, 9, 1)

    periods, amplitudes = dominant_periods(signal, top_k=2)

    assert periods[0].item() == 3
    assert amplitudes.shape == (1, 2)
    assert amplitudes[0, 0] > amplitudes[0, 1]


@pytest.mark.parametrize(
    "inputs",
    [
        object(),
        torch.ones(2, 9),
        torch.ones(1, 2, 1),
        torch.ones(1, 9, 1, dtype=torch.int64),
        torch.full((1, 9, 1), float("nan")),
    ],
)
def test_fft_period_selection_rejects_invalid_inputs(inputs):
    with pytest.raises((TypeError, ValueError)):
        dominant_periods(inputs, top_k=2)


@pytest.mark.parametrize("top_k", [True, 5])
def test_fft_period_selection_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError):
        dominant_periods(torch.ones(1, 9, 1), top_k=top_k)


def test_inception_uses_parallel_odd_shape_preserving_kernels():
    block = InceptionBlock2D(4, 8, num_kernels=3)
    inputs = torch.ones(2, 4, 3, 5)

    assert [kernel.kernel_size for kernel in block.kernels] == [(1, 1), (3, 3), (5, 5)]
    assert block(inputs).shape == (2, 8, 3, 5)


def test_network_preserves_shape_and_backpropagates_through_every_parameter():
    network = TimesNet(_params()).return_network()
    output = network(torch.ones(3, 1, 9))
    output.sum().backward()

    assert output.shape == (3, 1)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
    )


def test_constant_input_has_finite_frequency_aggregation():
    network = TimesNet(_params()).return_network()

    output = network(torch.ones(4, 1, 9))

    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"sequence_length": 8}, "odd"),
        ({"sequence_length": 1}, "at least 3"),
        ({"top_k": 5}, "non-DC"),
        ({"top_k": True}, "positive integer"),
        ({"num_kernels": 0}, "positive integer"),
        ({"d_model": 3.5}, "positive integer"),
        ({"dropout": 1.0}, "less than 1"),
        ({"weight_decay": -1.0}, "at least"),
    ],
)
def test_invalid_configuration_fails_at_construction(params, message):
    with pytest.raises(ValueError, match=message):
        TimesNet(_params(**params))


@pytest.mark.parametrize(
    "params",
    [
        {"sequence_length": 1},
        {"sequence_length": 8},
        {"sequence_length": 9, "top_k": 5},
        {"sequence_length": 9, "d_model": True},
        {"sequence_length": 9, "dropout": 1.0},
    ],
)
def test_network_constructor_enforces_its_public_contract(params):
    with pytest.raises(ValueError):
        TimesNetNetwork(**params)


@pytest.mark.parametrize(
    "inputs",
    [
        object(),
        torch.ones(2, 9),
        torch.ones(2, 1, 8),
        torch.ones(2, 1, 9, dtype=torch.int64),
        torch.full((2, 1, 9), float("nan")),
    ],
)
def test_network_rejects_invalid_inputs(inputs):
    network = TimesNet(_params()).return_network()

    with pytest.raises((TypeError, ValueError)):
        network(inputs)


def test_one_epoch_train_infer_smoke_preserves_every_row():
    mains, targets = _chunks()
    model = TimesNet(_params(appliance_params={}))

    model.partial_fit(mains, targets)
    result = model.disaggregate_chunk(mains)[0]

    assert result.index.equals(mains[0].index)
    assert result.columns.tolist() == ["fridge"]
    assert result["fridge"].dtype == np.float32
    assert np.isfinite(result["fridge"]).all()


def test_checkpoint_roundtrip_restores_architecture_and_predictions(tmp_path):
    mains, targets = _chunks()
    model = TimesNet(
        _params(appliance_params={}, save_model_path=str(tmp_path), d_model=6)
    )
    model.partial_fit(mains, targets)
    expected = model.disaggregate_chunk(mains)[0]

    loaded = TimesNet(
        _params(
            appliance_params={},
            pretrained_model_path=str(tmp_path),
            d_model=4,
        )
    )
    actual = loaded.disaggregate_chunk(mains)[0]

    assert loaded.d_model == 6
    assert np.allclose(actual, expected)


def test_public_export_catalog_and_defaults_are_complete():
    import nilmtk_contrib.torch as torch_models

    default = TimesNet({"device": "cpu"})
    entry = model_catalog_by_module()["nilmtk_contrib.torch.timesnet"]
    assert torch_models.TimesNet is TimesNet
    assert default.sequence_length == 299
    assert default.batch_size == 128
    assert default.top_k == 3
    assert default.weight_decay == pytest.approx(1e-4)
    assert entry.class_name == "TimesNet"
    assert entry.exported_from == "nilmtk_contrib.torch"
