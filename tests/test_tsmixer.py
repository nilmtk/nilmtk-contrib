import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
metadata_module = pytest.importorskip("nilmtk_contrib.metadata")
engine_module = pytest.importorskip("nilmtk_contrib.torch._seq2point")
tsmixer_module = pytest.importorskip("nilmtk_contrib.torch.tsmixer")
model_catalog_by_module = metadata_module.model_catalog_by_module
SequenceToPointTorchDisaggregator = engine_module.SequenceToPointTorchDisaggregator
TSMixer = tsmixer_module.TSMixer
TSMixerBlock = tsmixer_module.TSMixerBlock
TSMixerNetwork = tsmixer_module.TSMixerNetwork


def _params(**overrides):
    return {
        "device": "cpu",
        "sequence_length": 9,
        "n_epochs": 1,
        "batch_size": 4,
        "seed": 17,
        "mains_mean": 100.0,
        "mains_std": 40.0,
        "ff_dim": 8,
        "n_blocks": 2,
        "dropout": 0.0,
        "activation": "relu",
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


def test_tsmixer_is_an_architecture_only_engine_consumer():
    source = Path(inspect.getfile(TSMixer)).read_text()

    assert issubclass(TSMixer, SequenceToPointTorchDisaggregator)
    assert len(source.splitlines()) < 240
    for method in ("partial_fit", "disaggregate_chunk", "save_model", "load_model"):
        assert f"def {method}(" not in source


def test_block_matches_the_published_time_then_feature_mixing_shape():
    block = TSMixerBlock(
        sequence_length=9,
        channels=3,
        ff_dim=7,
        dropout=0.0,
        activation="gelu",
    )
    inputs = torch.randn(4, 9, 3)

    output = block(inputs)

    assert output.shape == inputs.shape
    assert block.temporal_linear.in_features == 9
    assert block.temporal_linear.out_features == 9
    assert block.feature_in.in_features == 3
    assert block.feature_in.out_features == 7
    assert block.feature_out.in_features == 7
    assert block.feature_out.out_features == 3


def test_network_is_all_mlp_without_attention_convolution_or_recurrence():
    network = TSMixer(_params()).return_network()
    forbidden = (
        torch.nn.modules.conv._ConvNd,
        torch.nn.RNNBase,
        torch.nn.MultiheadAttention,
    )

    assert not any(isinstance(module, forbidden) for module in network.modules())
    assert sum(isinstance(module, torch.nn.Linear) for module in network.modules()) == 7


@pytest.mark.parametrize("activation", ["relu", "ReLU", "gelu", "GELU"])
def test_supported_activations_preserve_shape_and_backpropagate(activation):
    network = TSMixer(_params(activation=activation)).return_network()
    output = network(torch.randn(3, 1, 9))
    output.sum().backward()

    assert output.shape == (3, 1)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
    )


def test_network_accepts_real_float64_and_returns_parameter_dtype():
    network = TSMixer(_params()).return_network()

    output = network(torch.ones(2, 1, 9, dtype=torch.float64))

    assert output.dtype == network.head.weight.dtype == torch.float32


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"sequence_length": 8}, "odd"),
        ({"ff_dim": 0}, "positive integer"),
        ({"ff_dim": True}, "positive integer"),
        ({"n_blocks": 0}, "positive integer"),
        ({"dropout": -0.1}, "at least"),
        ({"dropout": 1.0}, "less than 1"),
        ({"activation": "swish"}, "relu.*gelu"),
        ({"activation": None}, "relu.*gelu"),
        ({"weight_decay": -1.0}, "at least"),
    ],
)
def test_invalid_configuration_fails_at_construction(params, message):
    with pytest.raises(ValueError, match=message):
        TSMixer(_params(**params))


@pytest.mark.parametrize(
    "params",
    [
        {"sequence_length": 0},
        {"sequence_length": 8},
        {"sequence_length": 9, "ff_dim": True},
        {"sequence_length": 9, "n_blocks": 0},
        {"sequence_length": 9, "dropout": float("nan")},
        {"sequence_length": 9, "dropout": 1.0},
        {"sequence_length": 9, "activation": object()},
    ],
)
def test_network_constructor_enforces_its_public_contract(params):
    with pytest.raises(ValueError):
        TSMixerNetwork(**params)


@pytest.mark.parametrize(
    "params",
    [
        {"sequence_length": 0},
        {"sequence_length": 9, "channels": 0},
        {"sequence_length": 9, "ff_dim": True},
        {"sequence_length": 9, "dropout": float("inf")},
        {"sequence_length": 9, "dropout": 1.0},
        {"sequence_length": 9, "activation": "tanh"},
    ],
)
def test_block_constructor_enforces_its_public_contract(params):
    with pytest.raises(ValueError):
        TSMixerBlock(**params)


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
    network = TSMixer(_params()).return_network()

    with pytest.raises((TypeError, ValueError)):
        network(inputs)


def test_network_rejects_an_input_on_a_different_device():
    network = TSMixer(_params()).return_network().to("meta")

    with pytest.raises(ValueError, match="input is on cpu; model is on meta"):
        network(torch.ones(1, 1, 9))


def test_network_rejects_non_finite_internal_output():
    network = TSMixer(_params()).return_network()
    with torch.no_grad():
        network.head.weight.fill_(float("inf"))

    with pytest.raises(RuntimeError, match="non-finite output"):
        network(torch.ones(1, 1, 9))


def test_one_epoch_train_infer_smoke_preserves_every_row():
    mains, targets = _chunks()
    model = TSMixer(_params(appliance_params={}))

    model.partial_fit(mains, targets)
    result = model.disaggregate_chunk(mains)[0]

    assert result.index.equals(mains[0].index)
    assert result.columns.tolist() == ["fridge"]
    assert result["fridge"].dtype == np.float32
    assert np.isfinite(result["fridge"]).all()


def test_checkpoint_roundtrip_restores_architecture_and_predictions(tmp_path):
    mains, targets = _chunks()
    model = TSMixer(
        _params(
            appliance_params={},
            save_model_path=str(tmp_path),
            ff_dim=6,
            activation="gelu",
        )
    )
    model.partial_fit(mains, targets)
    expected = model.disaggregate_chunk(mains)[0]

    loaded = TSMixer(
        _params(
            appliance_params={},
            pretrained_model_path=str(tmp_path),
            ff_dim=4,
            activation="relu",
        )
    )
    actual = loaded.disaggregate_chunk(mains)[0]

    assert loaded.ff_dim == 6
    assert loaded.activation == "gelu"
    assert np.allclose(actual, expected)


def test_public_export_catalog_defaults_and_parameter_count_are_stable():
    import nilmtk_contrib.torch as torch_models

    default = TSMixer({"device": "cpu"})
    network = default.return_network()
    entry = model_catalog_by_module()["nilmtk_contrib.torch.tsmixer"]

    assert torch_models.TSMixer is TSMixer
    assert default.sequence_length == 299
    assert default.batch_size == 128
    assert default.ff_dim == 64
    assert default.n_blocks == 2
    assert default.activation == "relu"
    assert default.weight_decay == pytest.approx(1e-4)
    assert sum(parameter.numel() for parameter in network.parameters()) == 182478
    assert entry.class_name == "TSMixer"
    assert entry.exported_from == "nilmtk_contrib.torch"


def test_none_uses_documented_defaults():
    model = TSMixer()

    assert model.sequence_length == 299
    assert model.ff_dim == 64
