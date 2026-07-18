import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
metadata_module = pytest.importorskip("nilmtk_contrib.metadata")
engine_module = pytest.importorskip("nilmtk_contrib.torch._seq2point")
sgn_module = pytest.importorskip("nilmtk_contrib.torch.sgn")
nn = torch.nn
model_catalog_by_module = metadata_module.model_catalog_by_module
SequenceToPointTorchDisaggregator = engine_module.SequenceToPointTorchDisaggregator
PAPER_FILTERS = sgn_module.PAPER_FILTERS
PAPER_KERNEL_SIZES = sgn_module.PAPER_KERNEL_SIZES
SGN = sgn_module.SGN
SGNNetwork = sgn_module.SGNNetwork
SGNTower = sgn_module.SGNTower


class _FixedTower(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, inputs):
        return self.values[: len(inputs)].reshape(-1, 1)


def _params(**overrides):
    return {
        "device": "cpu",
        "sequence_length": 39,
        "n_epochs": 1,
        "batch_size": 8,
        "seed": 23,
        "mains_mean": 100.0,
        "mains_std": 40.0,
        "hidden_dim": 8,
        "dropout": 0.0,
        "appliance_params": {"fridge": {"mean": 10.0, "std": 5.0}},
    } | overrides


def _chunks(length=78):
    time = np.arange(length, dtype=np.float32)
    appliance = 30.0 * ((time // 6) % 2)
    mains = 70.0 + appliance + 4.0 * np.sin(time / 3)
    index = pd.date_range("2026-01-01", periods=length, freq="1min", tz="UTC")
    return (
        [pd.DataFrame({"power": mains}, index=index)],
        [("fridge", [pd.DataFrame({"power": appliance}, index=index)])],
    )


def test_sgn_is_an_architecture_only_engine_consumer():
    source = Path(inspect.getfile(SGN)).read_text()

    assert issubclass(SGN, SequenceToPointTorchDisaggregator)
    assert len(source.splitlines()) < 320
    for method in ("partial_fit", "disaggregate_chunk", "save_model", "load_model"):
        assert f"def {method}(" not in source


def test_tower_uses_the_paper_kernel_and_filter_schedule():
    tower = SGNTower(sequence_length=39, hidden_dim=8)

    assert tuple(layer.kernel_size[0] for layer in tower.convolutions) == (
        PAPER_KERNEL_SIZES
    )
    assert tuple(layer.out_channels for layer in tower.convolutions) == PAPER_FILTERS
    assert tower(torch.ones(2, 1, 39)).shape == (2, 1)


def test_gate_operates_in_raw_power_despite_normalized_training_targets():
    network = SGNNetwork(sequence_length=39, hidden_dim=4)
    network.regression_tower = _FixedTower([1.0, 2.0])
    network.classification_tower = _FixedTower([0.0, math.log(1 / 3)])
    network.configure_target(mean=100.0, std=20.0)

    gated, regression, logits = network.training_outputs(torch.zeros(2, 1, 39))
    decoded = 100.0 + 20.0 * gated
    expected = torch.sigmoid(logits) * (100.0 + 20.0 * regression)

    assert torch.allclose(decoded, expected)
    assert decoded[:, 0].tolist() == pytest.approx([60.0, 35.0])


def test_network_requires_target_normalization_before_forward():
    network = SGNNetwork(sequence_length=39, hidden_dim=4)

    with pytest.raises(RuntimeError, match="not configured"):
        network(torch.ones(2, 1, 39))


@pytest.mark.parametrize(
    ("mean", "std"),
    [
        (float("nan"), 1.0),
        (0.0, 0.0),
        (0.0, -1.0),
        (True, 1.0),
    ],
)
def test_network_rejects_invalid_target_normalization(mean, std):
    network = SGNNetwork(sequence_length=39, hidden_dim=4)

    with pytest.raises(ValueError):
        network.configure_target(mean, std)


@pytest.mark.parametrize(
    "inputs",
    [
        object(),
        torch.ones(2, 39),
        torch.ones(2, 1, 37),
        torch.ones(2, 1, 39, dtype=torch.int64),
        torch.full((2, 1, 39), float("nan")),
    ],
)
def test_network_rejects_invalid_inputs(inputs):
    network = SGNNetwork(sequence_length=39, hidden_dim=4)
    network.configure_target(10.0, 5.0)

    with pytest.raises((TypeError, ValueError)):
        network(inputs)


def test_joint_loss_uses_raw_threshold_labels_and_both_subtasks():
    model = SGN(_params(on_power_threshold=15.0))
    network = model.return_network()
    network.regression_tower = _FixedTower([0.0, 2.0])
    network.classification_tower = _FixedTower([-2.0, 2.0])
    model.configure_network(network, "fridge")
    inputs = torch.ones(2, model.sequence_length)
    targets = torch.tensor([[0.0], [2.0]])

    actual = model.training_loss(network, inputs, targets, "fridge")
    gated, _, logits = network.training_outputs(inputs.unsqueeze(1))
    expected = nn.functional.mse_loss(gated, targets)
    expected += nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([[0.0], [1.0]])
    )
    actual.backward()

    assert torch.allclose(actual, expected)
    assert network.regression_tower.values.grad is not None
    assert network.classification_tower.values.grad is not None


def test_threshold_precedence_is_explicit_then_metadata_then_paper_default():
    explicit = SGN(_params(on_power_threshold=20.0))
    metadata = SGN(
        _params(appliance_params={"fridge": {"mean": 10, "std": 5, "threshold": 60}})
    )
    default = SGN(_params())

    assert explicit._threshold_for("fridge") == 20.0
    assert metadata._threshold_for("fridge") == 60.0
    assert default._threshold_for("fridge") == 15.0


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"sequence_length": 33}, "at least"),
        ({"hidden_dim": 0}, "positive integer"),
        ({"dropout": 1.0}, "less than 1"),
        ({"classification_weight": 0.0}, "greater than"),
        ({"on_power_threshold": -1.0}, "at least"),
    ],
)
def test_invalid_configuration_fails_at_construction(params, message):
    with pytest.raises(ValueError, match=message):
        SGN(_params(**params))


def test_configuration_rejects_an_incompatible_network():
    model = SGN(_params())

    with pytest.raises(TypeError, match="SGNNetwork"):
        model.configure_network(nn.Linear(2, 1), "fridge")


def test_one_epoch_train_infer_smoke_preserves_rows_and_trains_both_towers():
    mains, targets = _chunks()
    model = SGN(_params(appliance_params={}))

    model.partial_fit(mains, targets)
    result = model.disaggregate_chunk(mains)[0]
    network = model.models["fridge"]

    assert result.index.equals(mains[0].index)
    assert result.columns.tolist() == ["fridge"]
    assert result["fridge"].dtype == np.float32
    assert np.isfinite(result["fridge"]).all()
    assert torch.isfinite(network.target_mean)
    assert torch.isfinite(network.target_std)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
    )


def test_checkpoint_roundtrip_restores_architecture_and_predictions(tmp_path):
    mains, targets = _chunks()
    model = SGN(
        _params(
            appliance_params={},
            save_model_path=str(tmp_path),
            hidden_dim=10,
        )
    )
    model.partial_fit(mains, targets)
    expected = model.disaggregate_chunk(mains)[0]

    loaded = SGN(
        _params(
            appliance_params={},
            pretrained_model_path=str(tmp_path),
            hidden_dim=4,
        )
    )
    actual = loaded.disaggregate_chunk(mains)[0]

    assert loaded.hidden_dim == 10
    assert np.allclose(actual, expected)


def test_public_export_catalog_and_paper_defaults_are_complete():
    import nilmtk_contrib.torch as torch_models

    default = SGN({"device": "cpu"})
    entry = model_catalog_by_module()["nilmtk_contrib.torch.sgn"]

    assert torch_models.SGN is SGN
    assert default.sequence_length == 299
    assert default.batch_size == 16
    assert default.learning_rate == pytest.approx(1e-4)
    assert default.hidden_dim == 1024
    assert default.classification_weight == 1.0
    assert default.on_power_threshold is None
    assert entry.class_name == "SGN"
    assert entry.exported_from == "nilmtk_contrib.torch"
