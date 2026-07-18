import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
metadata_module = pytest.importorskip("nilmtk_contrib.metadata")
engine_module = pytest.importorskip("nilmtk_contrib.torch._seq2point")
nilmmoe_module = pytest.importorskip("nilmtk_contrib.torch.nilmmoe")
model_catalog_by_module = metadata_module.model_catalog_by_module
SequenceToPointTorchDisaggregator = engine_module.SequenceToPointTorchDisaggregator
EXPERT_NAMES = nilmmoe_module.EXPERT_NAMES
GATE_FEATURES = nilmmoe_module.GATE_FEATURES
NILMMoE = nilmmoe_module.NILMMoE
NILMMoEGate = nilmmoe_module.NILMMoEGate
NILMMoENetwork = nilmmoe_module.NILMMoENetwork


def _params(**overrides):
    return {
        "device": "cpu",
        "sequence_length": 25,
        "n_epochs": 1,
        "batch_size": 5,
        "seed": 17,
        "mains_mean": 100.0,
        "mains_std": 40.0,
        "moving_average": 25,
        "gate_hidden_dim": 8,
        "gate_dropout": 0.0,
        "gate_temperature": 1.0,
        "expert_dropout": 0.0,
        "load_balance_weight": 0.01,
    } | overrides


def _chunks(length=50):
    values = np.arange(length, dtype=np.float32)
    index = pd.date_range("2026-01-01", periods=length, freq="1min", tz="UTC")
    appliance = 25.0 + 30.0 * ((values // 4) % 2)
    mains = 70.0 + appliance + 3.0 * np.sin(values / 2)
    return (
        [pd.DataFrame({"power": mains}, index=index)],
        [("fridge", [pd.DataFrame({"power": appliance}, index=index)])],
    )


def test_nilmmoe_is_an_architecture_only_engine_consumer():
    source = Path(inspect.getfile(NILMMoE)).read_text()

    assert issubclass(NILMMoE, SequenceToPointTorchDisaggregator)
    assert len(source.splitlines()) < 300
    for method in ("partial_fit", "disaggregate_chunk", "save_model", "load_model"):
        assert f"def {method}(" not in source


def test_experts_are_complementary_reviewed_networks_in_stable_order():
    from nilmtk_contrib.torch.dlinear import DLinearNetwork
    from nilmtk_contrib.torch.moderntcn import ModernTCNNetwork
    from nilmtk_contrib.torch.timesnet import TimesNetNetwork

    network = NILMMoE(_params()).return_network()

    assert tuple(network.experts) == EXPERT_NAMES == (
        "DLinear",
        "ModernTCN",
        "TimesNet",
    )
    assert isinstance(network.experts["DLinear"], DLinearNetwork)
    assert isinstance(network.experts["ModernTCN"], ModernTCNNetwork)
    assert isinstance(network.experts["TimesNet"], TimesNetNetwork)


def test_gate_summary_features_are_explicit_and_numerically_correct():
    gate = NILMMoEGate(25, hidden_dim=4, dropout=0.0)
    inputs = torch.arange(25, dtype=torch.float32).reshape(1, 1, 25)

    features = gate.summary_features(inputs)

    assert GATE_FEATURES == (
        "center",
        "mean",
        "standard_deviation",
        "minimum",
        "maximum",
        "mean_absolute_difference",
        "end_to_start_change",
    )
    assert features.shape == (1, len(GATE_FEATURES))
    assert features[0, 0].item() == 12.0
    assert features[0, 1].item() == 12.0
    assert features[0, 2].item() == pytest.approx(np.std(np.arange(25)))
    assert features[0, 3:].tolist() == pytest.approx([0.0, 24.0, 1.0, 24.0])


def test_gate_weights_are_finite_positive_and_sum_to_one():
    gate = NILMMoEGate(25, hidden_dim=4, dropout=0.0, temperature=0.5)

    weights = gate(torch.randn(6, 1, 25))

    assert weights.shape == (6, len(EXPERT_NAMES))
    assert torch.isfinite(weights).all()
    assert (weights > 0).all()
    assert torch.allclose(weights.sum(dim=1), torch.ones(6))


def test_forced_gate_selects_the_requested_expert():
    network = NILMMoE(_params()).return_network().eval()
    inputs = torch.randn(4, 1, 25)
    with torch.no_grad():
        network.gate.output_layer.weight.zero_()
        network.gate.output_layer.bias.copy_(torch.tensor([-30.0, 30.0, -30.0]))

    mixture, predictions, weights = network.training_outputs(inputs)

    assert torch.all(weights.argmax(dim=1) == 1)
    assert torch.allclose(mixture, predictions[:, :, 1], atol=1e-6)


def test_mixture_is_inside_the_expert_convex_hull():
    network = NILMMoE(_params()).return_network().eval()

    mixture, predictions, _ = network.training_outputs(torch.randn(5, 1, 25))

    assert torch.all(mixture >= predictions.amin(dim=-1) - 1e-6)
    assert torch.all(mixture <= predictions.amax(dim=-1) + 1e-6)


def test_training_objective_adds_exact_load_balance_penalty():
    model = NILMMoE(_params(load_balance_weight=2.0))
    network = model.return_network().eval()
    inputs = torch.randn(5, 25)
    targets = torch.randn(5, 1)
    prediction, _, weights = network.training_outputs(inputs.unsqueeze(1))
    expected = torch.nn.functional.mse_loss(prediction, targets)
    expected = expected + 2.0 * (
        len(EXPERT_NAMES) * weights.mean(dim=0).square().sum() - 1
    )

    actual = model.training_loss(network, inputs, targets, "fridge")

    assert torch.allclose(actual, expected)


def test_training_objective_backpropagates_through_gate_and_every_expert():
    model = NILMMoE(_params())
    network = model.return_network()

    loss = model.training_loss(
        network, torch.randn(5, 25), torch.randn(5, 1), "fridge"
    )
    loss.backward()

    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
    )


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"sequence_length": 23}, "at least 25"),
        ({"moving_average": 24}, "odd"),
        ({"moving_average": 27}, "must not exceed"),
        ({"gate_hidden_dim": True}, "positive integer"),
        ({"gate_dropout": 1.0}, "less than 1"),
        ({"gate_temperature": 0.0}, "greater than"),
        ({"expert_dropout": float("nan")}, "finite"),
        ({"expert_dropout": 1.0}, "less than 1"),
        ({"load_balance_weight": -0.1}, "at least"),
        ({"weight_decay": -1.0}, "at least"),
    ],
)
def test_invalid_configuration_fails_at_construction(params, message):
    with pytest.raises(ValueError, match=message):
        NILMMoE(_params(**params))


@pytest.mark.parametrize(
    "params",
    [
        {"sequence_length": 0},
        {"sequence_length": 23},
        {"sequence_length": 24},
        {"sequence_length": 25, "moving_average": 24},
        {"sequence_length": 25, "moving_average": 27},
        {"sequence_length": 25, "gate_hidden_dim": True},
        {"sequence_length": 25, "gate_dropout": 1.0},
        {"sequence_length": 25, "gate_temperature": 0.0},
        {"sequence_length": 25, "expert_dropout": 1.0},
    ],
)
def test_network_constructor_enforces_its_public_contract(params):
    with pytest.raises(ValueError):
        NILMMoENetwork(**params)


@pytest.mark.parametrize(
    "params",
    [
        {"sequence_length": 0},
        {"sequence_length": 25, "hidden_dim": True},
        {"sequence_length": 25, "dropout": 1.0},
        {"sequence_length": 25, "temperature": float("nan")},
        {"sequence_length": 25, "temperature": 0.0},
    ],
)
def test_gate_constructor_enforces_its_public_contract(params):
    with pytest.raises(ValueError):
        NILMMoEGate(**params)


@pytest.mark.parametrize(
    "inputs",
    [
        object(),
        torch.ones(2, 25),
        torch.ones(2, 1, 24),
        torch.ones(2, 1, 25, dtype=torch.int64),
        torch.full((2, 1, 25), float("inf")),
    ],
)
def test_gate_rejects_invalid_inputs(inputs):
    gate = NILMMoEGate(25)

    with pytest.raises((TypeError, ValueError)):
        gate(inputs)


def test_gate_rejects_an_input_on_a_different_device():
    gate = NILMMoEGate(25).to("meta")

    with pytest.raises(ValueError, match="input is on cpu; model is on meta"):
        gate(torch.ones(1, 1, 25))


@pytest.mark.parametrize(
    "inputs",
    [
        object(),
        torch.ones(2, 25),
        torch.ones(2, 1, 24),
        torch.ones(2, 1, 25, dtype=torch.int64),
        torch.full((2, 1, 25), float("nan")),
    ],
)
def test_network_rejects_invalid_inputs(inputs):
    network = NILMMoE(_params()).return_network()

    with pytest.raises((TypeError, ValueError)):
        network(inputs)


def test_network_rejects_an_input_on_a_different_device():
    network = NILMMoE(_params()).return_network().to("meta")

    with pytest.raises(ValueError, match="input is on cpu; model is on meta"):
        network(torch.ones(1, 1, 25))


def test_gate_rejects_non_finite_internal_weights():
    network = NILMMoE(_params()).return_network()
    with torch.no_grad():
        network.gate.output_layer.weight.fill_(float("inf"))

    with pytest.raises(RuntimeError, match="gate produced non-finite"):
        network(torch.ones(1, 1, 25))


def test_network_rejects_non_finite_expert_mixture():
    class NonFiniteExpert(torch.nn.Module):
        def forward(self, inputs):
            return inputs.new_full((len(inputs), 1), float("inf"))

    network = NILMMoE(_params()).return_network()
    network.experts["DLinear"] = NonFiniteExpert()

    with pytest.raises(RuntimeError, match="produced non-finite output"):
        network(torch.ones(1, 1, 25))


def test_training_loss_rejects_the_wrong_network_type():
    model = NILMMoE(_params())

    with pytest.raises(TypeError, match="NILMMoENetwork"):
        model.training_loss(torch.nn.Linear(25, 1), None, None, "fridge")


def test_one_epoch_train_infer_smoke_preserves_every_row():
    mains, targets = _chunks()
    model = NILMMoE(_params(appliance_params={}))

    model.partial_fit(mains, targets)
    result = model.disaggregate_chunk(mains)[0]

    assert result.index.equals(mains[0].index)
    assert result.columns.tolist() == ["fridge"]
    assert result["fridge"].dtype == np.float32
    assert np.isfinite(result["fridge"]).all()


def test_checkpoint_roundtrip_restores_gate_configuration_and_predictions(tmp_path):
    mains, targets = _chunks()
    model = NILMMoE(
        _params(
            appliance_params={},
            save_model_path=str(tmp_path),
            gate_hidden_dim=6,
            gate_temperature=0.75,
        )
    )
    model.partial_fit(mains, targets)
    expected = model.disaggregate_chunk(mains)[0]

    loaded = NILMMoE(
        _params(
            appliance_params={},
            pretrained_model_path=str(tmp_path),
            gate_hidden_dim=4,
            gate_temperature=1.0,
        )
    )
    actual = loaded.disaggregate_chunk(mains)[0]

    assert loaded.gate_hidden_dim == 6
    assert loaded.gate_temperature == pytest.approx(0.75)
    assert np.allclose(actual, expected)


def test_public_export_catalog_defaults_and_parameter_count_are_stable():
    import nilmtk_contrib.torch as torch_models

    default = NILMMoE({"device": "cpu"})
    network = default.return_network()
    entry = model_catalog_by_module()["nilmtk_contrib.torch.nilmmoe"]

    assert torch_models.NILMMoE is NILMMoE
    assert default.sequence_length == 299
    assert default.batch_size == 128
    assert default.moving_average == 25
    assert default.gate_hidden_dim == 32
    assert default.gate_temperature == pytest.approx(1.0)
    assert default.load_balance_weight == pytest.approx(0.01)
    assert default.weight_decay == pytest.approx(1e-4)
    assert sum(parameter.numel() for parameter in network.parameters()) == 330493
    assert entry.class_name == "NILMMoE"
    assert entry.exported_from == "nilmtk_contrib.torch"


def test_none_uses_documented_defaults():
    model = NILMMoE()

    assert model.sequence_length == 299
    assert model.gate_hidden_dim == 32
