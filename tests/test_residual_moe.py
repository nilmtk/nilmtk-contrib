import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
metadata_module = pytest.importorskip("nilmtk_contrib.metadata")
engine_module = pytest.importorskip("nilmtk_contrib.torch._seq2point")
residual_module = pytest.importorskip("nilmtk_contrib.torch.residual_moe")
model_catalog_by_module = metadata_module.model_catalog_by_module
SequenceToPointTorchDisaggregator = engine_module.SequenceToPointTorchDisaggregator
ANCHOR_NAME = residual_module.ANCHOR_NAME
SPECIALIST_NAMES = residual_module.SPECIALIST_NAMES
ResidualMoE = residual_module.ResidualMoE
ResidualMoENetwork = residual_module.ResidualMoENetwork
ResidualMoEOutputs = residual_module.ResidualMoEOutputs
ResidualRouter = residual_module.ResidualRouter


def _params(**overrides):
    return {
        "device": "cpu",
        "sequence_length": 25,
        "n_epochs": 1,
        "batch_size": 5,
        "seed": 17,
        "mains_mean": 100.0,
        "mains_std": 40.0,
        "gate_hidden_dim": 8,
        "gate_dropout": 0.0,
        "residual_limit": 0.25,
        "expert_dropout": 0.0,
        "auxiliary_weight": 0.1,
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


def test_residual_moe_is_an_architecture_only_engine_consumer():
    source = Path(inspect.getfile(ResidualMoE)).read_text()

    assert issubclass(ResidualMoE, SequenceToPointTorchDisaggregator)
    assert len(source.splitlines()) < 300
    for method in ("partial_fit", "disaggregate_chunk", "save_model", "load_model"):
        assert f"def {method}(" not in source


def test_anchor_and_specialists_are_strong_reviewed_networks_in_stable_order():
    from nilmtk_contrib.torch.moderntcn import ModernTCNNetwork
    from nilmtk_contrib.torch.patchtst import PatchTSTNetwork
    from nilmtk_contrib.torch.timesnet import TimesNetNetwork

    network = ResidualMoE(_params()).return_network()

    assert ANCHOR_NAME == "TimesNet"
    assert SPECIALIST_NAMES == ("PatchTST", "ModernTCN")
    assert isinstance(network.anchor, TimesNetNetwork)
    assert tuple(network.specialists) == SPECIALIST_NAMES
    assert isinstance(network.specialists["PatchTST"], PatchTSTNetwork)
    assert isinstance(network.specialists["ModernTCN"], ModernTCNNetwork)


def test_router_summary_features_are_shared_and_numerically_correct():
    from nilmtk_contrib.torch._window_features import WINDOW_FEATURE_NAMES

    router = ResidualRouter(25, hidden_dim=4, dropout=0.0)
    inputs = torch.arange(25, dtype=torch.float32).reshape(1, 1, 25)

    features = router.summary_features(inputs)

    assert WINDOW_FEATURE_NAMES == (
        "center",
        "mean",
        "standard_deviation",
        "minimum",
        "maximum",
        "mean_absolute_difference",
        "end_to_start_change",
    )
    assert features.shape == (1, len(WINDOW_FEATURE_NAMES))
    assert features[0, 0].item() == 12.0
    assert features[0, 1].item() == 12.0
    assert features[0, 2].item() == pytest.approx(np.std(np.arange(25)))
    assert features[0, 3:].tolist() == pytest.approx([0.0, 24.0, 1.0, 24.0])


def test_training_outputs_are_inspectable_normalized_and_bounded():
    network = ResidualMoE(_params()).return_network().eval()

    outputs = network.training_outputs(torch.randn(6, 1, 25))

    assert isinstance(outputs, ResidualMoEOutputs)
    assert outputs.prediction.shape == (6, 1)
    assert outputs.anchor_prediction.shape == (6, 1)
    assert outputs.specialist_predictions.shape == (6, 1, 2)
    assert outputs.specialist_weights.shape == (6, 2)
    assert outputs.correction_amplitude.shape == (6, 1)
    assert torch.isfinite(torch.cat(tuple(output.flatten() for output in outputs))).all()
    assert (outputs.specialist_weights > 0).all()
    assert torch.allclose(outputs.specialist_weights.sum(dim=1), torch.ones(6))
    assert torch.all(outputs.correction_amplitude.abs() <= 0.25)


def test_zero_initialized_router_starts_exactly_at_the_anchor():
    network = ResidualMoE(_params()).return_network().eval()

    outputs = network.training_outputs(torch.randn(5, 1, 25))

    assert torch.equal(outputs.correction_amplitude, torch.zeros(5, 1))
    assert torch.equal(outputs.prediction, outputs.anchor_prediction)


def test_forced_router_applies_the_selected_signed_residual():
    network = ResidualMoE(_params(residual_limit=0.4)).return_network().eval()
    with torch.no_grad():
        network.router.weight_layer.weight.zero_()
        network.router.weight_layer.bias.copy_(torch.tensor([30.0, -30.0]))
        network.router.amplitude_layer.weight.zero_()
        network.router.amplitude_layer.bias.fill_(-2.0)

    outputs = network.training_outputs(torch.randn(4, 1, 25))
    selected = outputs.specialist_predictions[:, :, 0]
    expected = outputs.anchor_prediction + outputs.correction_amplitude * (
        selected - outputs.anchor_prediction
    )

    assert torch.all(outputs.specialist_weights.argmax(dim=1) == 0)
    assert (outputs.correction_amplitude < 0).all()
    assert torch.allclose(outputs.prediction, expected, atol=1e-6)


def test_correction_cannot_exceed_the_configured_fraction_of_expert_disagreement():
    network = ResidualMoE(_params(residual_limit=0.2)).return_network().eval()
    with torch.no_grad():
        network.router.amplitude_layer.bias.fill_(30.0)

    outputs = network.training_outputs(torch.randn(5, 1, 25))
    disagreements = (
        outputs.specialist_predictions - outputs.anchor_prediction.unsqueeze(-1)
    ).abs()
    maximum_allowed = 0.2 * disagreements.amax(dim=-1)

    assert torch.all(
        (outputs.prediction - outputs.anchor_prediction).abs()
        <= maximum_allowed + 1e-6
    )


def test_training_objective_adds_the_exact_specialist_auxiliary_loss():
    model = ResidualMoE(_params(auxiliary_weight=2.0))
    network = model.return_network().eval()
    inputs = torch.randn(5, 25)
    targets = torch.randn(5, 1)
    outputs = network.training_outputs(inputs.unsqueeze(1))
    expected = torch.nn.functional.mse_loss(outputs.prediction, targets)
    expected = expected + 2.0 * torch.stack(
        tuple(
            torch.nn.functional.mse_loss(prediction, targets)
            for prediction in outputs.specialist_predictions.unbind(dim=-1)
        )
    ).mean()

    actual = model.training_loss(network, inputs, targets, "fridge")

    assert torch.allclose(actual, expected)


def test_training_objective_backpropagates_through_router_and_every_expert():
    model = ResidualMoE(_params())
    network = model.return_network()

    loss = model.training_loss(
        network, torch.randn(5, 25), torch.randn(5, 1), "fridge"
    )
    loss.backward()

    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
    )
    assert any(
        parameter.grad.abs().sum() > 0
        for parameter in network.specialists.parameters()
    )
    assert network.router.amplitude_layer.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"sequence_length": 23}, "at least 25"),
        ({"sequence_length": 24}, "odd"),
        ({"gate_hidden_dim": True}, "positive integer"),
        ({"gate_dropout": 1.0}, "less than 1"),
        ({"residual_limit": 0.0}, "greater than"),
        ({"residual_limit": 1.1}, "must not exceed"),
        ({"expert_dropout": float("nan")}, "finite"),
        ({"expert_dropout": 1.0}, "less than 1"),
        ({"auxiliary_weight": -0.1}, "at least"),
        ({"weight_decay": -1.0}, "at least"),
    ],
)
def test_invalid_configuration_fails_at_construction(params, message):
    with pytest.raises(ValueError, match=message):
        ResidualMoE(_params(**params))


@pytest.mark.parametrize(
    "params",
    [
        {"sequence_length": 0},
        {"sequence_length": 23},
        {"sequence_length": 24},
        {"sequence_length": 25, "gate_hidden_dim": True},
        {"sequence_length": 25, "gate_dropout": 1.0},
        {"sequence_length": 25, "residual_limit": 0.0},
        {"sequence_length": 25, "residual_limit": 1.1},
        {"sequence_length": 25, "expert_dropout": 1.0},
    ],
)
def test_network_constructor_enforces_its_public_contract(params):
    with pytest.raises(ValueError):
        ResidualMoENetwork(**params)


@pytest.mark.parametrize(
    "params",
    [
        {"sequence_length": 0},
        {"sequence_length": 1},
        {"sequence_length": 25, "hidden_dim": True},
        {"sequence_length": 25, "dropout": 1.0},
        {"sequence_length": 25, "residual_limit": float("nan")},
        {"sequence_length": 25, "residual_limit": 0.0},
        {"sequence_length": 25, "residual_limit": 1.1},
    ],
)
def test_router_constructor_enforces_its_public_contract(params):
    with pytest.raises(ValueError):
        ResidualRouter(**params)


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
def test_router_rejects_invalid_inputs(inputs):
    with pytest.raises((TypeError, ValueError)):
        ResidualRouter(25)(inputs)


def test_router_rejects_an_input_on_a_different_device():
    router = ResidualRouter(25).to("meta")

    with pytest.raises(ValueError, match="input is on cpu; model is on meta"):
        router(torch.ones(1, 1, 25))


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
    network = ResidualMoE(_params()).return_network()

    with pytest.raises((TypeError, ValueError)):
        network(inputs)


def test_network_rejects_an_input_on_a_different_device():
    network = ResidualMoE(_params()).return_network().to("meta")

    with pytest.raises(ValueError, match="input is on cpu; model is on meta"):
        network(torch.ones(1, 1, 25))


def test_router_rejects_non_finite_internal_weights():
    network = ResidualMoE(_params()).return_network()
    with torch.no_grad():
        network.router.weight_layer.weight.fill_(float("inf"))

    with pytest.raises(RuntimeError, match="router produced non-finite"):
        network(torch.ones(1, 1, 25))


def test_network_rejects_a_non_finite_specialist_prediction():
    class NonFiniteSpecialist(torch.nn.Module):
        def forward(self, inputs):
            return inputs.new_full((len(inputs), 1), float("inf"))

    network = ResidualMoE(_params()).return_network()
    network.specialists["PatchTST"] = NonFiniteSpecialist()

    with pytest.raises(RuntimeError, match="produced non-finite outputs"):
        network(torch.ones(1, 1, 25))


def test_training_loss_rejects_the_wrong_network_type():
    model = ResidualMoE(_params())

    with pytest.raises(TypeError, match="ResidualMoENetwork"):
        model.training_loss(torch.nn.Linear(25, 1), None, None, "fridge")


def test_one_epoch_train_infer_smoke_preserves_every_row():
    mains, targets = _chunks()
    model = ResidualMoE(_params(appliance_params={}))

    model.partial_fit(mains, targets)
    result = model.disaggregate_chunk(mains)[0]

    assert result.index.equals(mains[0].index)
    assert result.columns.tolist() == ["fridge"]
    assert result["fridge"].dtype == np.float32
    assert np.isfinite(result["fridge"]).all()


def test_checkpoint_roundtrip_restores_router_configuration_and_predictions(tmp_path):
    mains, targets = _chunks()
    model = ResidualMoE(
        _params(
            appliance_params={},
            save_model_path=str(tmp_path),
            gate_hidden_dim=6,
            residual_limit=0.4,
        )
    )
    model.partial_fit(mains, targets)
    expected = model.disaggregate_chunk(mains)[0]

    loaded = ResidualMoE(
        _params(
            appliance_params={},
            pretrained_model_path=str(tmp_path),
            gate_hidden_dim=4,
            residual_limit=0.2,
        )
    )
    actual = loaded.disaggregate_chunk(mains)[0]

    assert loaded.gate_hidden_dim == 6
    assert loaded.residual_limit == pytest.approx(0.4)
    assert np.allclose(actual, expected)


def test_public_export_catalog_defaults_and_parameter_count_are_stable():
    import nilmtk_contrib.torch as torch_models

    default = ResidualMoE({"device": "cpu"})
    network = default.return_network()
    entry = model_catalog_by_module()["nilmtk_contrib.torch.residual_moe"]

    assert torch_models.ResidualMoE is ResidualMoE
    assert default.sequence_length == 299
    assert default.batch_size == 128
    assert default.gate_hidden_dim == 32
    assert default.residual_limit == pytest.approx(0.25)
    assert default.auxiliary_weight == pytest.approx(0.1)
    assert default.weight_decay == pytest.approx(1e-4)
    assert sum(parameter.numel() for parameter in network.parameters()) == 440870
    assert entry.class_name == "ResidualMoE"
    assert entry.exported_from == "nilmtk_contrib.torch"


def test_none_uses_documented_defaults():
    model = ResidualMoE()

    assert model.sequence_length == 299
    assert model.gate_hidden_dim == 32
