import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
metadata_module = pytest.importorskip("nilmtk_contrib.metadata")
engine_module = pytest.importorskip("nilmtk_contrib.torch._seq2point")
feature_module = pytest.importorskip("nilmtk_contrib.torch.feature_mlp")
window_feature_module = pytest.importorskip("nilmtk_contrib.torch._window_features")
model_catalog_by_module = metadata_module.model_catalog_by_module
SequenceToPointTorchDisaggregator = engine_module.SequenceToPointTorchDisaggregator
FeatureMLP = feature_module.FeatureMLP
FeatureMLPNetwork = feature_module.FeatureMLPNetwork
PREDICTION_FEATURE_NAMES = window_feature_module.PREDICTION_FEATURE_NAMES
WindowFeatureExtractor = window_feature_module.WindowFeatureExtractor


def _params(**overrides):
    return {
        "device": "cpu",
        "sequence_length": 9,
        "hidden_dim": 16,
        "dropout": 0.0,
        "n_epochs": 1,
        "batch_size": 8,
        "seed": 17,
        "mains_mean": 100.0,
        "mains_std": 40.0,
    } | overrides


def _chunks(length=27):
    sample = np.arange(length, dtype=np.float32)
    index = pd.date_range("2026-01-01", periods=length, freq="15min", tz="UTC")
    appliance = 30.0 + 45.0 * ((sample // 3) % 2)
    mains = 65.0 + appliance + 4.0 * np.sin(sample * 2 * np.pi / 9)
    return (
        [pd.DataFrame({"power": mains}, index=index)],
        [("fridge", [pd.DataFrame({"power": appliance}, index=index)])],
    )


def _named(features):
    return dict(zip(PREDICTION_FEATURE_NAMES, features[0].tolist(), strict=True))


def test_feature_model_is_an_architecture_only_engine_consumer():
    source = Path(inspect.getfile(FeatureMLP)).read_text()

    assert issubclass(FeatureMLP, SequenceToPointTorchDisaggregator)
    assert len(source.splitlines()) < 130
    for method in ("partial_fit", "disaggregate_chunk", "save_model", "load_model"):
        assert f"def {method}(" not in source


def test_constant_window_has_exact_constant_and_zero_variation_features():
    extractor = WindowFeatureExtractor(9)

    actual = _named(extractor(torch.full((1, 1, 9), 5.0, dtype=torch.float64)))

    for name in ("center", "mean", "minimum", "maximum", "median"):
        assert actual[name] == 5.0
    assert actual["root_mean_square"] == pytest.approx(5.0)
    for name in (
        "standard_deviation",
        "interquartile_range",
        "mean_absolute_deviation",
        "mean_absolute_difference",
        "rms_difference",
        "maximum_absolute_difference",
        "end_to_start_change",
        "linear_slope",
        "lag_one_correlation",
        "quarter_window_lag_correlation",
        "low_frequency_power_fraction",
        "mid_frequency_power_fraction",
        "high_frequency_power_fraction",
        "spectral_centroid",
        "spectral_entropy",
    ):
        assert actual[name] == pytest.approx(0.0, abs=1e-12)


def test_ramp_features_have_analytic_slope_spread_and_correlations():
    extractor = WindowFeatureExtractor(9)
    ramp = torch.linspace(-1, 1, 9, dtype=torch.float64).reshape(1, 1, 9)

    actual = _named(extractor(ramp))

    assert actual["center"] == pytest.approx(0.0, abs=1e-12)
    assert actual["mean"] == pytest.approx(0.0, abs=1e-12)
    assert actual["minimum"] == -1.0
    assert actual["maximum"] == 1.0
    assert actual["median"] == 0.0
    assert actual["interquartile_range"] == pytest.approx(1.0, abs=1e-12)
    assert actual["mean_absolute_difference"] == pytest.approx(0.25, abs=1e-12)
    assert actual["end_to_start_change"] == 2.0
    assert actual["linear_slope"] == pytest.approx(1.0, abs=1e-12)
    assert actual["lag_one_correlation"] == pytest.approx(1.0, abs=1e-12)
    assert actual["quarter_window_lag_correlation"] == pytest.approx(1.0, abs=1e-12)
    assert sum(
        actual[name]
        for name in (
            "low_frequency_power_fraction",
            "mid_frequency_power_fraction",
            "high_frequency_power_fraction",
        )
    ) == pytest.approx(1.0, abs=1e-12)


def test_clean_high_frequency_signal_has_one_spectral_bin():
    time = torch.arange(9, dtype=torch.float64)
    signal = torch.sin(2 * math.pi * 4 * time / 9).reshape(1, 1, 9)

    actual = _named(WindowFeatureExtractor(9)(signal))

    assert actual["low_frequency_power_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert actual["mid_frequency_power_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert actual["high_frequency_power_fraction"] == pytest.approx(1.0, abs=1e-12)
    assert actual["spectral_centroid"] == pytest.approx(8 / 9, abs=1e-12)
    assert actual["spectral_entropy"] == pytest.approx(0.0, abs=1e-12)


def test_translation_leaves_shape_and_spectrum_features_unchanged():
    extractor = WindowFeatureExtractor(9)
    values = torch.tensor(
        [[[1.0, 4.0, 2.0, 8.0, 3.0, 7.0, 2.0, 5.0, 1.0]]],
        dtype=torch.float64,
    )
    shifted = extractor(values + 100.0)
    original = extractor(values)
    invariant = {
        "standard_deviation",
        "interquartile_range",
        "mean_absolute_deviation",
        "mean_absolute_difference",
        "rms_difference",
        "maximum_absolute_difference",
        "end_to_start_change",
        "linear_slope",
        "lag_one_correlation",
        "quarter_window_lag_correlation",
        "low_frequency_power_fraction",
        "mid_frequency_power_fraction",
        "high_frequency_power_fraction",
        "spectral_centroid",
        "spectral_entropy",
    }
    positions = [PREDICTION_FEATURE_NAMES.index(name) for name in invariant]

    assert torch.allclose(shifted[:, positions], original[:, positions], atol=1e-12)


def test_feature_bank_is_differentiable_with_finite_input_gradients():
    generator = torch.Generator().manual_seed(23)
    inputs = torch.randn(
        (4, 1, 9),
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )

    features = WindowFeatureExtractor(9)(inputs)
    features.square().mean().backward()

    assert features.shape == (4, len(PREDICTION_FEATURE_NAMES))
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_network_is_small_and_backpropagates_through_every_parameter():
    network = FeatureMLPNetwork(9, hidden_dim=16, dropout=0.0)
    inputs = torch.randn(6, 1, 9)

    output = network(inputs)
    output.square().mean().backward()

    assert output.shape == (6, 1)
    assert network.feature_names == PREDICTION_FEATURE_NAMES
    assert sum(parameter.numel() for parameter in network.parameters()) == 411
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
    )


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"sequence_length": 8}, "odd"),
        ({"sequence_length": 7}, "at least 9"),
        ({"sequence_length": 9, "hidden_dim": True}, "positive integer"),
        ({"sequence_length": 9, "dropout": 1.0}, "less than 1"),
    ],
)
def test_network_constructor_enforces_its_public_contract(params, message):
    with pytest.raises(ValueError, match=message):
        FeatureMLPNetwork(**params)


def test_network_and_feature_bank_fail_closed_on_non_finite_internal_results():
    extractor = WindowFeatureExtractor(9)
    extractor.feature_names = ("wrong",)
    with pytest.raises(RuntimeError, match="definition and names"):
        extractor(torch.ones(1, 1, 9))

    overflowing = torch.full(
        (1, 1, 9), torch.finfo(torch.float32).max, dtype=torch.float32
    )
    overflowing[:, :, 0] = 0
    with pytest.raises(RuntimeError, match="non-finite features"):
        WindowFeatureExtractor(9)(overflowing)

    network = FeatureMLPNetwork(9, dropout=0.0)
    with torch.no_grad():
        network.regressor[-1].weight.fill_(float("inf"))
    with pytest.raises(RuntimeError, match="non-finite output"):
        network(torch.ones(1, 1, 9))


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"sequence_length": 7}, "at least 9"),
        ({"sequence_length": 8}, "odd"),
        ({"hidden_dim": True}, "positive integer"),
        ({"dropout": 1.0}, "less than 1"),
        ({"dropout": -0.1}, "at least"),
    ],
)
def test_invalid_model_configuration_fails_at_construction(params, message):
    with pytest.raises(ValueError, match=message):
        FeatureMLP(_params(**params))


@pytest.mark.parametrize(
    "inputs",
    [
        object(),
        torch.ones(2, 9),
        torch.ones(2, 1, 8),
        torch.ones(2, 1, 9, dtype=torch.int64),
        torch.ones(2, 1, 9, dtype=torch.float16),
        torch.full((2, 1, 9), float("nan")),
    ],
)
def test_feature_extractor_rejects_invalid_inputs(inputs):
    with pytest.raises((TypeError, ValueError)):
        WindowFeatureExtractor(9)(inputs)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"sequence_length": 8}, "odd"),
        ({"sequence_length": 7}, "at least 9"),
        ({"sequence_length": 9, "epsilon": 0.0}, "positive finite"),
        ({"sequence_length": 9, "epsilon": True}, "positive finite"),
    ],
)
def test_feature_extractor_validates_its_configuration(params, message):
    with pytest.raises(ValueError, match=message):
        WindowFeatureExtractor(**params)


def test_feature_implementation_has_no_external_feature_dependency():
    source = Path(inspect.getfile(WindowFeatureExtractor)).read_text()

    for dependency in ("numpy", "pandas", "tsfresh", "tsfel", "scipy", "sklearn"):
        assert dependency not in source


def test_one_epoch_train_infer_smoke_preserves_every_row():
    mains, targets = _chunks()
    model = FeatureMLP(_params(appliance_params={}))

    model.partial_fit(mains, targets)
    result = model.disaggregate_chunk(mains)[0]

    assert result.index.equals(mains[0].index)
    assert result.columns.tolist() == ["fridge"]
    assert result["fridge"].dtype == np.float32
    assert np.isfinite(result["fridge"]).all()


def test_checkpoint_roundtrip_restores_architecture_and_predictions(tmp_path):
    mains, targets = _chunks()
    model = FeatureMLP(
        _params(
            appliance_params={},
            save_model_path=str(tmp_path),
            hidden_dim=12,
            dropout=0.0,
        )
    )
    model.partial_fit(mains, targets)
    expected = model.disaggregate_chunk(mains)[0]

    loaded = FeatureMLP(
        _params(
            appliance_params={},
            pretrained_model_path=str(tmp_path),
            hidden_dim=4,
            dropout=0.5,
        )
    )
    actual = loaded.disaggregate_chunk(mains)[0]

    assert loaded.hidden_dim == 12
    assert loaded.dropout == 0.0
    assert np.allclose(actual, expected)


def test_public_export_catalog_defaults_and_feature_names_are_complete():
    import nilmtk_contrib.torch as torch_models

    default = FeatureMLP({"device": "cpu"})
    entry = model_catalog_by_module()["nilmtk_contrib.torch.feature_mlp"]

    assert torch_models.FeatureMLP is FeatureMLP
    assert default.sequence_length == 299
    assert default.batch_size == 128
    assert default.hidden_dim == 32
    assert default.weight_decay == pytest.approx(1e-4)
    assert len(PREDICTION_FEATURE_NAMES) == len(set(PREDICTION_FEATURE_NAMES)) == 21
    assert entry.class_name == "FeatureMLP"
    assert entry.exported_from == "nilmtk_contrib.torch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_and_cuda_feature_extraction_agree():
    inputs = torch.randn(5, 1, 9)

    cpu = WindowFeatureExtractor(9)(inputs)
    cuda = WindowFeatureExtractor(9).cuda()(inputs.cuda()).cpu()

    assert torch.allclose(cpu, cuda, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="model is on cpu"):
        FeatureMLPNetwork(9)(inputs.cuda())
