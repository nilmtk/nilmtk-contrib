from collections import OrderedDict
from copy import deepcopy
import io
import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
patchtst = pytest.importorskip("nilmtk_contrib.torch.patchtst")
PatchTST = patchtst.PatchTST
PatchTSTNetwork = patchtst.PatchTSTNetwork


def _tiny_params(**overrides):
    params = {
        "device": "cpu",
        "sequence_length": 9,
        "n_epochs": 1,
        "batch_size": 3,
        "mains_mean": 100.0,
        "mains_std": 50.0,
        "patch_length": 3,
        "patch_stride": 2,
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 16,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "validation_fraction": 0.2,
        "seed": 23,
    }
    params.update(overrides)
    return params


def _processed_training(samples=7):
    base = np.linspace(-1.0, 1.0, samples * 9, dtype=np.float32).reshape(samples, 9)
    target = (base[:, 4] * 0.25 + 0.1).reshape(-1, 1).astype(np.float32)
    return [pd.DataFrame(base)], [("fridge", [pd.DataFrame(target)])]


def _raw_training(samples=11, constant_target=False):
    index = pd.date_range("2024-01-01", periods=samples, freq="min")
    time = np.arange(samples, dtype=np.float32)
    target = np.full(samples, 50.0, dtype=np.float32)
    if not constant_target:
        target = 50.0 + 10.0 * ((time // 3) % 2)
    mains = target + 25.0 + np.sin(time)
    return (
        [pd.DataFrame({"power": mains}, index=index)],
        [("fridge", [pd.DataFrame({"power": target}, index=index)])],
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"patch_length": 10}, "patch_length"),
        ({"patch_length": 3, "patch_stride": 4}, "patch_stride"),
        ({"d_model": 7, "n_heads": 2}, "divisible"),
        ({"dropout": 1.0}, "dropout"),
        ({"validation_fraction": 1.0}, "validation_fraction"),
        ({"validation_strategy": "middle"}, "validation_strategy"),
        ({"validation_strategy": []}, "validation_strategy"),
    ],
)
def test_adapter_rejects_invalid_geometry_and_training_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PatchTST(_tiny_params(**kwargs))


def test_adapter_requires_centered_odd_length():
    with pytest.raises(ValueError, match="odd"):
        PatchTST(_tiny_params(sequence_length=10))


def test_network_covers_awkward_length_and_preserves_batch_shape():
    network = PatchTSTNetwork(
        sequence_length=19,
        patch_length=6,
        patch_stride=4,
        d_model=12,
        n_heads=2,
        n_layers=1,
        d_ff=24,
        dropout=0.0,
    )

    assert network.num_patches == 5
    assert network(torch.randn(1, 1, 19)).shape == (1, 1)
    assert network(torch.randn(4, 1, 19)).shape == (4, 1)


@pytest.mark.parametrize("shape", [(2, 19), (2, 2, 19), (2, 1, 18)])
def test_network_rejects_ambiguous_input_shapes(shape):
    network = PatchTSTNetwork(19, patch_length=6, patch_stride=4)
    with pytest.raises(ValueError, match="expects input shape"):
        network(torch.randn(*shape))


def test_network_has_finite_gradients_for_flat_and_varying_batches():
    network = PatchTSTNetwork(
        9,
        patch_length=3,
        patch_stride=2,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
    )
    inputs = torch.stack([torch.zeros(1, 9), torch.linspace(-1, 1, 9).reshape(1, 9)])
    loss = network(inputs).square().mean()
    loss.backward()

    gradients = [parameter.grad for parameter in network.parameters()]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_network_validates_tensor_dtype_device_and_finiteness():
    network = PatchTSTNetwork(
        9,
        patch_length=3,
        patch_stride=2,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
    )

    with pytest.raises(TypeError, match="torch.Tensor"):
        network(np.zeros((1, 1, 9), dtype=np.float32))
    with pytest.raises(TypeError, match="real floating"):
        network(torch.zeros((1, 1, 9), dtype=torch.int64))
    with pytest.raises(ValueError, match="finite"):
        network(torch.full((1, 1, 9), float("nan")))
    with pytest.raises(ValueError, match="model is on cpu"):
        network(torch.empty((1, 1, 9), device="meta"))

    output = network(torch.zeros((2, 1, 9), dtype=torch.float64))
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_network_rejects_nonfinite_output():
    network = PatchTSTNetwork(
        9,
        patch_length=3,
        patch_stride=2,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
    )
    with torch.no_grad():
        network.head[-1].bias.fill_(float("inf"))

    with pytest.raises(RuntimeError, match="non-finite output"):
        network(torch.zeros((1, 1, 9)))


def test_training_rejects_misaligned_chunk_counts_lengths_and_indexes():
    model = PatchTST(_tiny_params())
    mains, appliances = _raw_training()

    with pytest.raises(ValueError, match="chunks but mains"):
        model.partial_fit(mains * 2, appliances)

    shorter = appliances[0][1][0].iloc[:-1]
    with pytest.raises(ValueError, match="same number of samples"):
        model.partial_fit(mains, [("fridge", [shorter])])

    shifted = appliances[0][1][0].copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="aligned indexes"):
        model.partial_fit(mains, [("fridge", [shifted])])


def test_training_rejects_multichannel_and_nonfinite_power_data():
    model = PatchTST(_tiny_params())
    _, appliances = _raw_training()
    two_channels = pd.DataFrame({"a": [1.0] * 11, "b": [2.0] * 11})
    with pytest.raises(ValueError, match="one power column"):
        model.partial_fit([two_channels], appliances)

    for bad in (np.nan, np.inf, -np.inf):
        mains, appliances = _raw_training()
        appliances[0][1][0].iloc[3, 0] = bad
        with pytest.raises(ValueError, match="finite"):
            model.partial_fit(mains, appliances)


def test_flat_target_batch_one_train_to_infer_is_finite_and_indexed():
    model = PatchTST(_tiny_params(batch_size=1))
    mains, appliances = _raw_training(samples=7, constant_target=True)
    model.partial_fit(mains, appliances)
    predictions = model.disaggregate_chunk(mains)

    assert model.appliance_params["fridge"] == {"mean": 50.0, "std": 100.0}
    assert predictions[0].index.equals(mains[0].index)
    assert predictions[0].shape == (7, 1)
    assert np.isfinite(predictions[0].to_numpy()).all()
    assert (predictions[0].to_numpy() >= 0).all()


def test_single_sample_training_uses_safe_no_validation_path():
    model = PatchTST(
        _tiny_params(
            batch_size=4,
            appliance_params={"fridge": {"mean": 0.0, "std": 1.0}},
        )
    )
    mains, appliances = _processed_training(samples=1)
    model.partial_fit(mains, appliances, do_preprocessing=False)
    result = model.disaggregate_chunk(mains, do_preprocessing=False)[0]

    metadata = model.last_split_metadata["fridge"]
    assert metadata.train_size == 1
    assert metadata.validation_enabled is False
    assert result.shape == (1, 1)
    assert np.isfinite(result.to_numpy()).all()


@pytest.mark.parametrize("batch_size", [1, 2])
def test_inference_handles_batch_one_and_partial_final_batch(batch_size):
    class ZeroPoint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))

        def forward(self, values):
            return self.bias.expand(len(values), 1)

    index = pd.date_range("2024-02-01", periods=5, freq="min")
    mains = pd.DataFrame({"power": np.full(5, 120.0)}, index=index)
    model = PatchTST(
        _tiny_params(
            batch_size=batch_size,
            appliance_params={"fridge": {"mean": 42.0, "std": 5.0}},
        )
    )
    result = model.disaggregate_chunk([mains], model={"fridge": ZeroPoint()})[0]

    assert result.index.equals(index)
    np.testing.assert_array_equal(result["fridge"], np.full(5, 42.0))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_inference_rejects_nonfinite_mains(bad):
    model = PatchTST(
        _tiny_params(appliance_params={"fridge": {"mean": 42.0, "std": 5.0}})
    )
    network = model.return_network()
    mains = pd.DataFrame({"power": np.full(5, 120.0)})
    mains.iloc[2, 0] = bad

    with pytest.raises(ValueError, match="finite"):
        model.disaggregate_chunk([mains], model={"fridge": network})


@pytest.mark.parametrize(
    "values",
    [
        np.full(5, 1e300, dtype=np.float64),
        np.asarray([10**1000] * 5, dtype=object),
    ],
)
def test_input_overflow_becomes_actionable_validation_without_warning(values):
    model = PatchTST(
        _tiny_params(appliance_params={"fridge": {"mean": 42.0, "std": 5.0}})
    )
    mains = pd.DataFrame({"power": pd.Series(values, dtype=object)})

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="representable as float32"):
            model.disaggregate_chunk([mains], model={"fridge": model.return_network()})


def test_inference_rejects_nonfinite_model_output():
    class NonFinitePoint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(float("nan")))

        def forward(self, values):
            return self.bias.expand(len(values), 1)

    model = PatchTST(
        _tiny_params(appliance_params={"fridge": {"mean": 42.0, "std": 5.0}})
    )
    mains = pd.DataFrame({"power": np.full(5, 120.0)})

    with pytest.raises(FloatingPointError, match="non-finite"):
        model.disaggregate_chunk([mains], model={"fridge": NonFinitePoint()})


@pytest.mark.parametrize(
    "kind, error, message",
    [
        ("array", TypeError, "torch.Tensor"),
        ("shape", ValueError, "returned shape"),
        ("integer", TypeError, "real floating"),
        ("device", ValueError, "expected cpu"),
    ],
)
def test_inference_rejects_invalid_external_model_output(kind, error, message):
    class BadPoint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))

        def forward(self, values):
            if kind == "array":
                return np.zeros((len(values), 1), dtype=np.float32)
            if kind == "shape":
                return torch.zeros((len(values), 2), device=values.device)
            if kind == "integer":
                return torch.zeros(
                    (len(values), 1), dtype=torch.int64, device=values.device
                )
            return torch.empty((len(values), 1), device="meta")

    model = PatchTST(
        _tiny_params(appliance_params={"fridge": {"mean": 42.0, "std": 5.0}})
    )
    mains = pd.DataFrame({"power": np.full(5, 120.0)})

    with pytest.raises(error, match=message):
        model.disaggregate_chunk([mains], model={"fridge": BadPoint()})


def test_inference_supports_empty_raw_and_processed_chunks_without_model_call():
    class FailIfCalled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))

        def forward(self, values):
            raise AssertionError("empty inference must not call the model")

    model = PatchTST(
        _tiny_params(appliance_params={"fridge": {"mean": 42.0, "std": 5.0}})
    )
    network = {"fridge": FailIfCalled()}
    raw_index = pd.DatetimeIndex([], name="time")
    raw = pd.DataFrame({"power": pd.Series(index=raw_index, dtype=np.float32)})
    processed_index = pd.Index([], dtype=np.int64, name="window")
    processed = pd.DataFrame(
        np.empty((0, model.sequence_length), dtype=np.float32),
        index=processed_index,
    )

    raw_result = model.disaggregate_chunk([raw], model=network)[0]
    processed_result = model.disaggregate_chunk(
        [processed], model=network, do_preprocessing=False
    )[0]

    assert raw_result.index.equals(raw_index)
    assert processed_result.index.equals(processed_index)
    assert list(raw_result.columns) == ["fridge"]
    assert list(processed_result.columns) == ["fridge"]


def test_inference_rejects_finite_values_that_overflow_public_float32():
    class OnePoint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.ones(1))

        def forward(self, values):
            return self.bias.expand(len(values), 1)

    model = PatchTST(
        _tiny_params(appliance_params={"fridge": {"mean": 0.0, "std": 1e300}})
    )
    mains = pd.DataFrame({"power": np.full(5, 120.0)})

    with pytest.raises(ValueError, match="overflow float32"):
        model.disaggregate_chunk([mains], model={"fridge": OnePoint()})


def test_processed_shape_and_alignment_are_checked_before_training():
    model = PatchTST(
        _tiny_params(appliance_params={"fridge": {"mean": 0.0, "std": 1.0}})
    )
    mains, appliances = _processed_training(samples=5)
    bad_shape = pd.DataFrame(np.zeros((5, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        model.partial_fit([bad_shape], appliances, do_preprocessing=False)

    short_target = appliances[0][1][0].iloc[:-1]
    with pytest.raises(ValueError, match="aligned"):
        model.partial_fit(mains, [("fridge", [short_target])], do_preprocessing=False)

    shifted_target = appliances[0][1][0].copy()
    shifted_target.index = shifted_target.index + 1
    with pytest.raises(ValueError, match="aligned indexes"):
        model.partial_fit(mains, [("fridge", [shifted_target])], do_preprocessing=False)


def test_processed_training_requires_precomputed_appliance_statistics():
    model = PatchTST(_tiny_params())
    mains, appliances = _processed_training(samples=5)

    with pytest.raises(ValueError, match="appliance_params"):
        model.partial_fit(mains, appliances, do_preprocessing=False)

    assert model.appliance_params == {}
    assert model.models == {}


def test_zero_epoch_training_fails_without_installing_state():
    model = PatchTST(_tiny_params(n_epochs=0))
    mains, appliances = _raw_training(samples=5)

    with pytest.raises(ValueError, match="at least one epoch"):
        model.partial_fit(mains, appliances)

    assert model.appliance_params == {}
    assert model.models == {}


def test_nonfinite_optimizer_step_restores_existing_model_exactly():
    model = PatchTST(
        _tiny_params(
            batch_size=1,
            learning_rate=1e308,
            appliance_params={"fridge": {"mean": 0.0, "std": 1.0}},
        )
    )
    model.models["fridge"] = model.return_network()
    model.models["fridge"].eval()
    original_gradients = {}
    for name, parameter in model.models["fridge"].named_parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
        original_gradients[name] = parameter.grad.detach().clone()
    original_state = {
        name: value.detach().clone()
        for name, value in model.models["fridge"].state_dict().items()
    }
    mains, appliances = _processed_training(samples=5)

    with pytest.raises((FloatingPointError, RuntimeError), match="non-finite"):
        model.partial_fit(mains, appliances, do_preprocessing=False)

    restored_state = model.models["fridge"].state_dict()
    assert model.models["fridge"].training is False
    assert all(torch.isfinite(value).all() for value in restored_state.values())
    for name, value in original_state.items():
        torch.testing.assert_close(restored_state[name], value, rtol=0, atol=0)
    for name, parameter in model.models["fridge"].named_parameters():
        torch.testing.assert_close(
            parameter.grad, original_gradients[name], rtol=0, atol=0
        )


def test_failed_raw_training_rolls_back_new_statistics_and_models(monkeypatch):
    class NonFiniteNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.ones(1))

        def forward(self, values):
            return self.bias.expand(len(values), 1) * float("inf")

    model = PatchTST(_tiny_params())
    monkeypatch.setattr(model, "return_network", NonFiniteNetwork)
    mains, appliances = _raw_training(samples=7)

    with pytest.raises((FloatingPointError, RuntimeError), match="non-finite"):
        model.partial_fit(mains, appliances)

    assert model.appliance_params == {}
    assert model.models == {}
    assert model.last_split_metadata == {}


def test_second_appliance_failure_rolls_back_every_model_and_runtime(monkeypatch):
    class NonFiniteNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.ones(1))

        def forward(self, values):
            return self.bias.expand(len(values), 1) * float("inf")

    statistics = {
        "fridge": {"mean": 0.0, "std": 1.0},
        "kettle": {"mean": 0.0, "std": 1.0},
    }
    model = PatchTST(_tiny_params(appliance_params=statistics, dropout=0.0))
    model.models["fridge"] = model.return_network()
    model.models["fridge"].eval()
    for parameter in model.models["fridge"].parameters():
        parameter.grad = torch.full_like(parameter, 0.125)
    original_state = {
        name: value.detach().clone()
        for name, value in model.models["fridge"].state_dict().items()
    }
    original_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.models["fridge"].named_parameters()
    }
    model.last_split_metadata["previous"] = "sentinel"
    monkeypatch.setattr(model, "return_network", NonFiniteNetwork)
    mains, first = _processed_training(samples=7)
    _, second = _processed_training(samples=7)
    second = [("kettle", second[0][1])]

    with pytest.raises((FloatingPointError, RuntimeError), match="non-finite"):
        model.partial_fit(mains, first + second, do_preprocessing=False)

    assert list(model.models) == ["fridge"]
    assert model.models["fridge"].training is False
    assert model.appliance_params == statistics
    assert model.last_split_metadata == {"previous": "sentinel"}
    for name, value in original_state.items():
        torch.testing.assert_close(
            model.models["fridge"].state_dict()[name], value, rtol=0, atol=0
        )
    for name, parameter in model.models["fridge"].named_parameters():
        torch.testing.assert_close(
            parameter.grad, original_gradients[name], rtol=0, atol=0
        )


def test_checkpoint_failure_at_end_of_partial_fit_rolls_back_memory(
    tmp_path, monkeypatch
):
    model = PatchTST(
        _tiny_params(
            save_model_path=tmp_path,
            appliance_params={"fridge": {"mean": 0.0, "std": 1.0}},
        )
    )
    model.models["fridge"] = model.return_network()
    model.models["fridge"].eval()
    original_state = {
        name: value.detach().clone()
        for name, value in model.models["fridge"].state_dict().items()
    }
    mains, appliances = _processed_training(samples=7)

    def fail_save():
        raise OSError("simulated checkpoint failure")

    monkeypatch.setattr(model, "save_model", fail_save)
    with pytest.raises(OSError, match="checkpoint failure"):
        model.partial_fit(mains, appliances, do_preprocessing=False)

    assert model.models["fridge"].training is False
    for name, value in original_state.items():
        torch.testing.assert_close(
            model.models["fridge"].state_dict()[name], value, rtol=0, atol=0
        )


def test_cpu_training_is_deterministic_for_fixed_seed():
    mains, appliances = _processed_training(samples=7)
    params = _tiny_params(
        batch_size=2,
        dropout=0.1,
        appliance_params={"fridge": {"mean": 0.0, "std": 1.0}},
    )

    first = PatchTST(params)
    first.partial_fit(mains, appliances, do_preprocessing=False)
    first_prediction = first.disaggregate_chunk(mains, do_preprocessing=False)[0]

    second = PatchTST(params)
    second.partial_fit(mains, appliances, do_preprocessing=False)
    second_prediction = second.disaggregate_chunk(mains, do_preprocessing=False)[0]

    np.testing.assert_array_equal(first_prediction, second_prediction)
    for key, value in first.models["fridge"].state_dict().items():
        torch.testing.assert_close(value, second.models["fridge"].state_dict()[key])


def test_fixed_seed_isolated_from_other_live_instance_training():
    mains, appliances = _processed_training(samples=7)
    params = _tiny_params(
        batch_size=2,
        dropout=0.2,
        appliance_params={"fridge": {"mean": 0.0, "std": 1.0}},
    )
    first = PatchTST(params)
    first.models["fridge"] = first.return_network()
    second = PatchTST(params)
    second.models["fridge"] = deepcopy(first.models["fridge"])

    first.partial_fit(mains, appliances, do_preprocessing=False)
    second.partial_fit(mains, appliances, do_preprocessing=False)

    for name, value in first.models["fridge"].state_dict().items():
        torch.testing.assert_close(
            value,
            second.models["fridge"].state_dict()[name],
            rtol=0,
            atol=0,
        )


def test_fixed_seed_training_restores_process_torch_rng_state():
    mains, appliances = _processed_training(samples=7)
    model = PatchTST(
        _tiny_params(appliance_params={"fridge": {"mean": 0.0, "std": 1.0}})
    )
    before = torch.random.get_rng_state().clone()

    model.partial_fit(mains, appliances, do_preprocessing=False)

    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)


def test_checkpoint_roundtrip_restores_config_predictions_and_safe_filenames(tmp_path):
    appliance_name = "../../fridge"
    params = _tiny_params(
        save_model_path=tmp_path,
        appliance_params={appliance_name: {"mean": 42.0, "std": 5.0}},
    )
    model = PatchTST(params)
    model.models[appliance_name] = model.return_network()
    mains = pd.DataFrame({"power": np.linspace(80.0, 140.0, 5)})
    expected = model.disaggregate_chunk([mains])[0]
    model.save_model()

    files = sorted(path.name for path in tmp_path.iterdir())
    assert files[0] == "metadata.json"
    assert len(files) == 2
    assert files[1].startswith("patchtst-") and files[1].endswith(".pt")
    assert "fridge" not in files[1]

    restored = PatchTST(
        {
            "device": "cpu",
            "pretrained_model_path": tmp_path,
            "batch_size": 1,
        }
    )
    actual = restored.disaggregate_chunk([mains])[0]

    assert restored._model_config() == model._model_config()
    assert list(restored.models) == [appliance_name]
    np.testing.assert_array_equal(actual, expected)


def test_successful_checkpoint_update_publishes_one_complete_new_generation(tmp_path):
    model = PatchTST(
        _tiny_params(
            save_model_path=tmp_path,
            appliance_params={"fridge": {"mean": 42.0, "std": 5.0}},
        )
    )
    model.models["fridge"] = model.return_network()
    model.save_model()
    first_metadata = json.loads(
        (tmp_path / "metadata.json").read_text(encoding="utf-8")
    )
    first_weight = first_metadata["model_files"]["fridge"]

    with torch.no_grad():
        next(model.models["fridge"].parameters()).add_(1.0)
    model.save_model()
    second_metadata = json.loads(
        (tmp_path / "metadata.json").read_text(encoding="utf-8")
    )
    second_weight = second_metadata["model_files"]["fridge"]

    assert second_weight != first_weight
    assert not (tmp_path / first_weight).exists()
    assert (tmp_path / second_weight).is_file()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "metadata.json",
        second_weight,
    ]
    restored = PatchTST({"device": "cpu", "pretrained_model_path": tmp_path})
    for name, value in model.models["fridge"].state_dict().items():
        torch.testing.assert_close(
            restored.models["fridge"].state_dict()[name], value, rtol=0, atol=0
        )


@pytest.mark.parametrize("failure_point", ["metadata_write", "metadata_publish"])
def test_failed_checkpoint_publication_preserves_previous_generation(
    tmp_path, monkeypatch, failure_point
):
    model = PatchTST(
        _tiny_params(
            save_model_path=tmp_path,
            appliance_params={"fridge": {"mean": 42.0, "std": 5.0}},
        )
    )
    model.models["fridge"] = model.return_network()
    model.save_model()
    original_files = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }

    with torch.no_grad():
        next(model.models["fridge"].parameters()).add_(10.0)

    if failure_point == "metadata_write":

        def fail_metadata(*_args, **_kwargs):
            raise OSError("simulated metadata write failure")

        monkeypatch.setattr(patchtst, "save_metadata", fail_metadata)
    else:
        real_replace = os.replace

        def fail_metadata_publish(source, destination):
            if os.fspath(source).endswith("metadata.json"):
                raise OSError("simulated metadata publication failure")
            return real_replace(source, destination)

        monkeypatch.setattr(patchtst.os, "replace", fail_metadata_publish)

    with pytest.raises(OSError, match="simulated metadata"):
        model.save_model()

    current_files = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    assert current_files == original_files
    assert not any(
        path.name.startswith(".patchtst-stage-") for path in tmp_path.iterdir()
    )


def test_failed_multi_model_weight_publication_removes_partial_generation(
    tmp_path, monkeypatch
):
    model = PatchTST(
        _tiny_params(
            save_model_path=tmp_path,
            appliance_params={
                "fridge": {"mean": 42.0, "std": 5.0},
                "kettle": {"mean": 800.0, "std": 200.0},
            },
        )
    )
    model.models["fridge"] = model.return_network()
    model.models["kettle"] = model.return_network()
    model.save_model()
    original_files = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    real_replace = os.replace
    published_weights = 0

    def fail_second_weight(source, destination):
        nonlocal published_weights
        if os.fspath(source).endswith(".pt"):
            published_weights += 1
            if published_weights == 2:
                raise OSError("simulated second weight failure")
        return real_replace(source, destination)

    monkeypatch.setattr(patchtst.os, "replace", fail_second_weight)
    with pytest.raises(OSError, match="second weight"):
        model.save_model()

    current_files = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    assert current_files == original_files


def test_checkpoint_save_rejects_nonfinite_state_without_mutating_disk(tmp_path):
    model = PatchTST(
        _tiny_params(
            save_model_path=tmp_path,
            appliance_params={"fridge": {"mean": 42.0, "std": 5.0}},
        )
    )
    model.models["fridge"] = model.return_network()
    model.save_model()
    original_files = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    with torch.no_grad():
        next(model.models["fridge"].parameters()).reshape(-1)[0] = float("nan")

    with pytest.raises(FloatingPointError, match="non-finite"):
        model.save_model()

    current_files = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    assert current_files == original_files


def test_checkpoint_roundtrip_preserves_model_and_output_order(tmp_path):
    statistics = OrderedDict(
        [
            ("kettle", {"mean": 800.0, "std": 200.0}),
            ("fridge", {"mean": 42.0, "std": 5.0}),
        ]
    )
    model = PatchTST(
        _tiny_params(save_model_path=tmp_path, appliance_params=statistics)
    )
    model.models["kettle"] = model.return_network()
    model.models["fridge"] = model.return_network()
    model.save_model()

    restored = PatchTST({"device": "cpu", "pretrained_model_path": tmp_path})
    mains = pd.DataFrame({"power": np.linspace(80.0, 140.0, 5)})
    result = restored.disaggregate_chunk([mains])[0]

    assert list(restored.models) == ["kettle", "fridge"]
    assert list(result.columns) == ["kettle", "fridge"]


@pytest.mark.parametrize(
    "model_order",
    ["fridge", ["fridge", "fridge"], ["kettle"]],
)
def test_checkpoint_rejects_invalid_model_order(tmp_path, model_order):
    model = PatchTST(
        _tiny_params(
            save_model_path=tmp_path,
            appliance_params={"fridge": {"mean": 42.0, "std": 5.0}},
        )
    )
    model.models["fridge"] = model.return_network()
    model.save_model()
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_order"] = model_order
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="model_order"):
        PatchTST({"device": "cpu", "pretrained_model_path": tmp_path})


def test_mappingless_checkpoint_rejects_malformed_order_cleanly(tmp_path):
    model = PatchTST(
        _tiny_params(
            save_model_path=tmp_path,
            appliance_params={"fridge": {"mean": 42.0, "std": 5.0}},
        )
    )
    model.models["fridge"] = model.return_network()
    model.save_model()
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("model_files")
    metadata["model_order"] = [{}]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="model_order"):
        PatchTST({"device": "cpu", "pretrained_model_path": tmp_path})


def test_nonfinite_checkpoint_is_rejected_without_mutating_runtime(tmp_path):
    saved = PatchTST(
        _tiny_params(
            save_model_path=tmp_path,
            appliance_params={"fridge": {"mean": 42.0, "std": 5.0}},
        )
    )
    saved.models["fridge"] = saved.return_network()
    saved.save_model()
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    weight_path = tmp_path / metadata["model_files"]["fridge"]
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    next(value for value in state.values() if value.is_floating_point()).reshape(-1)[
        0
    ] = float("nan")
    torch.save(state, weight_path)

    existing = PatchTST(
        _tiny_params(
            sequence_length=11,
            appliance_params={"kettle": {"mean": 800.0, "std": 200.0}},
        )
    )
    existing.models["kettle"] = existing.return_network()
    original_state = {
        name: value.detach().clone()
        for name, value in existing.models["kettle"].state_dict().items()
    }
    existing.load_model_path = tmp_path

    with pytest.raises(FloatingPointError, match="non-finite"):
        existing.load_model()

    assert list(existing.models) == ["kettle"]
    for name, value in original_state.items():
        torch.testing.assert_close(
            existing.models["kettle"].state_dict()[name], value, rtol=0, atol=0
        )


def test_checkpoint_roundtrip_ignores_untrained_appliance_statistics(tmp_path):
    appliance_params = {
        "fridge": {"mean": 42.0, "std": 5.0},
        "kettle": {"mean": 800.0, "std": 200.0},
    }
    model = PatchTST(
        _tiny_params(save_model_path=tmp_path, appliance_params=appliance_params)
    )
    model.models["fridge"] = model.return_network()
    mains = pd.DataFrame({"power": np.linspace(80.0, 140.0, 5)})
    expected = model.disaggregate_chunk([mains])[0]
    model.save_model()

    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert set(metadata["appliance_params"]) == {"fridge", "kettle"}
    assert set(metadata["model_files"]) == {"fridge"}

    restored = PatchTST({"device": "cpu", "pretrained_model_path": tmp_path})
    assert set(restored.appliance_params) == {"fridge", "kettle"}
    assert list(restored.models) == ["fridge"]
    np.testing.assert_array_equal(restored.disaggregate_chunk([mains])[0], expected)

    metadata.pop("model_files")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    legacy_restored = PatchTST({"device": "cpu", "pretrained_model_path": tmp_path})
    assert set(legacy_restored.appliance_params) == {"fridge", "kettle"}
    assert list(legacy_restored.models) == ["fridge"]
    np.testing.assert_array_equal(
        legacy_restored.disaggregate_chunk([mains])[0], expected
    )


def test_failed_checkpoint_load_preserves_existing_runtime_state(tmp_path):
    saved = PatchTST(
        _tiny_params(
            save_model_path=tmp_path,
            appliance_params={"fridge": {"mean": 42.0, "std": 5.0}},
        )
    )
    saved.models["fridge"] = saved.return_network()
    saved.save_model()
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    (tmp_path / metadata["model_files"]["fridge"]).unlink()

    existing = PatchTST(
        _tiny_params(
            sequence_length=11,
            appliance_params={"kettle": {"mean": 800.0, "std": 200.0}},
        )
    )
    existing.models["kettle"] = existing.return_network()
    original_config = existing._model_config()
    original_statistics = dict(existing.appliance_params)
    original_state = {
        name: value.detach().clone()
        for name, value in existing.models["kettle"].state_dict().items()
    }
    existing.load_model_path = tmp_path

    with pytest.raises(FileNotFoundError):
        existing.load_model()

    assert existing.sequence_length == 11
    assert existing._model_config() == original_config
    assert existing.appliance_params == original_statistics
    assert list(existing.models) == ["kettle"]
    for name, value in original_state.items():
        torch.testing.assert_close(
            existing.models["kettle"].state_dict()[name], value, rtol=0, atol=0
        )


def test_checkpoint_rejects_incomplete_architecture_metadata(tmp_path):
    model = PatchTST(
        _tiny_params(
            save_model_path=tmp_path,
            appliance_params={"fridge": {"mean": 42.0, "std": 5.0}},
        )
    )
    model.models["fridge"] = model.return_network()
    model.save_model()
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_config"].pop("d_model")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="model_config"):
        PatchTST({"device": "cpu", "pretrained_model_path": tmp_path})


def test_default_network_parameter_and_serialized_memory_budget():
    model = PatchTST({"device": "cpu"})
    network = model.return_network()
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    buffer = io.BytesIO()
    torch.save(network.state_dict(), buffer)

    assert 50_000 < parameter_count < 200_000
    assert buffer.tell() < 1_000_000
