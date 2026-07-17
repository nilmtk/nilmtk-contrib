import io
import json

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
