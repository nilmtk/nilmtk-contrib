import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
metadata_module = pytest.importorskip("nilmtk_contrib.metadata")
engine_module = pytest.importorskip("nilmtk_contrib.torch._seq2point")
patchtst_module = pytest.importorskip("nilmtk_contrib.torch.patchtst")
model_catalog_by_module = metadata_module.model_catalog_by_module
SequenceToPointTorchDisaggregator = engine_module.SequenceToPointTorchDisaggregator
PatchTST = patchtst_module.PatchTST
PatchTSTNetwork = patchtst_module.PatchTSTNetwork


def _params(**overrides):
    return {
        "device": "cpu",
        "sequence_length": 9,
        "n_epochs": 1,
        "batch_size": 4,
        "seed": 13,
        "mains_mean": 100.0,
        "mains_std": 40.0,
        "patch_length": 3,
        "patch_stride": 2,
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 16,
        "dropout": 0.0,
    } | overrides


def _chunks(length=18):
    values = np.arange(length, dtype=np.float32)
    index = pd.date_range("2026-01-01", periods=length, freq="1min", tz="UTC")
    appliance = 30.0 + 20.0 * ((values // 3) % 2)
    mains = 60.0 + appliance + np.sin(values)
    return (
        [pd.DataFrame({"power": mains}, index=index)],
        [("fridge", [pd.DataFrame({"power": appliance}, index=index)])],
    )


def test_patchtst_is_a_thin_consumer_of_the_shared_engine():
    source = Path(inspect.getfile(PatchTST)).read_text()

    assert issubclass(PatchTST, SequenceToPointTorchDisaggregator)
    assert len(source.splitlines()) < 210
    for method in ("partial_fit", "disaggregate_chunk", "save_model", "load_model"):
        assert f"def {method}(" not in source


def test_network_token_count_and_output_contract():
    network = PatchTSTNetwork(
        sequence_length=9,
        patch_length=3,
        patch_stride=2,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
    )

    output = network(torch.ones(4, 1, 9))

    assert network.num_patches == 5
    assert output.shape == (4, 1)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"sequence_length": 8}, "odd"),
        ({"patch_length": 10}, "patch_length"),
        ({"patch_stride": 4}, "patch_stride"),
        ({"d_model": 7}, "divisible"),
        ({"dropout": 1.0}, "less than 1"),
        ({"dropout": np.nan}, "finite"),
        ({"n_heads": True}, "positive integer"),
        ({"validation_fraction": 1.0}, "less than 1"),
        ({"learning_rate": 0.0}, "greater than"),
    ],
)
def test_invalid_configuration_fails_at_construction(params, message):
    with pytest.raises(ValueError, match=message):
        PatchTST(_params(**params))


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
    network = PatchTST(_params()).return_network()

    with pytest.raises((TypeError, ValueError)):
        network(inputs)


def test_one_epoch_real_pipeline_smoke_preserves_every_row():
    mains, targets = _chunks()
    model = PatchTST(_params(appliance_params={}))

    model.partial_fit(mains, targets)
    result = model.disaggregate_chunk(mains)[0]

    assert result.index.equals(mains[0].index)
    assert result.columns.tolist() == ["fridge"]
    assert result["fridge"].dtype == np.float32
    assert np.isfinite(result["fridge"]).all()


def test_seeded_training_is_reproducible_without_leaking_global_rng():
    mains, targets = _chunks()
    first = PatchTST(_params(appliance_params={}))
    second = PatchTST(_params(appliance_params={}))
    state_before = torch.random.get_rng_state().clone()

    first.partial_fit(mains, targets)
    state_after_first = torch.random.get_rng_state().clone()
    second.partial_fit(mains, targets)

    assert torch.equal(state_before, state_after_first)
    assert torch.equal(state_before, torch.random.get_rng_state())
    for key, value in first.models["fridge"].state_dict().items():
        assert torch.equal(value, second.models["fridge"].state_dict()[key])


def test_checkpoint_roundtrip_restores_architecture_and_predictions(tmp_path):
    mains, targets = _chunks()
    model = PatchTST(
        _params(appliance_params={}, save_model_path=str(tmp_path), d_model=12)
    )
    model.partial_fit(mains, targets)
    expected = model.disaggregate_chunk(mains)[0]

    loaded = PatchTST(
        _params(
            appliance_params={},
            pretrained_model_path=str(tmp_path),
            d_model=8,
        )
    )
    actual = loaded.disaggregate_chunk(mains)[0]

    assert loaded.d_model == 12
    assert np.allclose(actual, expected)


def test_public_export_and_catalog_are_lazy_and_complete():
    import nilmtk_contrib.torch as torch_models

    entry = model_catalog_by_module()["nilmtk_contrib.torch.patchtst"]
    assert torch_models.PatchTST is PatchTST
    assert entry.class_name == "PatchTST"
    assert entry.exported_from == "nilmtk_contrib.torch"
