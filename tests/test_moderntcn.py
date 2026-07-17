import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
metadata_module = pytest.importorskip("nilmtk_contrib.metadata")
engine_module = pytest.importorskip("nilmtk_contrib.torch._seq2point")
moderntcn_module = pytest.importorskip("nilmtk_contrib.torch.moderntcn")
model_catalog_by_module = metadata_module.model_catalog_by_module
SequenceToPointTorchDisaggregator = engine_module.SequenceToPointTorchDisaggregator
ModernTCN = moderntcn_module.ModernTCN
ModernTCNBlock = moderntcn_module.ModernTCNBlock
ModernTCNNetwork = moderntcn_module.ModernTCNNetwork


def _params(**overrides):
    return {
        "device": "cpu",
        "sequence_length": 9,
        "n_epochs": 1,
        "batch_size": 4,
        "seed": 17,
        "mains_mean": 100.0,
        "mains_std": 40.0,
        "patch_length": 3,
        "patch_stride": 2,
        "d_model": 8,
        "d_ff": 16,
        "n_blocks": 1,
        "large_kernel_size": 5,
        "small_kernel_size": 3,
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


def test_moderntcn_is_an_architecture_only_engine_consumer():
    source = Path(inspect.getfile(ModernTCN)).read_text()

    assert issubclass(ModernTCN, SequenceToPointTorchDisaggregator)
    assert len(source.splitlines()) < 300
    for method in ("partial_fit", "disaggregate_chunk", "save_model", "load_model"):
        assert f"def {method}(" not in source


def test_network_uses_depthwise_multiscale_kernels_and_center_features():
    network = ModernTCN(_params()).return_network()
    block = network.blocks[0]

    assert isinstance(block, ModernTCNBlock)
    assert block.large_kernel.groups == 8
    assert block.small_kernel.groups == 8
    assert block.large_kernel.kernel_size == (5,)
    assert block.small_kernel.kernel_size == (3,)
    assert network.num_patches == 4
    assert network.center_patch_index == 1
    assert network(torch.ones(4, 1, 9)).shape == (4, 1)


def test_network_end_padding_covers_nondivisible_patch_grid():
    network = ModernTCNNetwork(
        sequence_length=11,
        patch_length=4,
        patch_stride=3,
        d_model=4,
        d_ff=8,
        n_blocks=1,
        large_kernel_size=5,
        small_kernel_size=3,
        dropout=0.0,
    )

    assert network.end_padding == 2
    assert network.num_patches == 4
    assert torch.isfinite(network(torch.ones(2, 1, 11))).all()


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"sequence_length": 8}, "odd"),
        ({"patch_length": 10}, "patch_length"),
        ({"patch_stride": 4}, "patch_stride"),
        ({"large_kernel_size": 4}, "must be odd"),
        ({"small_kernel_size": 7}, "must not exceed"),
        ({"dropout": 1.0}, "less than 1"),
        ({"dropout": np.nan}, "finite"),
        ({"n_blocks": True}, "positive integer"),
        ({"weight_decay": -1.0}, "at least"),
    ],
)
def test_invalid_configuration_fails_at_construction(params, message):
    with pytest.raises(ValueError, match=message):
        ModernTCN(_params(**params))


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
    network = ModernTCN(_params()).return_network()

    with pytest.raises((TypeError, ValueError)):
        network(inputs)


def test_one_epoch_train_infer_smoke_preserves_every_row():
    mains, targets = _chunks()
    model = ModernTCN(_params(appliance_params={}))

    model.partial_fit(mains, targets)
    result = model.disaggregate_chunk(mains)[0]

    assert result.index.equals(mains[0].index)
    assert result.columns.tolist() == ["fridge"]
    assert result["fridge"].dtype == np.float32
    assert np.isfinite(result["fridge"]).all()


def test_checkpoint_roundtrip_restores_architecture_and_predictions(tmp_path):
    mains, targets = _chunks()
    model = ModernTCN(
        _params(appliance_params={}, save_model_path=str(tmp_path), d_model=12)
    )
    model.partial_fit(mains, targets)
    expected = model.disaggregate_chunk(mains)[0]

    loaded = ModernTCN(
        _params(
            appliance_params={},
            pretrained_model_path=str(tmp_path),
            d_model=8,
        )
    )
    actual = loaded.disaggregate_chunk(mains)[0]

    assert loaded.d_model == 12
    assert np.allclose(actual, expected)


def test_public_export_catalog_and_defaults_are_complete():
    import nilmtk_contrib.torch as torch_models

    default = ModernTCN({"device": "cpu"})
    entry = model_catalog_by_module()["nilmtk_contrib.torch.moderntcn"]
    assert torch_models.ModernTCN is ModernTCN
    assert default.sequence_length == 299
    assert default.weight_decay == pytest.approx(1e-4)
    assert entry.class_name == "ModernTCN"
    assert entry.exported_from == "nilmtk_contrib.torch"
