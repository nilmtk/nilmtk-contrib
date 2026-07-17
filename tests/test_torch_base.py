from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest


DEFAULTS = {
    "sequence_length": 99,
    "n_epochs": 1,
    "batch_size": 32,
    "mains_mean": 1800.0,
    "mains_std": 600.0,
    "appliance_params": {},
    "save_model_path": None,
    "pretrained_model_path": None,
    "chunk_wise_training": False,
    "seed": None,
    "verbose": False,
    "device": "cpu",
}


def test_torch_disaggregator_centralizes_common_runtime_state():
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import TorchDisaggregator

    model = TorchDisaggregator(
        {
            "sequence_length": 101,
            "batch_size": 8,
            "seed": 7,
            "save_model_path": "/tmp/save",
            "pretrained_model_path": "/tmp/load",
        },
        defaults=DEFAULTS,
    )

    assert model.sequence_length == 101
    assert model.n_epochs == 1
    assert model.batch_size == 8
    assert model.seed == 7
    assert model.save_model_path == "/tmp/save"
    assert model.load_model_path == "/tmp/load"
    assert model.device == torch.device("cpu")
    assert model.models == OrderedDict()


@pytest.mark.parametrize(
    ("params", "error", "message"),
    [
        ([], TypeError, "params"),
        ({"mains_mean": np.nan}, ValueError, "mains_mean"),
        ({"mains_std": np.inf}, ValueError, "mains_std"),
        ({"mains_std": -1.0}, ValueError, "mains_std"),
        ({"seed": True}, ValueError, "seed"),
        ({"verbose": 1}, ValueError, "verbose"),
        ({"chunk_wise_training": 1}, ValueError, "chunk_wise_training"),
        ({"appliance_params": []}, TypeError, "appliance_params"),
    ],
)
def test_torch_disaggregator_rejects_invalid_common_state(params, error, message):
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import TorchDisaggregator

    with pytest.raises(error, match=message):
        TorchDisaggregator(params, defaults=DEFAULTS)


def test_torch_disaggregator_requires_mapping_defaults():
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import TorchDisaggregator

    with pytest.raises(TypeError, match="defaults"):
        TorchDisaggregator(defaults=None)


def test_resolve_torch_device_defaults_to_cpu_without_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import resolve_torch_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_torch_device() == torch.device("cpu")


def test_resolve_torch_device_defaults_to_cuda_when_available(monkeypatch):
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import resolve_torch_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    assert resolve_torch_device() == torch.device("cuda")


@pytest.mark.parametrize("requested", ["", "   ", True, 3, "meta", "cuda:not-an-index"])
def test_resolve_torch_device_rejects_invalid_requests(requested):
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import resolve_torch_device

    with pytest.raises(ValueError, match="device"):
        resolve_torch_device(requested)


def test_resolve_torch_device_rejects_unavailable_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import resolve_torch_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_torch_device("cuda")


def test_resolve_torch_device_rejects_out_of_range_cuda_index(monkeypatch):
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import resolve_torch_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(RuntimeError, match="index 1"):
        resolve_torch_device("cuda:1")


def test_resolve_torch_device_rejects_unavailable_mps(monkeypatch):
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import resolve_torch_device

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="MPS"):
        resolve_torch_device("mps")


def test_compute_appliance_stats_handles_frames_and_explicit_flat_signal_policy():
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import compute_appliance_stats

    stats = compute_appliance_stats(
        [
            ("fridge", [np.array([0.0, 2.0]), np.array([4.0])]),
            ("kettle", [np.ones(3)]),
        ],
        std_fallback=10.0,
        include_extrema=True,
    )

    assert stats["fridge"] == pytest.approx(
        {"mean": 2.0, "std": np.std([0.0, 2.0, 4.0]), "max": 4.0, "min": 0.0}
    )
    assert stats["kettle"] == pytest.approx(
        {"mean": 1.0, "std": 10.0, "max": 1.0, "min": 1.0}
    )


def test_compute_appliance_stats_accepts_pandas_frames():
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import compute_appliance_stats

    frame = pd.DataFrame({"active": [1.0, 3.0], "apparent": [5.0, 7.0]})

    assert compute_appliance_stats([("fridge", [frame])])["fridge"] == pytest.approx(
        {"mean": 4.0, "std": np.std([1.0, 5.0, 3.0, 7.0])}
    )


@pytest.mark.parametrize(
    ("appliances", "message"),
    [
        (None, "At least one"),
        ([], "At least one"),
        ([("", [np.ones(2)])], "names"),
        ([("fridge", [])], "no frames"),
        ([("fridge", [np.array([])])], "no samples"),
        ([("fridge", [np.array([1.0, np.nan])])], "finite"),
        ([("fridge", [np.array([1e308, -1e308])])], "overflowed"),
        (
            [("fridge", [np.ones(2)]), ("fridge", [np.ones(2)])],
            "Duplicate",
        ),
    ],
)
def test_compute_appliance_stats_rejects_bad_training_data(appliances, message):
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import compute_appliance_stats

    with pytest.raises(ValueError, match=message):
        compute_appliance_stats(appliances)


@pytest.mark.parametrize(
    ("appliances", "error", "message"),
    [
        (1, TypeError, "iterable"),
        (["fridge"], ValueError, "pair"),
        ([("fridge", None)], TypeError, "frames"),
    ],
)
def test_compute_appliance_stats_rejects_malformed_containers(
    appliances, error, message
):
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import compute_appliance_stats

    with pytest.raises(error, match=message):
        compute_appliance_stats(appliances)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"std_floor": 0}, "std_floor"),
        ({"std_fallback": np.nan}, "std_fallback"),
        ({"include_extrema": 1}, "include_extrema"),
    ],
)
def test_compute_appliance_stats_rejects_invalid_policy(kwargs, message):
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import compute_appliance_stats

    with pytest.raises(ValueError, match=message):
        compute_appliance_stats([("fridge", [np.ones(2)])], **kwargs)


def test_base_appliance_policy_is_overridable_without_reimplementing_stats():
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import TorchDisaggregator

    class UnitScaleDisaggregator(TorchDisaggregator):
        APPLIANCE_STD_FALLBACK = 1.0
        INCLUDE_APPLIANCE_EXTREMA = True

    model = UnitScaleDisaggregator(defaults=DEFAULTS)
    model.set_appliance_params([("fridge", [np.ones(3)])])

    assert model.appliance_params["fridge"] == {
        "mean": 1.0,
        "std": 1.0,
        "max": 1.0,
        "min": 1.0,
    }


def test_require_models_validates_external_model_mapping():
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import TorchDisaggregator

    model = TorchDisaggregator(defaults=DEFAULTS)
    network = torch.nn.Linear(1, 1)

    installed = model.require_models({"fridge": network})

    assert installed == OrderedDict([("fridge", network)])
    assert installed is model.models


@pytest.mark.parametrize(
    ("models", "error", "message"),
    [
        (None, RuntimeError, "trained or loaded"),
        ([], TypeError, "mapping"),
        ({"": object()}, ValueError, "names"),
        ({"fridge": object()}, TypeError, "torch.nn.Module"),
    ],
)
def test_require_models_rejects_invalid_or_empty_models(models, error, message):
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import TorchDisaggregator

    model = TorchDisaggregator(defaults=DEFAULTS)

    with pytest.raises(error, match=message):
        model.require_models(models)
