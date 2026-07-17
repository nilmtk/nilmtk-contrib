from collections import OrderedDict
from copy import deepcopy
import random

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
        ({"mains_mean": True}, ValueError, "mains_mean"),
        ({"mains_mean": np.nan}, ValueError, "mains_mean"),
        ({"mains_mean": np.inf}, ValueError, "mains_mean"),
        ({"mains_std": True}, ValueError, "mains_std"),
        ({"mains_std": "large"}, ValueError, "mains_std"),
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


@pytest.mark.parametrize(
    ("appliance_params", "error", "message"),
    [
        ({1: {"mean": 1.0, "std": 2.0}}, ValueError, "names"),
        ({" fridge ": {"mean": 1.0, "std": 2.0}}, ValueError, "whitespace"),
        ({"fridge": []}, TypeError, "mapping"),
        ({"fridge": {"std": 2.0}}, ValueError, "mean"),
        ({"fridge": {"mean": 1.0}}, ValueError, "std"),
        ({"fridge": {"mean": True, "std": 2.0}}, ValueError, "finite"),
        ({"fridge": {"mean": np.inf, "std": 2.0}}, ValueError, "finite"),
        ({"fridge": {"mean": 1.0, "std": -2.0}}, ValueError, "positive"),
        (
            {"fridge": {"mean": 1.0, "std": 2.0, "min": np.nan}},
            ValueError,
            "finite",
        ),
        (
            {"fridge": {"mean": 1.0, "std": 2.0, "min": 5.0, "max": 4.0}},
            ValueError,
            "exceed",
        ),
    ],
)
def test_torch_disaggregator_rejects_invalid_appliance_params(
    appliance_params, error, message
):
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import TorchDisaggregator

    with pytest.raises(error, match=message):
        TorchDisaggregator({"appliance_params": appliance_params}, defaults=DEFAULTS)


def test_torch_disaggregator_detaches_caller_owned_appliance_params():
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import TorchDisaggregator

    params = {
        "appliance_params": {
            "fridge": {
                "mean": 1.0,
                "std": 2.0,
                "threshold": 60.0,
                "metadata": {"source": "caller"},
            }
        }
    }
    original = deepcopy(params)

    model = TorchDisaggregator(params, defaults=DEFAULTS)
    assert params == original

    params["appliance_params"]["fridge"]["mean"] = 99.0
    params["appliance_params"]["fridge"]["metadata"]["source"] = "mutated"
    model.appliance_params["fridge"]["std"] = 3.0

    assert model.appliance_params["fridge"] == {
        "mean": 1.0,
        "std": 3.0,
        "threshold": 60.0,
        "metadata": {"source": "caller"},
    }
    assert params["appliance_params"]["fridge"]["std"] == 2.0


def test_torch_disaggregator_seed_reproducibly_controls_all_rngs():
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import TorchDisaggregator

    TorchDisaggregator({"seed": 17}, defaults=DEFAULTS)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    TorchDisaggregator({"seed": 17}, defaults=DEFAULTS)
    second = (random.random(), np.random.random(), torch.rand(1).item())

    assert second == pytest.approx(first)


def test_invalid_device_does_not_mutate_global_rng_state(monkeypatch):
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import TorchDisaggregator

    def draw_rngs():
        return random.random(), np.random.random(), torch.rand(1).item()

    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    expected = draw_rngs()
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        TorchDisaggregator({"seed": 999, "device": "cuda"}, defaults=DEFAULTS)

    assert draw_rngs() == pytest.approx(expected)


def test_resolve_torch_device_defaults_to_cpu_without_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import resolve_torch_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

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


def test_resolve_torch_device_accepts_visible_cuda_and_counts_devices_once(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import resolve_torch_device

    calls = 0

    def device_count():
        nonlocal calls
        calls += 1
        return 1

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", device_count)

    assert resolve_torch_device("cuda:0") == torch.device("cuda:0")
    assert calls == 1


def test_resolve_torch_device_rejects_inconsistent_cuda_runtime(monkeypatch):
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import resolve_torch_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with pytest.raises(RuntimeError, match="no visible devices"):
        resolve_torch_device("cuda")


def test_resolve_torch_device_rejects_unavailable_mps(monkeypatch):
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import resolve_torch_device

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="MPS"):
        resolve_torch_device("mps")


def test_resolve_torch_device_accepts_explicit_cpu_and_available_mps(monkeypatch):
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import resolve_torch_device

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_torch_device(" cpu ") == torch.device("cpu")
    assert resolve_torch_device(torch.device("cpu")) == torch.device("cpu")
    assert resolve_torch_device("mps:0") == torch.device("mps:0")


def test_resolve_torch_device_rejects_nonzero_mps_index(monkeypatch):
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import resolve_torch_device

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match="index 0"):
        resolve_torch_device("mps:1")


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

    frame = pd.DataFrame({"active": [1.0, 3.0, 5.0, 7.0]})

    assert compute_appliance_stats([("fridge", [frame])])["fridge"] == pytest.approx(
        {"mean": 4.0, "std": np.std([1.0, 3.0, 5.0, 7.0])}
    )


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"active": [1.0, 2.0], "apparent": [2.0, 3.0]}),
        np.array([[1.0, 2.0, 3.0]]),
        np.ones((2, 1, 1)),
        np.array(1.0),
    ],
)
def test_compute_appliance_stats_rejects_ambiguous_target_shapes(frame):
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import compute_appliance_stats

    with pytest.raises(ValueError, match="one power column"):
        compute_appliance_stats([("fridge", [frame])])


def test_compute_appliance_stats_is_partition_invariant_at_large_offsets():
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import compute_appliance_stats

    values = 1e12 + np.arange(100, dtype=np.float64) * 2.0
    whole = compute_appliance_stats(
        [("fridge", [values])], std_floor=0.01, include_extrema=True
    )["fridge"]
    partitioned = compute_appliance_stats(
        [("fridge", [values[:1], values[1:17], values[17:63], values[63:]])],
        std_floor=0.01,
        include_extrema=True,
    )["fridge"]

    assert partitioned["mean"] == pytest.approx(whole["mean"], abs=1e-4)
    assert partitioned["std"] == pytest.approx(whole["std"], rel=1e-12)
    assert partitioned["min"] == whole["min"]
    assert partitioned["max"] == whole["max"]


def test_compute_appliance_stats_consumes_frames_without_materializing_them():
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import compute_appliance_stats

    class StreamingFrames:
        def __iter__(self):
            yield np.array([1.0, 2.0])
            yield np.array([3.0, 4.0])

        def __len__(self):
            raise AssertionError("streaming frames must not be materialized")

    stats = compute_appliance_stats([("fridge", StreamingFrames())])

    assert stats["fridge"] == pytest.approx(
        {"mean": 2.5, "std": np.std([1.0, 2.0, 3.0, 4.0])}
    )


def test_compute_appliance_stats_matches_numpy_for_uneven_nilm_like_chunks():
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import compute_appliance_stats

    rng = np.random.default_rng(31)
    samples = np.arange(4097)
    values = (
        8.0
        + 90.0 * ((samples // 137) % 2)
        + 1200.0 * ((samples % 997) < 9)
        + rng.normal(0.0, 1.5, samples.size)
    )
    frames = [
        np.array([]),
        values[:1],
        values[1:73],
        values[73:1000],
        values[1000:4096],
        values[4096:],
    ]

    stats = compute_appliance_stats(
        [("fridge", frames)], std_floor=0.01, include_extrema=True
    )["fridge"]

    assert stats == pytest.approx(
        {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
        },
        rel=1e-12,
    )


@pytest.mark.parametrize(
    ("appliances", "message"),
    [
        (None, "At least one"),
        ([], "At least one"),
        ([("", [np.ones(2)])], "names"),
        ([(" fridge ", [np.ones(2)])], "whitespace"),
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
        ([("fridge", "not frames")], TypeError, "arrays"),
        ([("fridge", [np.array([1.0 + 2.0j])])], TypeError, "real numeric"),
        ([("fridge", [np.array(["not power"])])], TypeError, "real numeric"),
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


def test_set_appliance_params_is_atomic_when_later_target_is_invalid():
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import TorchDisaggregator

    existing = {"fridge": {"mean": 10.0, "std": 2.0}}
    model = TorchDisaggregator({"appliance_params": existing}, defaults=DEFAULTS)
    before = deepcopy(model.appliance_params)

    with pytest.raises(ValueError, match="finite"):
        model.set_appliance_params(
            [
                ("fridge", [np.array([1.0, 2.0])]),
                ("kettle", [np.array([1.0, np.nan])]),
            ]
        )

    assert model.appliance_params == before


def test_invalid_subclass_normalization_policy_fails_before_state_update():
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import TorchDisaggregator

    class InvalidPolicyDisaggregator(TorchDisaggregator):
        APPLIANCE_STD_FALLBACK = 0

    model = InvalidPolicyDisaggregator(defaults=DEFAULTS)

    with pytest.raises(ValueError, match="std_fallback"):
        model.set_appliance_params([("fridge", [np.ones(2)])])

    assert model.appliance_params == {}


def test_require_models_validates_external_model_mapping():
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import TorchDisaggregator

    model = TorchDisaggregator(defaults=DEFAULTS)
    network = torch.nn.Linear(1, 1)

    external = {"fridge": network}
    installed = model.require_models(external)
    external.clear()

    assert installed == OrderedDict([("fridge", network)])
    assert installed is model.models
    assert model.require_models() is installed


@pytest.mark.parametrize(
    ("models", "error", "message"),
    [
        (None, RuntimeError, "trained or loaded"),
        ([], TypeError, "mapping"),
        ({}, ValueError, "at least one"),
        ({"": object()}, ValueError, "names"),
        ({" fridge ": object()}, ValueError, "whitespace"),
        ({"fridge": object()}, TypeError, "torch.nn.Module"),
    ],
)
def test_require_models_rejects_invalid_or_empty_models(models, error, message):
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import TorchDisaggregator

    model = TorchDisaggregator(defaults=DEFAULTS)

    with pytest.raises(error, match=message):
        model.require_models(models)


@pytest.mark.parametrize("invalid_models", [{}, {"kettle": object()}])
def test_require_models_rejects_atomically_without_erasing_installed_models(
    invalid_models,
):
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import TorchDisaggregator

    model = TorchDisaggregator(defaults=DEFAULTS)
    network = torch.nn.Linear(1, 1)
    installed = model.require_models({"fridge": network})

    with pytest.raises((TypeError, ValueError)):
        model.require_models(invalid_models)

    assert model.models is installed
    assert model.models == OrderedDict([("fridge", network)])


def test_require_models_rejects_partially_valid_map_atomically():
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import TorchDisaggregator

    model = TorchDisaggregator(defaults=DEFAULTS)
    fridge = torch.nn.Linear(1, 1)
    installed = model.require_models({"fridge": fridge})

    with pytest.raises(TypeError, match="bad"):
        model.require_models(
            OrderedDict(
                [
                    ("kettle", torch.nn.Linear(1, 1)),
                    ("bad", object()),
                ]
            )
        )

    assert model.models is installed
    assert model.models == OrderedDict([("fridge", fridge)])


@pytest.mark.parametrize(
    "installed_models",
    [
        {"fridge": object()},
        {" fridge ": object()},
    ],
)
def test_require_models_validates_previously_installed_internal_state(
    installed_models,
):
    pytest.importorskip("torch")
    from nilmtk_contrib.torch._base import TorchDisaggregator

    model = TorchDisaggregator(defaults=DEFAULTS)
    model.models = installed_models

    with pytest.raises((TypeError, ValueError)):
        model.require_models()
