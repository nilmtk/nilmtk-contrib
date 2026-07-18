import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
base_module = pytest.importorskip("nilmtk_contrib.torch._base")
engine_module = pytest.importorskip("nilmtk_contrib.torch._seq2point")
nn = torch.nn
torch_defaults = base_module.torch_defaults
SequenceToPointTorchDisaggregator = engine_module.SequenceToPointTorchDisaggregator
centered_windows = engine_module.centered_windows
power_vector = engine_module.power_vector
window_matrix = engine_module.window_matrix


class TinyNetwork(nn.Module):
    def __init__(self, sequence_length):
        super().__init__()
        self.linear = nn.Linear(sequence_length, 1)

    def forward(self, inputs):
        return self.linear(inputs.squeeze(1))


class TinyDisaggregator(SequenceToPointTorchDisaggregator):
    MODEL_NAME = "Tiny"
    CHECKPOINT_PREFIX = "tiny"
    MODEL_CONFIG_FIELDS = (
        "learning_rate",
        "weight_decay",
        "validation_fraction",
        "validation_strategy",
        "gradient_clip_norm",
    )

    def __init__(self, params=None):
        super().__init__(params, defaults=torch_defaults(sequence_length=5, n_epochs=1))
        if self.load_model_path:
            self.load_model()

    def return_network(self):
        return TinyNetwork(self.sequence_length).to(self.device)


class L1TinyDisaggregator(TinyDisaggregator):
    def __init__(self, params=None):
        self.loss_calls = []
        super().__init__(params)

    def training_loss(self, network, batch_inputs, batch_targets, appliance_name):
        self.loss_calls.append((appliance_name, len(batch_inputs)))
        prediction = self._checked_prediction(network, batch_inputs, appliance_name)
        target = batch_targets.to(self.device)
        return nn.functional.l1_loss(prediction, target)


def _params(**overrides):
    return {
        "device": "cpu",
        "sequence_length": 5,
        "n_epochs": 1,
        "batch_size": 4,
        "seed": 7,
        "mains_mean": 10.0,
        "mains_std": 2.0,
        "appliance_params": {"fridge": {"mean": 5.0, "std": 2.0}},
    } | overrides


def _raw(values, *, start="2026-01-01"):
    index = pd.date_range(start, periods=len(values), freq="1min", tz="UTC")
    return pd.DataFrame({"power": np.asarray(values, dtype=np.float32)}, index=index)


def _zero_network(sequence_length=5, bias=0.0):
    network = TinyNetwork(sequence_length)
    with torch.no_grad():
        network.linear.weight.zero_()
        network.linear.bias.fill_(bias)
    return network


def test_centered_windows_preserve_one_row_per_raw_sample():
    result = centered_windows(np.array([8.0, 10.0, 12.0]), 5, 10.0, 2.0)

    assert result.shape == (3, 5)
    assert result.dtype == np.float32
    assert result.tolist() == [
        [-5.0, -5.0, -1.0, 0.0, 1.0],
        [-5.0, -1.0, 0.0, 1.0, -5.0],
        [-1.0, 0.0, 1.0, -5.0, -5.0],
    ]


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        ([[1.0, 2.0], [3.0, 4.0]], ValueError, "one power column"),
        ([1.0, np.nan], ValueError, "finite"),
        ([1.0, np.inf], ValueError, "finite"),
        ([True, False], TypeError, "real numeric"),
        ([1 + 2j], TypeError, "real numeric"),
        (["1", "2"], TypeError, "real numeric"),
    ],
)
def test_power_vector_rejects_ambiguous_or_lossy_inputs(values, error, message):
    with pytest.raises(error, match=message):
        power_vector(np.asarray(values), "power")


def test_window_matrix_requires_exact_finite_shape():
    valid = window_matrix(np.ones((3, 5)), 5, "windows")
    assert valid.shape == (3, 5)

    with pytest.raises(ValueError, match="shape"):
        window_matrix(np.ones((3, 4)), 5, "windows")
    with pytest.raises(ValueError, match="finite"):
        window_matrix(np.full((3, 5), np.nan), 5, "windows")


def test_raw_inference_preserves_independent_indexes_and_empty_chunks():
    model = TinyDisaggregator(_params())
    model.require_models({"fridge": _zero_network(bias=1.0)})
    first = _raw([10.0, 11.0, 12.0])
    second = _raw([13.0, 14.0], start="2026-02-01")
    empty = _raw([], start="2026-03-01")

    results = model.disaggregate_chunk([first, second, empty])

    assert [len(result) for result in results] == [3, 2, 0]
    assert results[0].index.equals(first.index)
    assert results[1].index.equals(second.index)
    assert results[2].index.equals(empty.index)
    assert results[0]["fridge"].tolist() == [7.0, 7.0, 7.0]


def test_preprocessed_inference_consumes_windows_without_rewindowing():
    model = TinyDisaggregator(_params())
    model.require_models({"fridge": _zero_network()})
    index = pd.Index(["a", "b", "c"])
    windows = pd.DataFrame(np.ones((3, 5), dtype=np.float32), index=index)

    result = model.disaggregate_chunk([windows], do_preprocessing=False)[0]

    assert len(result) == 3
    assert result.index.equals(index)


class _BadOutput(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.kind = kind

    def forward(self, inputs):
        if self.kind == "shape":
            return torch.zeros((len(inputs), 2)) + self.anchor
        if self.kind == "integer":
            return torch.zeros((len(inputs), 1), dtype=torch.int64)
        if self.kind == "bfloat16":
            return (torch.zeros((len(inputs), 1)) + self.anchor).to(torch.bfloat16)
        return torch.full((len(inputs), 1), float("inf")) + self.anchor


@pytest.mark.parametrize(
    ("kind", "error", "message"),
    [
        ("shape", ValueError, "returned shape"),
        ("integer", TypeError, "floating"),
        ("nonfinite", FloatingPointError, "non-finite"),
    ],
)
def test_inference_fails_closed_on_invalid_model_outputs(kind, error, message):
    model = TinyDisaggregator(_params())
    model.require_models({"fridge": _BadOutput(kind)})

    with pytest.raises(error, match=message):
        model.disaggregate_chunk([_raw([10.0, 11.0])])


def test_inference_accepts_bfloat16_and_publishes_float32():
    model = TinyDisaggregator(_params())
    model.require_models({"fridge": _BadOutput("bfloat16")})

    result = model.disaggregate_chunk([_raw([10.0, 11.0])])[0]

    assert result["fridge"].dtype == np.float32
    assert np.isfinite(result["fridge"]).all()


def test_raw_training_validates_alignment_and_trains_a_finite_model():
    model = TinyDisaggregator(_params(appliance_params={}))
    mains = _raw([10, 12, 14, 16, 18, 20, 22, 24])
    target = _raw([2, 4, 6, 8, 10, 12, 14, 16])

    model.partial_fit([mains], [("fridge", [target])])
    predictions = model.disaggregate_chunk([mains])[0]

    assert list(model.models) == ["fridge"]
    assert model.last_split_metadata["fridge"].should_train
    assert predictions.index.equals(mains.index)
    assert np.isfinite(predictions["fridge"]).all()


def test_training_loss_hook_reuses_the_engine_lifecycle():
    model = L1TinyDisaggregator(_params(appliance_params={}))
    mains = _raw([10, 12, 14, 16, 18, 20, 22, 24])
    target = _raw([2, 4, 6, 8, 10, 12, 14, 16])

    model.partial_fit([mains], [("fridge", [target])])

    assert model.loss_calls
    assert {name for name, _ in model.loss_calls} == {"fridge"}
    assert sum(size for _, size in model.loss_calls) > 0
    assert model.last_split_metadata["fridge"].should_train


@pytest.mark.parametrize(
    ("result", "error", "message"),
    [
        (object(), TypeError, "torch.Tensor"),
        (torch.ones(1, requires_grad=True), ValueError, "scalar"),
        (torch.ones((), dtype=torch.int64), TypeError, "real floating"),
        (
            torch.full((), float("inf"), requires_grad=True),
            FloatingPointError,
            "non-finite",
        ),
        (torch.ones(()), ValueError, "autograd"),
        (
            torch.ones((), device="meta", requires_grad=True),
            ValueError,
            "expected cpu",
        ),
    ],
)
def test_training_loss_hook_fails_closed(result, error, message, monkeypatch):
    model = TinyDisaggregator(_params())
    network = model.return_network()
    inputs = torch.ones(2, model.sequence_length)
    targets = torch.ones(2, 1)
    monkeypatch.setattr(model, "training_loss", lambda *_: result)

    with pytest.raises(error, match=message):
        model._checked_training_loss(network, inputs, targets, "fridge")


def test_raw_training_rejects_misaligned_indexes_before_mutating_state():
    model = TinyDisaggregator(_params(appliance_params={}))
    mains = _raw([10, 12, 14])
    target = _raw([2, 4, 6], start="2026-02-01")

    with pytest.raises(ValueError, match="aligned indexes"):
        model.partial_fit([mains], [("fridge", [target])])

    assert model.models == {}
    assert model.appliance_params == {}


def test_failed_multitarget_fit_rolls_back_models_stats_and_split_metadata(monkeypatch):
    model = TinyDisaggregator(_params())
    original = _zero_network(bias=0.25)
    model.require_models({"fridge": original})
    original_state = {
        key: value.detach().clone() for key, value in original.state_dict().items()
    }
    model.last_split_metadata["prior"] = "kept"
    mains = _raw([10, 12, 14, 16])
    target = _raw([2, 4, 6, 8])

    def fail_on_second(appliance_name, inputs, targets):
        if appliance_name == "kettle":
            raise RuntimeError("boom")
        with torch.no_grad():
            model.models["fridge"].linear.bias.add_(10)

    monkeypatch.setattr(model, "_fit_appliance", fail_on_second)
    with pytest.raises(RuntimeError, match="boom"):
        model.partial_fit([mains], [("fridge", [target]), ("kettle", [target])])

    assert model.last_split_metadata == {"prior": "kept"}
    assert list(model.appliance_params) == ["fridge"]
    for key, value in model.models["fridge"].state_dict().items():
        assert torch.equal(value, original_state[key])


def test_checkpoint_roundtrip_is_single_bundle_checksum_bound_and_ordered(tmp_path):
    model = TinyDisaggregator(_params(save_model_path=str(tmp_path)))
    model.appliance_params["kettle"] = {"mean": 10.0, "std": 3.0}
    model.require_models(
        OrderedDict(
            [
                ("fridge", _zero_network(bias=1.0)),
                ("kettle", _zero_network(bias=2.0)),
            ]
        )
    )
    model.save_model()
    loaded = TinyDisaggregator(
        _params(pretrained_model_path=str(tmp_path), appliance_params={})
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["model_order"] == ["fridge", "kettle"]
    assert len(list(tmp_path.glob("tiny-*.pt"))) == 1
    assert list(loaded.models) == ["fridge", "kettle"]
    actual = loaded.disaggregate_chunk([_raw([10.0, 12.0])])[0]
    assert actual.columns.tolist() == ["fridge", "kettle"]
    assert actual["fridge"].tolist() == [7.0, 7.0]
    assert actual["kettle"].tolist() == [16.0, 16.0]


def test_checkpoint_rejects_tampered_weights_and_path_traversal(tmp_path):
    model = TinyDisaggregator(_params(save_model_path=str(tmp_path)))
    model.require_models({"fridge": _zero_network()})
    model.save_model()
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    weights_path = tmp_path / metadata["weights_file"]
    weights_path.write_bytes(weights_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="checksum"):
        TinyDisaggregator(_params(pretrained_model_path=str(tmp_path)))

    model.save_model()
    metadata = json.loads(metadata_path.read_text())
    metadata["weights_file"] = "../outside.pt"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="unsafe"):
        TinyDisaggregator(_params(pretrained_model_path=str(tmp_path)))


def test_failed_metadata_publish_removes_uncommitted_weight(tmp_path, monkeypatch):
    model = TinyDisaggregator(_params(save_model_path=str(tmp_path)))
    model.require_models({"fridge": _zero_network()})

    def fail(*_):
        raise OSError("metadata failure")

    monkeypatch.setattr("nilmtk_contrib.torch._seq2point.save_metadata_atomic", fail)
    with pytest.raises(OSError, match="metadata failure"):
        model.save_model()

    assert list(tmp_path.glob("tiny-*.pt")) == []
    assert not (tmp_path / "metadata.json").exists()


def test_model_source_owns_engine_contract_once():
    source = Path(
        __import__("nilmtk_contrib.torch._seq2point", fromlist=["__file__"]).__file__
    ).read_text()

    for method in ("partial_fit", "disaggregate_chunk", "save_model", "load_model"):
        assert source.count(f"def {method}(") == 1
