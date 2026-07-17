from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from model_smoke_helpers import TORCH_MODEL_SPECS


@dataclass(frozen=True)
class ExpectedDefaults:
    sequence_length: int
    n_epochs: int
    batch_size: int
    mains_mean: float
    mains_std: float


DEFAULTS_BY_MODULE = {
    "nilmtk_contrib.torch.TCN": ExpectedDefaults(99, 10, 512, 1800, 600),
    "nilmtk_contrib.torch.WindowGRU": ExpectedDefaults(99, 10, 512, 1800, 600),
    "nilmtk_contrib.torch.bert": ExpectedDefaults(99, 10, 512, 1800, 600),
    "nilmtk_contrib.torch.conv_lstm": ExpectedDefaults(99, 10, 512, 1800, 600),
    "nilmtk_contrib.torch.dae": ExpectedDefaults(99, 10, 512, 1000, 600),
    "nilmtk_contrib.torch.dlinear": ExpectedDefaults(299, 10, 128, 1800, 600),
    "nilmtk_contrib.torch.msdc": ExpectedDefaults(99, 50, 256, 1800, 600),
    "nilmtk_contrib.torch.msdc_without_crf": ExpectedDefaults(99, 50, 256, 1800, 600),
    "nilmtk_contrib.torch.moderntcn": ExpectedDefaults(299, 10, 128, 1800, 600),
    "nilmtk_contrib.torch.nilmformer": ExpectedDefaults(99, 100, 1024, 1800, 600),
    "nilmtk_contrib.torch.patchtst": ExpectedDefaults(99, 10, 128, 1800, 600),
    "nilmtk_contrib.torch.reformer": ExpectedDefaults(99, 10, 512, 1800, 600),
    "nilmtk_contrib.torch.resnet": ExpectedDefaults(299, 10, 512, 1800, 600),
    "nilmtk_contrib.torch.resnet_classification": ExpectedDefaults(
        99, 10, 512, 1800, 600
    ),
    "nilmtk_contrib.torch.rnn": ExpectedDefaults(19, 10, 512, 1800, 600),
    "nilmtk_contrib.torch.rnn_attention": ExpectedDefaults(19, 10, 512, 1800, 600),
    "nilmtk_contrib.torch.rnn_attention_classification": ExpectedDefaults(
        99, 10, 512, 1800, 600
    ),
    "nilmtk_contrib.torch.seq2point": ExpectedDefaults(99, 10, 512, 1800, 600),
    "nilmtk_contrib.torch.seq2seq": ExpectedDefaults(99, 10, 512, 1800, 600),
}

EXTREMA_MODULES = {
    "nilmtk_contrib.torch.resnet",
    "nilmtk_contrib.torch.resnet_classification",
    "nilmtk_contrib.torch.rnn_attention_classification",
}

NO_STATISTICS_MODULES = {"nilmtk_contrib.torch.WindowGRU"}
MINMAX_MODULES = {
    "nilmtk_contrib.torch.resnet_classification",
    "nilmtk_contrib.torch.rnn_attention_classification",
}


def _load_class(spec):
    pytest.importorskip("torch")
    module = importlib.import_module(spec.module_name)
    return getattr(module, spec.class_name)


@pytest.mark.parametrize(
    "spec",
    TORCH_MODEL_SPECS,
    ids=lambda spec: f"{spec.module_name}.{spec.class_name}",
)
def test_every_torch_model_uses_shared_runtime_with_legacy_defaults(spec):
    from nilmtk_contrib.torch._base import TorchDisaggregator

    cls = _load_class(spec)
    expected = DEFAULTS_BY_MODULE[spec.module_name]

    assert issubclass(cls, TorchDisaggregator)
    model = cls({"device": "cpu"})

    assert model.sequence_length == expected.sequence_length
    assert model.n_epochs == expected.n_epochs
    assert model.batch_size == expected.batch_size
    assert model.mains_mean == expected.mains_mean
    assert model.mains_std == expected.mains_std
    assert model.appliance_params == {}
    assert model.models == {}
    assert model.chunk_wise_training is False
    assert model.seed is None
    assert model.verbose is False
    assert str(model.device) == "cpu"


@pytest.mark.parametrize(
    "spec",
    TORCH_MODEL_SPECS,
    ids=lambda spec: f"{spec.module_name}.{spec.class_name}",
)
def test_every_torch_model_accepts_none_and_rejects_non_mapping_params(spec):
    cls = _load_class(spec)

    assert cls().models == {}
    with pytest.raises(TypeError, match="params"):
        cls([])


@pytest.mark.parametrize(
    "spec",
    TORCH_MODEL_SPECS,
    ids=lambda spec: f"{spec.module_name}.{spec.class_name}",
)
def test_every_torch_model_honors_validated_common_overrides(spec):
    cls = _load_class(spec)
    supplied_stats = {
        "fridge": {
            "mean": 42.0,
            "std": 3.0,
            "metadata": {"source": "caller"},
        }
    }
    model = cls(
        {
            "device": "cpu",
            "sequence_length": 101,
            "n_epochs": 3,
            "batch_size": 7,
            "mains_mean": 123.0,
            "mains_std": 45.0,
            "appliance_params": supplied_stats,
            "chunk_wise_training": True,
            "seed": 17,
            "verbose": True,
        }
    )

    supplied_stats["fridge"]["metadata"]["source"] = "mutated"
    assert model.sequence_length == 101
    assert model.n_epochs == 3
    assert model.batch_size == 7
    assert model.mains_mean == 123.0
    assert model.mains_std == 45.0
    assert model.chunk_wise_training is True
    assert model.seed == 17
    assert model.verbose is True
    assert model.appliance_params["fridge"]["metadata"] == {"source": "caller"}


@pytest.mark.parametrize(
    "spec",
    TORCH_MODEL_SPECS,
    ids=lambda spec: f"{spec.module_name}.{spec.class_name}",
)
def test_every_torch_model_inherits_shared_statistics_policy(spec):
    from nilmtk_contrib.torch._base import TorchDisaggregator

    cls = _load_class(spec)

    assert cls.set_appliance_params is TorchDisaggregator.set_appliance_params
    assert cls.APPLIANCE_STD_FLOOR == 1.0
    assert cls.APPLIANCE_STD_FALLBACK == 100.0
    assert cls.INCLUDE_APPLIANCE_EXTREMA is (spec.module_name in EXTREMA_MODULES)
    if spec.module_name in NO_STATISTICS_MODULES:
        assert cls.REQUIRED_APPLIANCE_STATS == ()
    elif spec.module_name in MINMAX_MODULES:
        assert cls.REQUIRED_APPLIANCE_STATS == ("mean", "std", "min", "max")
    else:
        assert cls.REQUIRED_APPLIANCE_STATS == ("mean", "std")


@pytest.mark.parametrize(
    "spec",
    TORCH_MODEL_SPECS,
    ids=lambda spec: f"{spec.module_name}.{spec.class_name}",
)
def test_torch_model_source_does_not_reintroduce_shared_boilerplate(spec):
    module = importlib.import_module(spec.module_name)
    source_path = Path(module.__file__)
    source = source_path.read_text(encoding="utf-8")

    forbidden = {
        "direct Disaggregator inheritance": rf"class\s+{spec.class_name}\(Disaggregator\)",
        "direct Disaggregator import": r"from nilmtk\.disaggregate import Disaggregator",
        "duplicated runtime initialization": r"\binitialize_runtime\(",
        "duplicated common parameter normalization": r"\bnormalize_common_params\(",
        "duplicated appliance statistics": r"def set_appliance_params\(",
        "module-global device selection": r"(?m)^device\s*=\s*torch\.device",
        "unsafe model-map aliasing": r"self\.models\s*=\s*model\b",
    }
    for label, pattern in forbidden.items():
        assert (
            re.search(pattern, source) is None
        ), f"{label} remains in {source_path.name}"

    parameters = inspect.signature(getattr(module, spec.class_name).disaggregate_chunk)
    method_source = inspect.getsource(
        getattr(module, spec.class_name).disaggregate_chunk
    )
    if "model" in parameters.parameters:
        assert "self.require_models(model)" in method_source
    else:
        assert "self.require_models()" in method_source


@pytest.mark.parametrize(
    "spec",
    TORCH_MODEL_SPECS,
    ids=lambda spec: f"{spec.module_name}.{spec.class_name}",
)
def test_every_torch_disaggregation_path_validates_and_detaches_models(spec):
    torch = pytest.importorskip("torch")
    cls = _load_class(spec)
    normalization = {"fridge": {"mean": 95.0, "std": 40.0, "min": 0.0, "max": 200.0}}
    model = cls({"device": "cpu", "appliance_params": normalization})
    parameters = inspect.signature(model.disaggregate_chunk).parameters

    with pytest.raises(RuntimeError, match="trained or loaded"):
        model.disaggregate_chunk([], do_preprocessing=False)

    external = {"fridge": torch.nn.Identity()}
    if cls.REQUIRED_APPLIANCE_STATS:
        missing_stats = cls({"device": "cpu"})
        with pytest.raises(ValueError, match="normalization parameters"):
            if "model" in parameters:
                missing_stats.disaggregate_chunk(
                    [], model=external, do_preprocessing=False
                )
            else:
                missing_stats.models.update(external)
                missing_stats.disaggregate_chunk([], do_preprocessing=False)

    if "model" in parameters:
        assert (
            model.disaggregate_chunk([], model=external, do_preprocessing=False) == []
        )
        assert model.models is not external
    else:
        model.models.update(external)
        assert model.disaggregate_chunk([], do_preprocessing=False) == []

    external["kettle"] = torch.nn.Identity()
    assert list(model.models) == ["fridge"]


def test_external_model_runs_nonempty_seq2point_inference_with_validated_state():
    torch = pytest.importorskip("torch")
    from nilmtk_contrib.torch.seq2point import Seq2PointTorch

    class ZeroPoint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))

        def forward(self, values):
            return self.bias.expand(values.shape[0], 1)

    model = Seq2PointTorch(
        {
            "device": "cpu",
            "appliance_params": {"fridge": {"mean": 95.0, "std": 40.0}},
        }
    )
    mains = pd.DataFrame({"power": np.full(99, 140.0, dtype=np.float32)})

    predictions = model.disaggregate_chunk([mains], model={"fridge": ZeroPoint()})

    assert len(predictions) == 1
    assert len(predictions[0]) == len(mains)
    assert np.isfinite(predictions[0]["fridge"]).all()
    assert predictions[0]["fridge"].to_numpy() == pytest.approx(95.0)


@pytest.mark.parametrize("module_name", MINMAX_MODULES)
def test_classification_preprocessing_is_finite_for_all_off_target(module_name):
    module = importlib.import_module(module_name)
    cls = getattr(
        module,
        (
            "ResNet_classification"
            if module_name.endswith("resnet_classification")
            else "RNN_attention_classification"
        ),
    )
    model = cls({"device": "cpu"})
    all_off = pd.DataFrame({"power": np.zeros(99, dtype=np.float32)})
    model.set_appliance_params([("fridge", [all_off])])

    _, processed_appliances = model.call_preprocessing(
        [all_off], [("fridge", [all_off])], "train"
    )
    normalized = processed_appliances[0][1][0].to_numpy()

    assert np.isfinite(normalized).all()
    assert normalized == pytest.approx(0.0)


@pytest.mark.parametrize(
    "module_name",
    (
        "nilmtk_contrib.torch.msdc",
        "nilmtk_contrib.torch.msdc_without_crf",
    ),
)
def test_msdc_preserves_dataset_specific_defaults_and_common_overrides(module_name):
    cls = getattr(importlib.import_module(module_name), "MSDC")

    redd = cls({"device": "cpu", "dataset": " REDD "})
    assert redd.dataset == "redd"
    assert redd.mains_mean == pytest.approx(352.32)
    assert redd.mains_std == pytest.approx(608.42)

    overridden = cls(
        {
            "device": "cpu",
            "dataset": "redd",
            "mains_mean": 1.0,
            "mains_std": 2.0,
        }
    )
    assert overridden.mains_mean == 1.0
    assert overridden.mains_std == 2.0

    with pytest.raises(ValueError, match="dataset"):
        cls({"device": "cpu", "dataset": " "})


@pytest.mark.parametrize(
    ("module_name", "class_name", "parameter", "value", "attribute"),
    (
        ("nilmtk_contrib.torch.TCN", "TCN", "num_levels", 2, "num_levels"),
        ("nilmtk_contrib.torch.msdc", "MSDC", "num_states", 4, "num_states"),
        (
            "nilmtk_contrib.torch.msdc_without_crf",
            "MSDC",
            "out_len",
            32,
            "out_len",
        ),
        (
            "nilmtk_contrib.torch.nilmformer",
            "NILMFormer",
            "d_model",
            48,
            "d_model",
        ),
        ("nilmtk_contrib.torch.reformer", "Reformer", "dim", 128, "dim"),
        (
            "nilmtk_contrib.torch.resnet_classification",
            "ResNet_classification",
            "classification_threshold",
            25.0,
            "classification_threshold",
        ),
        (
            "nilmtk_contrib.torch.rnn_attention_classification",
            "RNN_attention_classification",
            "classification_loss_weight",
            2.0,
            "classification_loss_weight",
        ),
    ),
)
def test_model_specific_params_remain_owned_by_subclasses(
    module_name, class_name, parameter, value, attribute
):
    cls = getattr(importlib.import_module(module_name), class_name)

    model = cls({"device": "cpu", parameter: value})

    assert getattr(model, attribute) == value


def test_migration_contract_covers_every_registered_torch_model():
    assert set(DEFAULTS_BY_MODULE) == {spec.module_name for spec in TORCH_MODEL_SPECS}
