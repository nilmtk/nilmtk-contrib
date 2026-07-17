import math

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
dae_module = pytest.importorskip("nilmtk_contrib.torch.dae")
DAE = dae_module.DAE


class RecordingSequence(torch.nn.Module):
    def __init__(self, value=0.0):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(float(value)))
        self.batch_sizes = []

    def forward(self, values):
        self.batch_sizes.append(len(values))
        return torch.ones_like(values) * self.bias


class EchoSequence(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, values):
        return values + self.bias


class BadShape(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, values):
        return values.squeeze(-1) + self.bias


class NonFiniteSequence(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(float("nan")))

    def forward(self, values):
        return torch.ones_like(values) * self.bias


class NonTensorSequence(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, values):
        return values.detach().cpu().numpy()


class WrongOutputDevice(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, values):
        return torch.empty(values.shape, device="meta")


def _model(sequence_length=99, batch_size=4, appliance_params=None):
    if appliance_params is None:
        appliance_params = {"fridge": {"mean": 11.0, "std": 2.0}}
    return DAE(
        {
            "device": "cpu",
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "appliance_params": appliance_params,
        }
    )


def _frame(length, start=0):
    index = pd.Index(
        np.arange(start, start + 2 * length, 2, dtype=np.int64), name="sample"
    )
    values = np.linspace(20.0, 40.0, length, dtype=np.float32)
    return pd.DataFrame({"power": values}, index=index)


@pytest.mark.parametrize("length", [0, 1, 98, 99, 100, 1024])
def test_raw_inference_preserves_every_length_and_original_index(length):
    model = _model()
    mains = _frame(length, start=1000)

    prediction = model.disaggregate_chunk(
        [mains], model={"fridge": RecordingSequence(0.0)}
    )[0]

    assert len(prediction) == length
    assert prediction.index.equals(mains.index)
    assert prediction.columns.tolist() == ["fridge"]
    assert prediction.dtypes.tolist() == [np.dtype("float32")]
    np.testing.assert_array_equal(
        prediction["fridge"].to_numpy(), np.full(length, 11.0, dtype=np.float32)
    )


def test_multiple_chunks_each_preserve_their_own_length_and_index():
    lengths = [0, 1, 98, 99, 100, 2448]
    mains = [
        _frame(length, start=position * 10_000)
        for position, length in enumerate(lengths)
    ]
    model = _model()

    predictions = model.disaggregate_chunk(
        mains, model={"fridge": RecordingSequence(0.5)}
    )

    assert len(predictions) == len(mains)
    for source, prediction in zip(mains, predictions, strict=True):
        assert len(prediction) == len(source)
        assert prediction.index.equals(source.index)
        assert np.isfinite(prediction.to_numpy()).all()
        np.testing.assert_array_equal(
            prediction["fridge"].to_numpy(),
            np.full(len(source), 12.0, dtype=np.float32),
        )


def test_partial_final_batch_is_run_but_padding_is_trimmed():
    model = _model(sequence_length=99, batch_size=4)
    network = RecordingSequence(0.0)
    mains = _frame(1024)

    prediction = model.disaggregate_chunk([mains], model={"fridge": network})[0]

    assert network.batch_sizes == [4, 4, 3]
    assert len(prediction) == 1024
    assert prediction.index.equals(mains.index)


def test_trimmed_predictions_remain_row_aligned_not_shifted():
    model = DAE(
        {
            "device": "cpu",
            "sequence_length": 99,
            "batch_size": 4,
            "mains_mean": 0.0,
            "mains_std": 1.0,
            "appliance_params": {"fridge": {"mean": 0.0, "std": 1.0}},
        }
    )
    mains = _frame(100, start=500)

    prediction = model.disaggregate_chunk([mains], model={"fridge": EchoSequence()})[0]

    assert prediction.index.equals(mains.index)
    np.testing.assert_array_equal(
        prediction["fridge"].to_numpy(), mains["power"].to_numpy()
    )


def test_external_model_mapping_is_validated_and_detached():
    model = _model(sequence_length=7, batch_size=2)
    mains = _frame(8)
    external = {"fridge": RecordingSequence(0.0)}

    prediction = model.disaggregate_chunk([mains], model=external)[0]
    external["kettle"] = RecordingSequence(0.0)

    assert len(prediction) == 8
    assert model.models is not external
    assert list(model.models) == ["fridge"]


def test_real_dae_network_preserves_a_partial_raw_chunk():
    model = _model(sequence_length=7, batch_size=2)
    mains = _frame(8)

    prediction = model.disaggregate_chunk(
        [mains], model={"fridge": model.return_network()}
    )[0]

    assert len(prediction) == len(mains)
    assert prediction.index.equals(mains.index)
    assert np.isfinite(prediction.to_numpy()).all()


def test_inference_requires_a_model_and_matching_normalization():
    mains = _frame(1)
    with pytest.raises(RuntimeError, match="trained or loaded"):
        _model().disaggregate_chunk([mains])

    with pytest.raises(ValueError, match="normalization parameters"):
        _model(appliance_params={}).disaggregate_chunk(
            [mains], model={"fridge": RecordingSequence()}
        )

    with pytest.raises(TypeError, match="torch.nn.Module"):
        _model().disaggregate_chunk([mains], model={"fridge": object()})


def test_inference_rejects_model_on_an_incompatible_device():
    model = _model()
    meta_model = torch.nn.Linear(1, 1, device="meta")

    with pytest.raises(ValueError, match="incompatible device"):
        model.disaggregate_chunk([_frame(1)], model={"fridge": meta_model})


def test_inference_rejects_invalid_external_model_outputs():
    model = _model(sequence_length=7)
    mains = _frame(8)

    with pytest.raises(ValueError, match="returned shape"):
        model.disaggregate_chunk([mains], model={"fridge": BadShape()})
    with pytest.raises(RuntimeError, match="non-finite"):
        model.disaggregate_chunk([mains], model={"fridge": NonFiniteSequence()})
    with pytest.raises(TypeError, match="torch.Tensor"):
        model.disaggregate_chunk([mains], model={"fridge": NonTensorSequence()})
    with pytest.raises(ValueError, match="returned values on"):
        model.disaggregate_chunk([mains], model={"fridge": WrongOutputDevice()})


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_raw_inference_rejects_nonfinite_mains(bad_value):
    model = _model(sequence_length=7)
    mains = _frame(8)
    mains.iloc[3, 0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        model.disaggregate_chunk([mains], model={"fridge": RecordingSequence()})


def test_raw_inference_rejects_ambiguous_multiple_power_columns():
    mains = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    with pytest.raises(ValueError, match="one power column"):
        _model(sequence_length=7).disaggregate_chunk(
            [mains], model={"fridge": RecordingSequence()}
        )


def test_public_float32_output_overflow_is_rejected():
    model = _model(
        sequence_length=7,
        appliance_params={"fridge": {"mean": 0.0, "std": 1e300}},
    )

    with pytest.raises(ValueError, match="overflow float32"):
        model.disaggregate_chunk([_frame(1)], model={"fridge": RecordingSequence(1.0)})


def test_preprocessed_inference_retains_legacy_window_semantics():
    model = _model(sequence_length=7, batch_size=2)
    windows = pd.DataFrame(np.zeros((3, 7), dtype=np.float32))

    prediction = model.disaggregate_chunk(
        [windows],
        do_preprocessing=False,
        model={"fridge": RecordingSequence(0.0)},
    )[0]

    assert len(prediction) == 21
    assert prediction.index.equals(pd.RangeIndex(21))
