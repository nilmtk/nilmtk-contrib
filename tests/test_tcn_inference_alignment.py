import math

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
tcn_module = pytest.importorskip("nilmtk_contrib.torch.TCN")
TCN = tcn_module.TCN


class RecordingPoint(torch.nn.Module):
    def __init__(self, value=0.0):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(float(value)))
        self.batch_sizes = []

    def forward(self, values):
        self.batch_sizes.append(len(values))
        return torch.ones((len(values), 1), device=values.device) * self.bias


class BadShape(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, values):
        return torch.ones((len(values),), device=values.device) + self.bias


class NonFinitePoint(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(float("nan")))

    def forward(self, values):
        return torch.ones((len(values), 1), device=values.device) * self.bias


class NonTensorPoint(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, values):
        return np.zeros((len(values), 1), dtype=np.float32)


def _model(sequence_length=99, batch_size=512, appliance_params=None):
    if appliance_params is None:
        appliance_params = {"fridge": {"mean": 11.0, "std": 2.0}}
    return TCN(
        {
            "device": "cpu",
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "num_levels": 1,
            "num_filters": 2,
            "kernel_size": 3,
            "dropout": 0.0,
            "appliance_params": appliance_params,
        }
    )


def _frame(length, start=0):
    index = pd.Index(
        np.arange(start, start + 3 * length, 3, dtype=np.int64), name="sample"
    )
    values = np.linspace(20.0, 40.0, length, dtype=np.float32)
    return pd.DataFrame({"power": values}, index=index)


@pytest.mark.parametrize("length", [0, 1, 98, 99, 100, 2448])
def test_raw_inference_preserves_every_length_and_original_index(length):
    model = _model()
    mains = _frame(length, start=1000)

    prediction = model.disaggregate_chunk(
        [mains], model={"fridge": RecordingPoint(0.5)}
    )[0]

    assert len(prediction) == length
    assert prediction.index.equals(mains.index)
    assert prediction.columns.tolist() == ["fridge"]
    assert prediction.dtypes.tolist() == [np.dtype("float32")]
    assert np.isfinite(prediction.to_numpy()).all()
    np.testing.assert_array_equal(
        prediction["fridge"].to_numpy(), np.full(length, 12.0, dtype=np.float32)
    )


def test_multiple_chunks_preserve_independent_indexes_and_partial_batches():
    lengths = [1, 100, 2448]
    mains = [
        _frame(length, start=position * 10_000)
        for position, length in enumerate(lengths)
    ]
    model = _model(batch_size=512)
    network = RecordingPoint()

    predictions = model.disaggregate_chunk(mains, model={"fridge": network})

    assert network.batch_sizes == [1, 100, 512, 512, 512, 512, 400]
    for source, prediction in zip(mains, predictions, strict=True):
        assert len(prediction) == len(source)
        assert prediction.index.equals(source.index)


def test_preprocessed_inference_keeps_one_point_per_window_and_its_index():
    model = _model(sequence_length=7, batch_size=2)
    index = pd.Index(["a", "c", "f"], name="window")
    windows = pd.DataFrame(np.zeros((3, 7), dtype=np.float32), index=index)

    prediction = model.disaggregate_chunk(
        [windows], do_preprocessing=False, model={"fridge": RecordingPoint()}
    )[0]

    assert len(prediction) == len(windows)
    assert prediction.index.equals(index)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_raw_inference_rejects_nonfinite_mains(bad_value):
    mains = _frame(8)
    mains.iloc[3, 0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        _model(sequence_length=7).disaggregate_chunk(
            [mains], model={"fridge": RecordingPoint()}
        )


def test_raw_inference_rejects_ambiguous_multiple_power_columns():
    mains = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    with pytest.raises(ValueError, match="one power column"):
        _model(sequence_length=7).disaggregate_chunk(
            [mains], model={"fridge": RecordingPoint()}
        )


def test_inference_rejects_invalid_model_outputs():
    mains = _frame(8)
    model = _model(sequence_length=7)

    with pytest.raises(ValueError, match="returned shape"):
        model.disaggregate_chunk([mains], model={"fridge": BadShape()})
    with pytest.raises(FloatingPointError, match="non-finite"):
        model.disaggregate_chunk([mains], model={"fridge": NonFinitePoint()})
    with pytest.raises(TypeError, match="torch.Tensor"):
        model.disaggregate_chunk([mains], model={"fridge": NonTensorPoint()})


def test_public_float32_output_overflow_is_rejected():
    model = _model(
        sequence_length=7,
        appliance_params={"fridge": {"mean": 0.0, "std": 1e300}},
    )

    with pytest.raises(ValueError, match="overflow float32"):
        model.disaggregate_chunk([_frame(1)], model={"fridge": RecordingPoint(1.0)})


def test_real_tcn_network_preserves_2448_row_chunk_alignment():
    model = _model()
    mains = _frame(2448, start=50_000)

    prediction = model.disaggregate_chunk(
        [mains], model={"fridge": model.return_network()}
    )[0]

    assert len(prediction) == len(mains)
    assert prediction.index.equals(mains.index)
    assert prediction.dtypes.tolist() == [np.dtype("float32")]
    assert np.isfinite(prediction.to_numpy()).all()
