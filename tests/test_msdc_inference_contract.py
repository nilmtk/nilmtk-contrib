import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")


class _ViterbiStub:
    @staticmethod
    def viterbi_decode(emissions):
        return emissions.argmax(dim=-1)


class _CRFNetworkStub(torch.nn.Module):
    def __init__(self, num_states=3):
        super().__init__()
        self.num_states = num_states
        self.crf = _ViterbiStub()

    def forward(self, inputs):
        batch_size, sequence_length, _ = inputs.shape
        shape = (batch_size, sequence_length, self.num_states)
        return torch.zeros(shape), torch.ones(shape)


class _DirectNetworkStub(torch.nn.Module):
    def __init__(self, out_len=64, num_states=3):
        super().__init__()
        self.out_len = out_len
        self.num_states = num_states

    def forward(self, inputs):
        batch_size = len(inputs)
        shape = (batch_size, self.out_len * self.num_states)
        return torch.ones(shape), torch.zeros(shape)


@pytest.mark.parametrize(
    ("module_name", "network_factory"),
    [
        ("nilmtk_contrib.torch.msdc", _CRFNetworkStub),
        ("nilmtk_contrib.torch.msdc_without_crf", _DirectNetworkStub),
    ],
)
def test_msdc_prediction_preserves_each_raw_chunk_index(module_name, network_factory):
    import importlib

    model_class = importlib.import_module(module_name).MSDC
    model = model_class(
        {
            "device": "cpu",
            "sequence_length": 19,
            "appliance_params": {"fridge": {"mean": 100.0, "std": 20.0}},
        }
    )
    indexes = [
        pd.date_range(
            "2026-01-01",
            periods=23,
            freq="15min",
            tz="US/Eastern",
            name="timestamp",
        ),
        pd.Index(["a", "b", "c"], name="sample"),
    ]
    mains = [
        pd.DataFrame({"power": np.linspace(100.0, 200.0, len(index))}, index=index)
        for index in indexes
    ]

    predictions = model.disaggregate_chunk(mains, model={"fridge": network_factory()})

    assert len(predictions) == len(mains)
    for prediction, source in zip(predictions, mains, strict=True):
        assert prediction.index.equals(source.index)
        assert prediction.index is not source.index
        assert prediction.columns.tolist() == ["fridge"]
        assert prediction.dtypes.tolist() == [np.dtype("float32")]
        assert np.isfinite(prediction.to_numpy()).all()


@pytest.mark.parametrize(
    ("module_name", "network_factory"),
    [
        ("nilmtk_contrib.torch.msdc", _CRFNetworkStub),
        ("nilmtk_contrib.torch.msdc_without_crf", _DirectNetworkStub),
    ],
)
def test_msdc_prediction_preserves_preprocessed_chunk_index(
    module_name, network_factory
):
    import importlib

    model_class = importlib.import_module(module_name).MSDC
    model = model_class(
        {
            "device": "cpu",
            "sequence_length": 19,
            "appliance_params": {"fridge": {"mean": 100.0, "std": 20.0}},
        }
    )
    index = pd.Index([11, 13, 17], name="window")
    windows = pd.DataFrame(np.ones((3, 19)), index=index)

    prediction = model.disaggregate_chunk(
        [windows],
        model={"fridge": network_factory()},
        do_preprocessing=False,
    )[0]

    assert prediction.index.equals(index)
