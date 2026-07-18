from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
WindowGRU = pytest.importorskip("nilmtk_contrib.torch.WindowGRU").WindowGRU
torch_base = pytest.importorskip("nilmtk_contrib.torch._base")
TorchDisaggregator = torch_base.TorchDisaggregator
torch_defaults = torch_base.torch_defaults
BERT = pytest.importorskip("nilmtk_contrib.torch.bert").BERT
ConvLSTM = pytest.importorskip("nilmtk_contrib.torch.conv_lstm").ConvLSTM
MSDC = pytest.importorskip("nilmtk_contrib.torch.msdc").MSDC
MSDCWithoutCRF = pytest.importorskip(
    "nilmtk_contrib.torch.msdc_without_crf"
).MSDC
NILMFormer = pytest.importorskip("nilmtk_contrib.torch.nilmformer").NILMFormer
Reformer = pytest.importorskip("nilmtk_contrib.torch.reformer").Reformer
ResNet = pytest.importorskip("nilmtk_contrib.torch.resnet").ResNet
ResNet_classification = pytest.importorskip(
    "nilmtk_contrib.torch.resnet_classification"
).ResNet_classification
RNN = pytest.importorskip("nilmtk_contrib.torch.rnn").RNN
RNN_attention = pytest.importorskip("nilmtk_contrib.torch.rnn_attention").RNN_attention
RNN_attention_classification = pytest.importorskip(
    "nilmtk_contrib.torch.rnn_attention_classification"
).RNN_attention_classification
Seq2PointTorch = pytest.importorskip(
    "nilmtk_contrib.torch.seq2point"
).Seq2PointTorch
Seq2Seq = pytest.importorskip("nilmtk_contrib.torch.seq2seq").Seq2Seq


SEQUENCE_LENGTH = 3
RAW_LENGTH = 7
APPLIANCE_PARAMS = {
    "fridge": {"mean": 10.0, "std": 2.0, "min": 0.0, "max": 20.0}
}


class FakeNetwork(torch.nn.Module):
    """Cheap deterministic outputs matching the legacy adapters' signatures."""

    def __init__(
        self, mode, sequence_length=SEQUENCE_LENGTH, num_states=2, value=0.0
    ):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor([value], dtype=torch.float32))
        self.mode = mode
        self.sequence_length = sequence_length
        self.num_states = num_states
        if mode == "msdc":
            self.crf = FakeCRF()

    def forward(self, values):
        batch_size = len(values)
        point = torch.zeros((batch_size, 1), device=values.device) + self.bias
        sequence = torch.zeros(
            (batch_size, self.sequence_length), device=values.device
        ) + self.bias
        if self.mode == "point":
            return point
        if self.mode == "sequence":
            return sequence
        if self.mode == "resnet_classification":
            return sequence, sequence
        if self.mode == "rnn_attention_classification":
            return sequence, sequence, sequence
        if self.mode == "nilmformer":
            return sequence.unsqueeze(1)
        if self.mode == "msdc":
            shape = (batch_size, self.sequence_length, self.num_states)
            emissions = torch.zeros(shape, device=values.device) + self.bias
            power = torch.zeros(shape, device=values.device) + self.bias
            return emissions, power
        if self.mode == "msdc_without_crf":
            shape = (batch_size, self.sequence_length, self.num_states)
            power = torch.zeros(shape, device=values.device) + self.bias
            states = torch.zeros(shape, device=values.device) + self.bias
            return power, states
        raise AssertionError(f"Unknown fake-network mode {self.mode!r}.")


class FakeCRF:
    @staticmethod
    def viterbi_decode(emissions):
        return torch.zeros(emissions.shape[:2], dtype=torch.long, device=emissions.device)


MODEL_CASES = [
    pytest.param(BERT, "sequence", id="bert"),
    pytest.param(ConvLSTM, "point", id="conv-lstm"),
    pytest.param(MSDC, "msdc", id="msdc"),
    pytest.param(
        MSDCWithoutCRF, "msdc_without_crf", id="msdc-without-crf"
    ),
    pytest.param(NILMFormer, "nilmformer", id="nilmformer"),
    pytest.param(Reformer, "point", id="reformer"),
    pytest.param(ResNet, "sequence", id="resnet"),
    pytest.param(
        ResNet_classification,
        "resnet_classification",
        id="resnet-classification",
    ),
    pytest.param(RNN, "point", id="rnn"),
    pytest.param(RNN_attention, "point", id="rnn-attention"),
    pytest.param(
        RNN_attention_classification,
        "rnn_attention_classification",
        id="rnn-attention-classification",
    ),
    pytest.param(Seq2PointTorch, "point", id="seq2point"),
    pytest.param(Seq2Seq, "sequence", id="seq2seq"),
    pytest.param(WindowGRU, "point", id="window-gru"),
]
ROW_COUNT_CASES = MODEL_CASES


def _model(model_class, mode, *, value=0.0):
    params = {
        "device": "cpu",
        "sequence_length": SEQUENCE_LENGTH,
        "batch_size": 2,
        "appliance_params": APPLIANCE_PARAMS,
    }
    if model_class is MSDCWithoutCRF:
        params["out_len"] = SEQUENCE_LENGTH
        params["num_states"] = 2
    model = model_class(params)
    if model_class in {MSDC, MSDCWithoutCRF}:
        model.num_states = 2
        model._get_appliance_config = lambda appliance: None
    model.models = OrderedDict(
        fridge=FakeNetwork(
            mode,
            num_states=getattr(model, "num_states", 2),
            value=value,
        )
    )
    return model


def _raw_chunk(length=RAW_LENGTH):
    index = pd.Index(
        np.arange(100, 100 + 5 * length, 5, dtype=np.int64), name="sample"
    )
    return pd.DataFrame(
        {"power": np.linspace(40.0, 60.0, length, dtype=np.float32)},
        index=index,
    )


def _install_fake_preprocessor(model, mode, *, row_delta=0):
    sequence_output = mode in {
        "sequence",
        "resnet_classification",
        "rnn_attention_classification",
    }

    def preprocess(chunks, submeters_lst, method):
        assert submeters_lst is None
        assert method == "test"
        rows = []
        for chunk in chunks:
            count = len(chunk)
            if sequence_output:
                count -= model.sequence_length - 1
            count += row_delta
            rows.append(
                pd.DataFrame(
                    np.zeros((count, model.sequence_length), dtype=np.float32)
                )
            )
        return rows

    model.call_preprocessing = preprocess


@pytest.mark.parametrize("model_class,mode", MODEL_CASES)
def test_raw_partial_inference_preserves_exact_non_range_index(model_class, mode):
    model = _model(model_class, mode)
    _install_fake_preprocessor(model, mode)
    mains = _raw_chunk()

    result = model.disaggregate_chunk([mains])[0]

    assert len(result) == len(mains)
    assert result.index.equals(mains.index)
    assert result.columns.tolist() == ["fridge"]
    assert result.dtypes.tolist() == [np.dtype("float32")]
    assert np.isfinite(result.to_numpy()).all()


@pytest.mark.parametrize("model_class,mode", MODEL_CASES)
def test_real_preprocessor_preserves_hostile_2448_row_timezone_index(
    model_class, mode
):
    model = _model(model_class, mode)
    model.batch_size = 512
    index = pd.date_range(
        "2026-01-01 00:00:00",
        periods=2448,
        freq="6s",
        tz="Asia/Kolkata",
        name="timestamp",
    )
    mains = pd.DataFrame(
        {"power": np.linspace(40.0, 80.0, len(index), dtype=np.float32)},
        index=index,
    )

    result = model.disaggregate_chunk([mains])[0]

    assert len(result) == len(mains)
    assert result.index.equals(mains.index)
    assert result.index is not mains.index
    assert result.dtypes.tolist() == [np.dtype("float32")]
    assert np.isfinite(result.to_numpy()).all()


@pytest.mark.parametrize("model_class,mode", ROW_COUNT_CASES)
def test_raw_inference_fails_closed_when_model_returns_wrong_row_count(
    model_class, mode
):
    model = _model(model_class, mode)
    _install_fake_preprocessor(model, mode, row_delta=-1)

    with pytest.raises(ValueError, match="(expected 7 raw rows|for 7 raw rows)"):
        model.disaggregate_chunk([_raw_chunk()])


class MinimalDisaggregator(TorchDisaggregator):
    def __init__(self):
        super().__init__({"device": "cpu"}, defaults=torch_defaults())


def test_helper_fails_closed_on_invalid_raw_chunks_and_preprocessor_chunk_count():
    model = MinimalDisaggregator()
    model.call_preprocessing = lambda chunks, submeters_lst, method: []

    with pytest.raises(TypeError, match="iterable of pandas objects"):
        model.preprocess_raw_inference_chunks(pd.DataFrame({"power": [1.0]}))
    with pytest.raises(TypeError, match="chunk 0"):
        model.preprocess_raw_inference_chunks([np.zeros(3)])
    with pytest.raises(TypeError, match="chunk 0"):
        model.preprocess_raw_inference_chunks([pd.Series([1.0])])
    with pytest.raises(ValueError, match="one power column"):
        model.preprocess_raw_inference_chunks(
            [pd.DataFrame({"a": [1.0], "b": [2.0]})]
        )
    with pytest.raises(ValueError, match="finite"):
        model.preprocess_raw_inference_chunks(
            [pd.DataFrame({"power": [np.nan]})]
        )
    with pytest.raises(ValueError, match="at least one sample"):
        model.preprocess_raw_inference_chunks(
            [pd.DataFrame({"power": pd.Series(dtype="float32")})]
        )
    with pytest.raises(ValueError, match="expected 1, got 0"):
        model.preprocess_raw_inference_chunks([_raw_chunk(1)])


@pytest.mark.parametrize(
    "values,error,message",
    [
        (["bad"], TypeError, "real numeric"),
        ([True], TypeError, "real numeric"),
        ([1 + 2j], TypeError, "real numeric"),
        ([np.nan], ValueError, "non-finite"),
        ([np.inf], ValueError, "non-finite"),
    ],
)
def test_helper_rejects_invalid_prediction_values(values, error, message):
    prediction = pd.DataFrame({"fridge": values})

    with pytest.raises(error, match=message):
        TorchDisaggregator.align_raw_inference_predictions(
            [prediction], (pd.Index([17]),)
        )


def test_helper_fails_closed_on_prediction_chunk_and_row_count():
    index = pd.Index([10, 20])

    with pytest.raises(ValueError, match="expected 1, got 0"):
        TorchDisaggregator.align_raw_inference_predictions([], (index,))
    with pytest.raises(ValueError, match="has 1 rows; expected 2"):
        TorchDisaggregator.align_raw_inference_predictions(
            [pd.DataFrame({"fridge": [1.0]})], (index,)
        )


@pytest.mark.parametrize("model_class,mode", MODEL_CASES)
def test_preprocessed_inference_does_not_apply_raw_alignment(model_class, mode):
    model = _model(model_class, mode)
    model.preprocess_raw_inference_chunks = lambda chunks: pytest.fail(
        "raw alignment must not run for preprocessed inference"
    )
    if mode in {
        "sequence",
        "resnet_classification",
        "rnn_attention_classification",
    }:
        rows = RAW_LENGTH - SEQUENCE_LENGTH + 1
        windows = pd.DataFrame(
            np.zeros((rows, SEQUENCE_LENGTH), dtype=np.float32),
            index=pd.Index(np.arange(50, 50 + rows), name="window"),
        )
    else:
        windows = pd.DataFrame(
            np.zeros((RAW_LENGTH, SEQUENCE_LENGTH), dtype=np.float32),
            index=pd.Index(np.arange(50, 50 + RAW_LENGTH), name="window"),
        )

    result = model.disaggregate_chunk([windows], do_preprocessing=False)[0]

    expected_length = len(windows) if mode == "nilmformer" else RAW_LENGTH
    assert len(result) == expected_length
    if model_class in {MSDC, MSDCWithoutCRF}:
        assert result.index.equals(windows.index)
    else:
        assert isinstance(result.index, pd.RangeIndex)


@pytest.mark.parametrize("model_class,mode", MODEL_CASES)
def test_preprocessed_inference_rejects_nonfinite_predictions(model_class, mode):
    model = _model(model_class, mode, value=float("inf"))
    if mode in {
        "sequence",
        "resnet_classification",
        "rnn_attention_classification",
    }:
        rows = RAW_LENGTH - SEQUENCE_LENGTH + 1
    else:
        rows = RAW_LENGTH
    windows = pd.DataFrame(
        np.zeros((rows, SEQUENCE_LENGTH), dtype=np.float32)
    )

    with pytest.raises((FloatingPointError, ValueError), match="non-finite"):
        model.disaggregate_chunk([windows], do_preprocessing=False)
