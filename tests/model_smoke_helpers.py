import importlib
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@dataclass(frozen=True)
class ModelSpec:
    backend: str
    package: str
    class_name: str
    module_name: str
    min_sequence_length: int = 99
    trainable: bool = True
    real_dataset: bool = True


DISAGGREGATE_MODEL_SPECS = (
    ModelSpec("classical", "nilmtk_contrib.disaggregate", "AFHMM", "nilmtk_contrib.disaggregate.afhmm", trainable=False),
    ModelSpec("classical", "nilmtk_contrib.disaggregate", "AFHMM_SAC", "nilmtk_contrib.disaggregate.afhmm_sac", trainable=False),
    ModelSpec("classical", "nilmtk_contrib.disaggregate", "DSC", "nilmtk_contrib.disaggregate.dsc", trainable=False),
    ModelSpec("tensorflow", "nilmtk_contrib.disaggregate", "BERT", "nilmtk_contrib.disaggregate.bert"),
    ModelSpec("tensorflow", "nilmtk_contrib.disaggregate", "DAE", "nilmtk_contrib.disaggregate.dae"),
    ModelSpec("tensorflow", "nilmtk_contrib.disaggregate", "RNN", "nilmtk_contrib.disaggregate.rnn"),
    ModelSpec("tensorflow", "nilmtk_contrib.disaggregate", "RNN_attention", "nilmtk_contrib.disaggregate.rnn_attention"),
    ModelSpec("tensorflow", "nilmtk_contrib.disaggregate", "RNN_attention_classification", "nilmtk_contrib.disaggregate.rnn_attention_classification"),
    ModelSpec("tensorflow", "nilmtk_contrib.disaggregate", "ResNet", "nilmtk_contrib.disaggregate.resnet"),
    ModelSpec("tensorflow", "nilmtk_contrib.disaggregate", "ResNet_classification", "nilmtk_contrib.disaggregate.resnet_classification"),
    ModelSpec("tensorflow", "nilmtk_contrib.disaggregate", "Seq2Point", "nilmtk_contrib.disaggregate.seq2point"),
    ModelSpec("tensorflow", "nilmtk_contrib.disaggregate", "Seq2Seq", "nilmtk_contrib.disaggregate.seq2seq"),
    ModelSpec("tensorflow", "nilmtk_contrib.disaggregate", "WindowGRU", "nilmtk_contrib.disaggregate.WindowGRU"),
)


TORCH_MODEL_SPECS = (
    ModelSpec("torch", "nilmtk_contrib.torch", "BERT", "nilmtk_contrib.torch.bert"),
    ModelSpec("torch", "nilmtk_contrib.torch", "ConvLSTM", "nilmtk_contrib.torch.conv_lstm"),
    ModelSpec("torch", "nilmtk_contrib.torch", "DAE", "nilmtk_contrib.torch.dae"),
    ModelSpec("torch", "nilmtk_contrib.torch", "MSDC", "nilmtk_contrib.torch.msdc", min_sequence_length=121),
    ModelSpec("torch", "nilmtk_contrib.torch", "MSDC", "nilmtk_contrib.torch.msdc_without_crf", min_sequence_length=121),
    ModelSpec("torch", "nilmtk_contrib.torch", "NILMFormer", "nilmtk_contrib.torch.nilmformer"),
    ModelSpec("torch", "nilmtk_contrib.torch", "Reformer", "nilmtk_contrib.torch.reformer"),
    ModelSpec("torch", "nilmtk_contrib.torch", "ResNet", "nilmtk_contrib.torch.resnet"),
    ModelSpec("torch", "nilmtk_contrib.torch", "ResNet_classification", "nilmtk_contrib.torch.resnet_classification"),
    ModelSpec("torch", "nilmtk_contrib.torch", "RNN", "nilmtk_contrib.torch.rnn"),
    ModelSpec("torch", "nilmtk_contrib.torch", "RNN_attention", "nilmtk_contrib.torch.rnn_attention"),
    ModelSpec("torch", "nilmtk_contrib.torch", "RNN_attention_classification", "nilmtk_contrib.torch.rnn_attention_classification"),
    ModelSpec("torch", "nilmtk_contrib.torch", "Seq2PointTorch", "nilmtk_contrib.torch.seq2point"),
    ModelSpec("torch", "nilmtk_contrib.torch", "Seq2Seq", "nilmtk_contrib.torch.seq2seq"),
    ModelSpec("torch", "nilmtk_contrib.torch", "TCN", "nilmtk_contrib.torch.TCN"),
    ModelSpec("torch", "nilmtk_contrib.torch", "WindowGRU", "nilmtk_contrib.torch.WindowGRU"),
)


ALL_MODEL_SPECS = DISAGGREGATE_MODEL_SPECS + TORCH_MODEL_SPECS


def import_or_skip(spec):
    required = {
        "classical": ("nilmtk", "cvxpy", "hmmlearn", "sklearn"),
        "tensorflow": ("nilmtk", "tensorflow"),
        "torch": ("nilmtk", "torch", "tqdm"),
    }[spec.backend]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(
            f"{spec.class_name} {spec.backend} smoke requires missing packages: "
            f"{', '.join(missing)}"
        )
    module = importlib.import_module(spec.module_name)
    return getattr(module, spec.class_name)


def synthetic_nilmtk_chunks(sequence_length, num_windows=4):
    total_length = sequence_length * num_windows
    t = np.arange(total_length, dtype=np.float32)
    fridge = 80.0 + 35.0 * ((t // max(1, sequence_length // 3)) % 2)
    kettle = np.where((t % sequence_length) < max(1, sequence_length // 8), 220.0, 0.0)
    mains = 40.0 + fridge + kettle + 5.0 * np.sin(t / 5.0)
    index = pd.date_range("2020-01-01", periods=total_length, freq="1min")
    mains_df = pd.DataFrame(mains, index=index, columns=["power"])
    fridge_df = pd.DataFrame(fridge, index=index, columns=["power"])
    return [mains_df], [("fridge", [fridge_df])]


def smoke_params(spec, epochs):
    sequence_length = spec.min_sequence_length
    return {
        "sequence_length": sequence_length,
        "n_epochs": epochs,
        "batch_size": 32,
        "mains_mean": 140.0,
        "mains_std": 80.0,
        "device": "cpu",
        "seed": 123,
        "verbose": False,
        "appliance_params": {
            "fridge": {
                "mean": 95.0,
                "std": 40.0,
                "min": 0.0,
                "max": 200.0,
                "on_power_threshold": 60.0,
                "threshold": 60.0,
            }
        },
        "classification_weight": 1.0,
        "regression_weight": 1.0,
    }


def assert_prediction_frames(predictions, appliance="fridge"):
    assert isinstance(predictions, list)
    assert predictions, "disaggregate_chunk should return at least one result chunk"
    for frame in predictions:
        assert isinstance(frame, pd.DataFrame)
        assert appliance in frame.columns
        values = frame[appliance].to_numpy(dtype=float)
        assert values.size > 0
        assert np.isfinite(values).all()


def load_real_dataset_chunks(path, building, appliance, sequence_length, max_samples=512):
    nilmtk = pytest.importorskip("nilmtk")
    dataset_path = Path(path)
    if not dataset_path.exists():
        pytest.skip(f"real dataset path does not exist: {dataset_path}")

    data_set = nilmtk.DataSet(str(dataset_path))
    try:
        elec = data_set.buildings[building].elec
        mains = elec.mains().power_series_all_data().dropna()
        appliance_meter = None
        for meter in elec.submeters().meters:
            for meter_appliance in getattr(meter, "appliances", []):
                if meter_appliance.identifier.type == appliance:
                    appliance_meter = meter
                    break
            if appliance_meter is not None:
                break
        if appliance_meter is None:
            pytest.skip(f"appliance {appliance!r} not found in building {building}")

        submeter = appliance_meter.power_series_all_data().dropna()
        joined = pd.concat([mains, submeter], axis=1).dropna().iloc[:max_samples]
        if len(joined) < sequence_length:
            pytest.skip(
                f"real dataset has {len(joined)} aligned samples; "
                f"need at least {sequence_length}"
            )

        windows = math.ceil(len(joined) / sequence_length)
        target_length = windows * sequence_length
        joined = joined.iloc[:target_length]
        joined.columns = ["mains", appliance]
        mains_df = pd.DataFrame(joined["mains"].to_numpy(), index=joined.index, columns=["power"])
        appliance_df = pd.DataFrame(joined[appliance].to_numpy(), index=joined.index, columns=["power"])
        return [mains_df], [(appliance, [appliance_df])]
    finally:
        store = getattr(data_set, "store", None)
        if store is not None:
            store.close()
