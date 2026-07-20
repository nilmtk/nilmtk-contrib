from pathlib import Path

import numpy as np
import pandas as pd
import torch

import nilmtk_contrib.torch.WindowGRU as window_gru_module
from nilmtk_contrib.torch.WindowGRU import WindowGRU


class TinyPointModel(torch.nn.Module):
    def __init__(self, sequence_length):
        super().__init__()
        self.output = torch.nn.Linear(sequence_length, 1)

    def forward(self, values):
        return self.output(values.flatten(start_dim=1))


def _model(sequence_length=5):
    model = WindowGRU(
        {
            "sequence_length": sequence_length,
            "n_epochs": 1,
            "batch_size": 2,
            "device": "cpu",
            "seed": 123,
            "verbose": False,
        }
    )
    model.return_network = lambda: TinyPointModel(sequence_length)
    return model


def _training_data(num_windows, sequence_length=5):
    mains = pd.DataFrame(
        np.arange(num_windows * sequence_length, dtype=np.float32).reshape(
            num_windows, sequence_length
        )
    )
    appliance = pd.DataFrame(
        np.linspace(1.0, 2.0, num_windows, dtype=np.float32).reshape(-1, 1)
    )
    return [mains], [("fridge", [appliance])]


def test_single_window_trains_without_an_empty_validation_failure(monkeypatch):
    model = _model()
    train_main, train_appliances = _training_data(1)
    saved_paths = []
    loaded_paths = []
    real_save = window_gru_module.save_torch_state
    real_load = window_gru_module.load_torch_state

    def recording_save(network, path):
        saved_paths.append(Path(path))
        real_save(network, path)

    def recording_load(network, path, device):
        loaded_paths.append((Path(path), device))
        return real_load(network, path, device)

    monkeypatch.setattr(window_gru_module, "save_torch_state", recording_save)
    monkeypatch.setattr(window_gru_module, "load_torch_state", recording_load)

    model.partial_fit(train_main, train_appliances, do_preprocessing=False)

    assert list(model.models) == ["fridge"]
    assert saved_paths
    assert loaded_paths == [(saved_paths[-1], torch.device("cpu"))]
    assert not saved_paths[-1].exists()
    assert all(torch.isfinite(value).all() for value in model.models["fridge"].parameters())


def test_tail_validation_still_trains_and_restores_the_best_weights():
    model = _model()
    train_main, train_appliances = _training_data(4)

    model.partial_fit(train_main, train_appliances, do_preprocessing=False)

    predictions = model.disaggregate_chunk(
        train_main,
        do_preprocessing=False,
    )
    assert len(predictions) == 1
    assert predictions[0].shape == (4, 1)
    assert np.isfinite(predictions[0].to_numpy()).all()
