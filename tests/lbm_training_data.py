"""Shared synthetic LBM population windows for focused tests."""

from datetime import datetime, timedelta, timezone

import torch

from nilmtk_contrib.torch._lbm_training import ApplianceTrainingWindow, SourceWindow


FINGERPRINT = "sha256:" + "a" * 64
POWER_WINDOWS = (
    [0, 1, 0, 2, 0, 1],
    [1, 0, 2, 1, 0, 2],
    [2, 1, 0, 1, 2, 0],
    [0, 90, 100, 100, 0, 0],
    [0, 0, 95, 105, 0, 0],
    [0, 110, 100, 0, 0, 0],
    [0, 90, 0, 100, 0, 0],
    [0, 80, 0, 110, 0, 0],
    [0, 100, 0, 90, 0, 0],
)


def source_window(index, *, building="1", start=None, end=None, **overrides):
    start_value = start or datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
        hours=index
    )
    if isinstance(start_value, str):
        start_time = datetime.fromisoformat(start_value.replace("Z", "+00:00"))
    else:
        start_time = start_value
    period = overrides.get("sample_period_seconds", 60)
    end_value = end or start_time + timedelta(seconds=6 * max(period, 1))
    params = {
        "dataset": "REDD",
        "dataset_version": "nilmtk-converted-v1",
        "data_uri": "https://example.org/redd.h5",
        "data_fingerprint": FINGERPRINT,
        "building": building,
        "start": (
            start_value if isinstance(start_value, str) else start_value.isoformat()
        ),
        "end": end_value if isinstance(end_value, str) else end_value.isoformat(),
        "sample_period_seconds": 60,
    }
    params.update(overrides)
    return SourceWindow(**params)


def training_windows():
    return tuple(
        ApplianceTrainingWindow(source_window(index), power)
        for index, power in enumerate(POWER_WINDOWS)
    )


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)
