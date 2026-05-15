"""Normalization helpers."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizationMetadata:
    mean: float
    requested_std: float
    std_used: float


def normalize(values, mean, std, min_std=1, fallback_std=100):
    """Normalize values without dividing by zero or tiny std values."""
    std_used = std
    if std_used is None or abs(std_used) < min_std:
        std_used = fallback_std
    if std_used == 0:
        std_used = fallback_std

    normalized = (np.asarray(values) - mean) / std_used
    metadata = NormalizationMetadata(
        mean=mean,
        requested_std=std,
        std_used=std_used,
    )
    return normalized, metadata


def denormalize(values, mean, std):
    """Undo simple z-score normalization."""
    return mean + np.asarray(values) * std
