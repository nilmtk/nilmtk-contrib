"""Shared, inspectable PyTorch features for normalized NILM windows."""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch import nn


WINDOW_FEATURE_NAMES = (
    "center",
    "mean",
    "standard_deviation",
    "minimum",
    "maximum",
    "mean_absolute_difference",
    "end_to_start_change",
)

PREDICTION_FEATURE_NAMES = (
    "center",
    "mean",
    "standard_deviation",
    "minimum",
    "maximum",
    "median",
    "interquartile_range",
    "root_mean_square",
    "mean_absolute_deviation",
    "mean_absolute_difference",
    "rms_difference",
    "maximum_absolute_difference",
    "end_to_start_change",
    "linear_slope",
    "lag_one_correlation",
    "quarter_window_lag_correlation",
    "low_frequency_power_fraction",
    "mid_frequency_power_fraction",
    "high_frequency_power_fraction",
    "spectral_centroid",
    "spectral_entropy",
)


def _validated_windows(inputs, *, sequence_length, owner, reference=None):
    if not isinstance(inputs, torch.Tensor):
        raise TypeError(f"{owner} inputs must be a torch.Tensor.")
    expected_shape = (1, sequence_length)
    if inputs.ndim != 3 or inputs.shape[1:] != expected_shape:
        raise ValueError(
            f"{owner} expects input shape "
            f"(batch, 1, {sequence_length}); got {tuple(inputs.shape)}."
        )
    if not torch.is_floating_point(inputs) or torch.is_complex(inputs):
        raise TypeError(f"{owner} inputs must be real floating tensors.")
    if reference is None and inputs.dtype not in {torch.float32, torch.float64}:
        raise TypeError(f"{owner} inputs must use torch.float32 or torch.float64.")
    if reference is not None:
        if inputs.device != reference.device:
            raise ValueError(
                f"{owner} input is on {inputs.device}; model is on {reference.device}."
            )
        inputs = inputs.to(dtype=reference.dtype)
    if not bool(torch.isfinite(inputs).all()):
        raise ValueError(f"{owner} inputs must be finite.")
    return inputs.squeeze(1)


def window_summary_features(inputs, *, sequence_length, reference, owner):
    """Validate one-channel windows and return seven summary features."""
    values = _validated_windows(
        inputs,
        sequence_length=sequence_length,
        reference=reference,
        owner=owner,
    )
    differences = values.diff(dim=-1)
    return torch.cat(
        (
            values[:, sequence_length // 2 : sequence_length // 2 + 1],
            values.mean(dim=-1, keepdim=True),
            values.std(dim=-1, unbiased=False, keepdim=True),
            values.amin(dim=-1, keepdim=True),
            values.amax(dim=-1, keepdim=True),
            differences.abs().mean(dim=-1, keepdim=True),
            values[:, -1:] - values[:, :1],
        ),
        dim=1,
    )


class WindowFeatureExtractor(nn.Module):
    """Map each one-channel window to statistical and spectral descriptors.

    The extractor has no trainable parameters and uses only PyTorch tensor
    operations. Frequency features exclude the DC component and are normalized
    to remain comparable across window lengths and input scales.
    """

    feature_names = PREDICTION_FEATURE_NAMES

    def __init__(self, sequence_length, epsilon=1e-8):
        super().__init__()
        if isinstance(sequence_length, bool) or not isinstance(
            sequence_length, Integral
        ):
            raise ValueError("sequence_length must be an integer.")
        if sequence_length % 2 == 0:
            raise ValueError("sequence_length must be odd.")
        if sequence_length < 9:
            raise ValueError("sequence_length must be at least 9.")
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, Real)
            or not math.isfinite(epsilon)
            or epsilon <= 0
        ):
            raise ValueError("epsilon must be a positive finite number.")
        self.sequence_length = int(sequence_length)
        self.epsilon = float(epsilon)

    def _correlation(self, values, lag):
        left = values[:, :-lag]
        right = values[:, lag:]
        left = left - left.mean(dim=1, keepdim=True)
        right = right - right.mean(dim=1, keepdim=True)
        numerator = (left * right).sum(dim=1, keepdim=True)
        denominator = torch.linalg.vector_norm(
            left, dim=1, keepdim=True
        ) * torch.linalg.vector_norm(right, dim=1, keepdim=True)
        return numerator / denominator.clamp_min(self.epsilon)

    def forward(self, inputs):
        values = _validated_windows(
            inputs,
            sequence_length=self.sequence_length,
            owner="WindowFeatureExtractor",
        )
        center = values[:, self.sequence_length // 2 : self.sequence_length // 2 + 1]
        mean = values.mean(dim=1, keepdim=True)
        centered = values - mean
        differences = values.diff(dim=1)
        absolute_differences = differences.abs()
        quartiles = torch.quantile(
            values, values.new_tensor([0.25, 0.75]), dim=1
        ).transpose(0, 1)

        time = torch.linspace(
            -1,
            1,
            self.sequence_length,
            dtype=values.dtype,
            device=values.device,
        )
        slope = (centered * time).sum(dim=1, keepdim=True) / time.square().sum()

        frequencies = torch.fft.rfftfreq(
            self.sequence_length, dtype=values.dtype, device=values.device
        )[1:]
        spectral_power = torch.fft.rfft(centered, norm="ortho")[:, 1:].abs().square()
        total_power = spectral_power.sum(dim=1, keepdim=True)
        spectral_distribution = spectral_power / total_power.clamp_min(self.epsilon)
        low = frequencies <= (1 / 6)
        mid = (frequencies > (1 / 6)) & (frequencies <= (1 / 3))
        high = frequencies > (1 / 3)
        band_fractions = torch.stack(
            [
                spectral_distribution[:, low].sum(dim=1),
                spectral_distribution[:, mid].sum(dim=1),
                spectral_distribution[:, high].sum(dim=1),
            ],
            dim=1,
        )
        spectral_centroid = (spectral_distribution * frequencies).sum(
            dim=1, keepdim=True
        ) / 0.5
        spectral_entropy = -torch.special.xlogy(
            spectral_distribution, spectral_distribution
        ).sum(dim=1, keepdim=True) / math.log(spectral_distribution.shape[1])

        features = torch.cat(
            (
                center,
                mean,
                values.std(dim=1, unbiased=False, keepdim=True),
                values.amin(dim=1, keepdim=True),
                values.amax(dim=1, keepdim=True),
                values.median(dim=1, keepdim=True).values,
                quartiles[:, 1:] - quartiles[:, :1],
                torch.linalg.vector_norm(values, dim=1, keepdim=True)
                / math.sqrt(self.sequence_length),
                centered.abs().mean(dim=1, keepdim=True),
                absolute_differences.mean(dim=1, keepdim=True),
                torch.linalg.vector_norm(differences, dim=1, keepdim=True)
                / math.sqrt(self.sequence_length - 1),
                absolute_differences.amax(dim=1, keepdim=True),
                values[:, -1:] - values[:, :1],
                slope,
                self._correlation(values, 1),
                self._correlation(values, self.sequence_length // 4),
                band_fractions,
                spectral_centroid,
                spectral_entropy,
            ),
            dim=1,
        )
        if features.shape[1] != len(self.feature_names):
            raise RuntimeError("Window feature definition and names are inconsistent.")
        if not bool(torch.isfinite(features).all()):
            raise RuntimeError("WindowFeatureExtractor produced non-finite features.")
        return features


__all__ = [
    "PREDICTION_FEATURE_NAMES",
    "WINDOW_FEATURE_NAMES",
    "WindowFeatureExtractor",
    "window_summary_features",
]
