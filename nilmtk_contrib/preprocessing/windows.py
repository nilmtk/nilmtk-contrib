"""Windowing and sequence reconstruction helpers."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowMetadata:
    original_length: int
    window_length: int
    pad: str
    pad_left: int
    pad_right: int
    pad_value: float
    trim_slice: tuple[int, int]


def _as_1d(values):
    return np.asarray(values).reshape(-1)


def _windows_from_padded(values, window_length):
    if len(values) < window_length:
        return np.empty((0, window_length), dtype=values.dtype)
    return np.lib.stride_tricks.sliding_window_view(values, window_length).copy()


def make_sliding_windows(values, window_length, pad="center", pad_value=0):
    """Create sliding windows with explicit padding metadata."""
    if not isinstance(window_length, int) or window_length <= 0:
        raise ValueError("window_length must be a positive integer.")
    if pad not in {"center", "right", "none"}:
        raise ValueError("pad must be one of 'center', 'right', or 'none'.")

    flat = _as_1d(values)
    original_length = len(flat)

    if pad == "center":
        total_pad = window_length - 1
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
    elif pad == "right":
        pad_left = 0
        pad_right = window_length - 1
    else:
        pad_left = 0
        pad_right = 0

    padded = np.pad(
        flat,
        (pad_left, pad_right),
        mode="constant",
        constant_values=pad_value,
    )
    windows = _windows_from_padded(padded, window_length)
    metadata = WindowMetadata(
        original_length=original_length,
        window_length=window_length,
        pad=pad,
        pad_left=pad_left,
        pad_right=pad_right,
        pad_value=pad_value,
        trim_slice=(pad_left, pad_left + original_length),
    )
    return windows, metadata


def sequence_to_point_targets(appliance_values, window_length, center=True):
    """Create sequence-to-point targets from appliance readings."""
    flat = _as_1d(appliance_values)
    if not center:
        if len(flat) < window_length:
            return np.asarray([], dtype=flat.dtype)
        return flat[window_length - 1 :]

    windows, _ = make_sliding_windows(flat, window_length, pad="center")
    center_index = window_length // 2
    return windows[:, center_index]


def overlap_average(windows, original_length, trim=True):
    """Average overlapping sequence windows back to a single 1D signal."""
    arr = np.asarray(windows)
    if arr.ndim != 2:
        raise ValueError("windows must be a 2D array.")
    if original_length < 0:
        raise ValueError("original_length must be non-negative.")
    if arr.size == 0:
        return np.asarray([], dtype=arr.dtype)

    window_count, window_length = arr.shape
    output_length = window_count + window_length - 1
    totals = np.zeros(output_length, dtype=float)
    counts = np.zeros(output_length, dtype=float)

    for start, window in enumerate(arr):
        stop = start + window_length
        totals[start:stop] += window
        counts[start:stop] += 1

    averaged = totals / np.maximum(counts, 1)
    if not trim:
        return averaged

    if len(averaged) == original_length:
        return averaged

    excess = len(averaged) - original_length
    if excess <= 0:
        return averaged[:original_length]

    trim_left = excess // 2
    trim_right = trim_left + original_length
    return averaged[trim_left:trim_right]
