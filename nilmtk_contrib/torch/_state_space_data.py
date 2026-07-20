"""Shared input contracts for supervised PyTorch state-space models."""

from __future__ import annotations

import pandas as pd
import torch

from nilmtk_contrib.torch._data import power_vector


def as_power_tensor(frame, label) -> torch.Tensor:
    """Return one finite, nonnegative power channel as an owned CPU tensor."""

    values = power_vector(frame, label)
    if bool((values < 0).any()):
        raise ValueError(f"{label} must be non-negative.")
    return torch.as_tensor(values, dtype=torch.float64, device="cpu").clone()


def frame_index(frame, length):
    """Preserve a frame index or construct a positional fallback."""

    index = getattr(frame, "index", None)
    return index if index is not None else pd.RangeIndex(length)


def aligned_power_windows(main_frames, target_frames, appliance_name):
    """Validate aligned mains and appliance chunks without joining boundaries."""

    if len(main_frames) != len(target_frames):
        raise ValueError(
            f"{appliance_name!r} has {len(target_frames)} chunks but mains has "
            f"{len(main_frames)}."
        )
    mains = []
    targets = []
    for index, (main_frame, target_frame) in enumerate(
        zip(main_frames, target_frames, strict=True)
    ):
        main = as_power_tensor(main_frame, f"mains chunk {index}")
        target = as_power_tensor(
            target_frame, f"{appliance_name!r} target chunk {index}"
        )
        if main.numel() != target.numel():
            raise ValueError(
                f"{appliance_name!r} target chunk {index} length does not match mains."
            )
        main_index = getattr(main_frame, "index", None)
        target_index = getattr(target_frame, "index", None)
        if (
            main_index is not None
            and target_index is not None
            and not main_index.equals(target_index)
        ):
            raise ValueError(
                f"{appliance_name!r} target chunk {index} index does not match mains."
            )
        mains.append(main)
        targets.append(target)
    return tuple(mains), tuple(targets)


__all__ = ["aligned_power_windows", "as_power_tensor", "frame_index"]
