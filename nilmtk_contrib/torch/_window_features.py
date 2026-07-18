"""Shared, inspectable features for routing normalized NILM windows."""

import torch


WINDOW_FEATURE_NAMES = (
    "center",
    "mean",
    "standard_deviation",
    "minimum",
    "maximum",
    "mean_absolute_difference",
    "end_to_start_change",
)


def window_summary_features(inputs, *, sequence_length, reference, owner):
    """Validate one-channel windows and return seven summary features."""
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
    if inputs.device != reference.device:
        raise ValueError(
            f"{owner} input is on {inputs.device}; model is on {reference.device}."
        )
    if not torch.isfinite(inputs).all():
        raise ValueError(f"{owner} inputs must be finite.")

    inputs = inputs.to(dtype=reference.dtype)
    differences = inputs.diff(dim=-1)
    return torch.cat(
        (
            inputs[:, :, sequence_length // 2],
            inputs.mean(dim=-1),
            inputs.std(dim=-1, unbiased=False),
            inputs.amin(dim=-1),
            inputs.amax(dim=-1),
            differences.abs().mean(dim=-1),
            inputs[:, :, -1] - inputs[:, :, 0],
        ),
        dim=1,
    )


__all__ = ["WINDOW_FEATURE_NAMES", "window_summary_features"]
