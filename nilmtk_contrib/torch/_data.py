"""Shared strict conversion of NILM power channels to numeric vectors."""

import numpy as np


def _float32_array(frame, label):
    """Convert real numeric input without silently accepting lossy values."""
    raw = frame.to_numpy() if hasattr(frame, "to_numpy") else np.asarray(frame)
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"{label} must contain real numeric data.")
    if np.iscomplexobj(raw):
        raise TypeError(f"{label} must contain real numeric data.")
    if not np.isfinite(raw).all():
        raise ValueError(f"{label} must contain only finite values.")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} is not representable as float32.")
    return values


def power_vector(frame, label, *, allow_empty=False):
    """Validate one raw power channel and return a float32 vector."""
    values = _float32_array(frame, label)
    if values.ndim == 1:
        vector = values
    elif values.ndim == 2 and values.shape[1] == 1:
        vector = values[:, 0]
    else:
        raise ValueError(f"{label} must contain exactly one power column.")
    if not allow_empty and not len(vector):
        raise ValueError(f"{label} must contain at least one sample.")
    return vector


__all__ = ["power_vector"]
