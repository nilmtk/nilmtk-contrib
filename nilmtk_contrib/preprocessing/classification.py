"""Classification label helpers."""

import numpy as np


def make_on_off_labels(values, threshold):
    """Create binary on/off labels using an explicit power threshold."""
    if threshold is None:
        raise ValueError("threshold must be explicit.")
    return (np.asarray(values) >= threshold).astype(int)
