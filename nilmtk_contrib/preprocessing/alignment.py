"""Index alignment helpers."""

import pandas as pd


def restore_index(predictions, original_index):
    """Return a pandas object indexed like the original signal."""
    if len(predictions) != len(original_index):
        raise ValueError("predictions and original_index must have the same length.")

    if isinstance(predictions, pd.DataFrame):
        restored = predictions.copy()
        restored.index = original_index
        return restored

    if isinstance(predictions, pd.Series):
        restored = predictions.copy()
        restored.index = original_index
        return restored

    return pd.Series(predictions, index=original_index)
