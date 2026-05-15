import pandas as pd
import pytest

from nilmtk_contrib.preprocessing.alignment import restore_index


def test_restore_index_from_array_returns_series():
    index = pd.date_range("2026-01-01", periods=3, freq="min")

    restored = restore_index([1, 2, 3], index)

    assert isinstance(restored, pd.Series)
    assert restored.index.equals(index)
    assert restored.tolist() == [1, 2, 3]


def test_restore_index_preserves_series_name():
    index = pd.date_range("2026-01-01", periods=2, freq="min")
    predictions = pd.Series([5, 6], name="fridge")

    restored = restore_index(predictions, index)

    assert restored.name == "fridge"
    assert restored.index.equals(index)


def test_restore_index_preserves_dataframe_columns():
    index = pd.date_range("2026-01-01", periods=2, freq="min")
    predictions = pd.DataFrame({"fridge": [5, 6], "kettle": [0, 1]})

    restored = restore_index(predictions, index)

    assert restored.columns.tolist() == ["fridge", "kettle"]
    assert restored.index.equals(index)


def test_restore_index_rejects_length_mismatch():
    index = pd.date_range("2026-01-01", periods=2, freq="min")

    with pytest.raises(ValueError, match="same length"):
        restore_index([1, 2, 3], index)
