import numpy as np
import pytest

from nilmtk_contrib.preprocessing.classification import make_on_off_labels
from nilmtk_contrib.preprocessing.normalization import denormalize, normalize
from nilmtk_contrib.preprocessing.windows import (
    make_sliding_windows,
    overlap_average,
    sequence_to_point_targets,
)


def test_center_padded_windows_match_original_length():
    windows, metadata = make_sliding_windows([1, 2, 3], 3, pad="center")

    assert windows.tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 0]]
    assert len(windows) == 3
    assert metadata.original_length == 3
    assert metadata.pad_left == 1
    assert metadata.pad_right == 1


def test_center_padded_windows_honor_pad_value_and_even_window_metadata():
    windows, metadata = make_sliding_windows([1, 2], 4, pad="center", pad_value=-1)

    assert windows.tolist() == [[-1, 1, 2, -1], [1, 2, -1, -1]]
    assert metadata.pad_left == 1
    assert metadata.pad_right == 2
    assert metadata.pad_value == -1
    assert metadata.trim_slice == (1, 3)


def test_center_padded_windows_handle_short_input():
    windows, metadata = make_sliding_windows([5], 5, pad="center")

    assert windows.tolist() == [[0, 0, 5, 0, 0]]
    assert metadata.original_length == 1
    assert metadata.trim_slice == (2, 3)


def test_right_padded_windows_match_original_length():
    windows, metadata = make_sliding_windows([1, 2, 3], 3, pad="right")

    assert windows.tolist() == [[1, 2, 3], [2, 3, 0], [3, 0, 0]]
    assert len(windows) == 3
    assert metadata.pad_left == 0
    assert metadata.pad_right == 2


def test_unpadded_windows_use_only_complete_windows():
    windows, metadata = make_sliding_windows([1, 2, 3, 4], 3, pad="none")

    assert windows.tolist() == [[1, 2, 3], [2, 3, 4]]
    assert metadata.original_length == 4
    assert metadata.pad_left == 0
    assert metadata.pad_right == 0


def test_unpadded_windows_short_input_returns_empty_rows():
    windows, _ = make_sliding_windows([1, 2], 3, pad="none")

    assert windows.shape == (0, 3)


def test_make_sliding_windows_validates_arguments():
    with pytest.raises(ValueError, match="window_length must be a positive integer"):
        make_sliding_windows([1, 2, 3], 0)

    with pytest.raises(ValueError, match="pad must be one of"):
        make_sliding_windows([1, 2, 3], 3, pad="left")


def test_sequence_to_point_targets_use_center_values():
    targets = sequence_to_point_targets([10, 20, 30], 3, center=True)

    assert targets.tolist() == [10, 20, 30]


def test_sequence_to_point_targets_non_center_uses_right_edge():
    targets = sequence_to_point_targets([10, 20, 30, 40], 3, center=False)

    assert targets.tolist() == [30, 40]


def test_sequence_to_point_targets_non_center_short_input_is_empty():
    targets = sequence_to_point_targets([10, 20], 3, center=False)

    assert targets.size == 0


def test_overlap_average_combines_known_windows():
    averaged = overlap_average(np.array([[1, 2, 3], [4, 5, 6]]), original_length=4)

    assert averaged.tolist() == [1, 3, 4, 6]


def test_overlap_average_trims_center_excess():
    averaged = overlap_average(np.array([[1, 2, 3], [4, 5, 6]]), original_length=2)

    assert averaged.tolist() == [3, 4]


def test_overlap_average_supports_untrimmed_short_and_empty_outputs():
    untrimmed = overlap_average(
        np.array([[1, 2, 3], [4, 5, 6]]),
        original_length=2,
        trim=False,
    )
    short = overlap_average(np.array([[1, 2, 3]]), original_length=5)
    empty = overlap_average(np.empty((0, 3)), original_length=5)

    assert untrimmed.tolist() == [1, 3, 4, 6]
    assert short.tolist() == [1, 2, 3]
    assert empty.size == 0


def test_overlap_average_validates_shape_and_original_length():
    with pytest.raises(ValueError, match="2D array"):
        overlap_average(np.array([1, 2, 3]), original_length=3)

    with pytest.raises(ValueError, match="non-negative"):
        overlap_average(np.array([[1, 2, 3]]), original_length=-1)


def test_normalize_records_fallback_std_and_denormalizes():
    normalized, metadata = normalize([100, 200], mean=100, std=0)

    assert normalized.tolist() == [0, 1]
    assert metadata.requested_std == 0
    assert metadata.std_used == 100
    assert denormalize(normalized, mean=100, std=metadata.std_used).tolist() == [100, 200]


def test_normalize_uses_requested_std_when_large_enough_and_handles_none():
    normalized, metadata = normalize([12, 16], mean=10, std=2)
    fallback, fallback_metadata = normalize([10, 110], mean=10, std=None)

    assert normalized.tolist() == [1, 3]
    assert metadata.std_used == 2
    assert fallback.tolist() == [0, 1]
    assert fallback_metadata.requested_std is None
    assert fallback_metadata.std_used == 100


def test_make_on_off_labels_requires_explicit_threshold():
    assert make_on_off_labels([0, 10, 20], threshold=10).tolist() == [0, 1, 1]

    with pytest.raises(ValueError, match="threshold must be explicit"):
        make_on_off_labels([0, 10], threshold=None)
