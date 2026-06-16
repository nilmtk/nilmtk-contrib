"""Shared preprocessing helpers for NILM models."""

from nilmtk_contrib.preprocessing.alignment import restore_index
from nilmtk_contrib.preprocessing.classification import make_on_off_labels
from nilmtk_contrib.preprocessing.normalization import denormalize, normalize
from nilmtk_contrib.preprocessing.windows import (
    make_sliding_windows,
    overlap_average,
    sequence_to_point_targets,
)

__all__ = [
    "denormalize",
    "make_on_off_labels",
    "make_sliding_windows",
    "normalize",
    "overlap_average",
    "restore_index",
    "sequence_to_point_targets",
]
