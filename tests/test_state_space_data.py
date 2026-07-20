import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from nilmtk_contrib.torch._state_space_data import (  # noqa: E402
    aligned_power_windows,
    as_power_tensor,
    frame_index,
)


def test_power_tensor_is_owned_float64_and_nonnegative():
    values = np.array([0.0, 10.0], dtype=np.float32)

    tensor = as_power_tensor(values, "power")
    values[1] = 99.0

    assert tensor.dtype == torch.float64
    assert tensor.device.type == "cpu"
    assert tensor.tolist() == [0.0, 10.0]
    with pytest.raises(ValueError, match="non-negative"):
        as_power_tensor(np.array([0.0, -1.0]), "power")


def test_aligned_windows_preserve_chunks_without_aliasing():
    index = pd.date_range("2026-01-01", periods=3, freq="1min")
    mains = [pd.DataFrame({"power": [10.0, 20.0, 30.0]}, index=index)]
    targets = [pd.DataFrame({"power": [0.0, 10.0, 20.0]}, index=index)]

    main_tensors, target_tensors = aligned_power_windows(mains, targets, "fridge")
    mains[0].iloc[0, 0] = 999.0

    assert len(main_tensors) == len(target_tensors) == 1
    assert main_tensors[0].tolist() == [10.0, 20.0, 30.0]
    assert target_tensors[0].tolist() == [0.0, 10.0, 20.0]


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        ([], "chunks but mains"),
        ([pd.DataFrame({"power": [0.0]})], "length does not match"),
        (
            [
                pd.DataFrame(
                    {"power": [0.0, 10.0]},
                    index=pd.date_range("2027-01-01", periods=2, freq="1min"),
                )
            ],
            "index does not match",
        ),
    ],
)
def test_aligned_windows_reject_mismatched_chunk_contracts(targets, message):
    mains = [
        pd.DataFrame(
            {"power": [10.0, 20.0]},
            index=pd.date_range("2026-01-01", periods=2, freq="1min"),
        )
    ]

    with pytest.raises(ValueError, match=message):
        aligned_power_windows(mains, targets, "fridge")


def test_frame_index_preserves_labels_or_builds_range():
    index = pd.date_range("2026-01-01", periods=2, freq="1min", tz="UTC")
    frame = pd.DataFrame({"power": [1.0, 2.0]}, index=index)

    assert frame_index(frame, 2).equals(index)
    assert frame_index(np.array([1.0, 2.0]), 2).equals(pd.RangeIndex(2))
