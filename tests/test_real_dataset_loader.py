import numpy as np
import pandas as pd
import pytest

from model_smoke_helpers import _bounded_power_series


class _TrackingGenerator:
    def __init__(self, chunks, error=None):
        self._chunks = iter(chunks)
        self._error = error
        self.yielded = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._chunks)
        except StopIteration:
            if self._error is not None:
                raise self._error
            raise
        self.yielded += 1
        return chunk

    def close(self):
        self.closed = True


class _Meter:
    def __init__(self, chunks, error=None):
        self.generator = _TrackingGenerator(chunks, error=error)
        self.kwargs = None

    def power_series(self, **kwargs):
        self.kwargs = kwargs
        return self.generator


def _series(values, start):
    return pd.Series(
        values,
        index=pd.date_range(start, periods=len(values), freq="1min", tz="UTC"),
        dtype=np.float32,
    )


def test_bounded_loader_stops_without_consuming_the_remaining_meter():
    meter = _Meter(
        [
            _series([1.0, 2.0, np.nan], "2026-01-01"),
            _series([3.0, 4.0, 5.0, 6.0], "2026-02-01"),
            _series([999.0], "2026-03-01"),
        ]
    )

    result = _bounded_power_series(meter, 5)

    assert result.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert meter.kwargs == {"chunksize": 5}
    assert meter.generator.yielded == 2
    assert meter.generator.closed


def test_bounded_loader_forwards_alignment_options():
    meter = _Meter([_series([1.0], "2026-01-01")])
    section = object()

    _bounded_power_series(
        meter,
        1,
        sections=[section],
        sample_period=6,
    )

    assert meter.kwargs == {
        "chunksize": 1,
        "sections": [section],
        "sample_period": 6,
    }


def test_bounded_loader_closes_an_empty_generator():
    meter = _Meter([_series([], "2026-01-01")])

    result = _bounded_power_series(meter, 5)

    assert result.empty
    assert result.dtype == np.float32
    assert meter.generator.closed


def test_bounded_loader_coalesces_duplicate_timestamps_by_mean():
    first = _series([1.0, 2.0], "2026-01-01")
    second = pd.Series(
        [4.0, 8.0],
        index=pd.DatetimeIndex([first.index[1], first.index[1] + pd.Timedelta("1min")]),
        dtype=np.float32,
    )
    meter = _Meter([first, second])

    result = _bounded_power_series(meter, 4)

    assert result.index.is_unique
    assert result.tolist() == [1.0, 3.0, 8.0]
    assert result.index.is_monotonic_increasing


def test_bounded_loader_closes_generator_when_iteration_fails():
    meter = _Meter([_series([1.0], "2026-01-01")], error=RuntimeError("broken"))

    with pytest.raises(RuntimeError, match="broken"):
        _bounded_power_series(meter, 2)

    assert meter.generator.closed


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_bounded_loader_rejects_invalid_limits(value):
    with pytest.raises(ValueError, match="positive integer"):
        _bounded_power_series(_Meter([]), value)
