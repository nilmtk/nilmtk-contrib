import sys
from types import SimpleNamespace

import pandas as pd

from nilmtk_contrib.mains_stats import (
    _empty_stats,
    calculate_multi_building_mains_stats,
)


class FakeStore:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeMains:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def power_series_all_data(self, ac_type, sample_period):
        self.calls.append({"ac_type": ac_type, "sample_period": sample_period})
        if self.error:
            raise self.error
        return self.result


class FakeElec:
    def __init__(self, mains):
        self._mains = mains

    def mains(self):
        return self._mains


class FakeBuilding:
    def __init__(self, mains):
        self.elec = FakeElec(mains)


class FakeDataSet:
    last_instance = None

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.store = FakeStore()
        self.window = None
        self.buildings = {}
        FakeDataSet.last_instance = self

    def set_window(self, start, end):
        self.window = {"start": start, "end": end}


def _install_fake_nilmtk(monkeypatch):
    module = SimpleNamespace(DataSet=FakeDataSet)
    monkeypatch.setitem(sys.modules, "nilmtk", module)
    return module


def test_empty_stats_has_expected_shape():
    assert _empty_stats("apparent") == {
        "mean": 0,
        "std": 0,
        "min": 0,
        "max": 0,
        "data_points": 0,
        "ac_type": "apparent",
    }


def test_calculate_multi_building_mains_stats_aggregates_and_closes_store(monkeypatch):
    _install_fake_nilmtk(monkeypatch)

    original_init = FakeDataSet.__init__

    def init_with_buildings(self, dataset_path):
        original_init(self, dataset_path)
        self.buildings = {
            1: FakeBuilding(FakeMains(pd.Series([1.0, 2.0, None]))),
            2: FakeBuilding(FakeMains(pd.Series([4.0]))),
        }

    monkeypatch.setattr(FakeDataSet, "__init__", init_with_buildings)

    stats = calculate_multi_building_mains_stats(
        "dataset.h5",
        building_ids=[1, 2],
        start_time="2020-01-01",
        end_time="2020-01-02",
        ac_type="active",
        sample_period=30,
    )

    dataset = FakeDataSet.last_instance
    assert dataset.dataset_path == "dataset.h5"
    assert dataset.window == {"start": "2020-01-01", "end": "2020-01-02"}
    assert dataset.store.closed is True
    assert stats["data_points"] == 3
    assert stats["mean"] == pd.Series([1.0, 2.0, 4.0]).mean()
    assert stats["std"] == pd.Series([1.0, 2.0, 4.0]).std()
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0
    assert stats["ac_type"] == "active"


def test_calculate_multi_building_mains_stats_ignores_missing_empty_and_failed_buildings(
    monkeypatch,
):
    _install_fake_nilmtk(monkeypatch)
    original_init = FakeDataSet.__init__

    def init_with_buildings(self, dataset_path):
        original_init(self, dataset_path)
        self.buildings = {
            1: FakeBuilding(FakeMains(pd.Series(dtype=float))),
            2: FakeBuilding(FakeMains(error=RuntimeError("meter failed"))),
        }

    monkeypatch.setattr(FakeDataSet, "__init__", init_with_buildings)

    stats = calculate_multi_building_mains_stats(
        "dataset.h5",
        building_ids=[1, 2, 3],
        start_time="2020-01-01",
        end_time="2020-01-02",
        verbose=True,
    )

    assert stats == _empty_stats("active")
    assert FakeDataSet.last_instance.store.closed is True


def test_calculate_multi_building_mains_stats_handles_dataset_without_store(monkeypatch):
    _install_fake_nilmtk(monkeypatch)
    original_init = FakeDataSet.__init__

    def init_without_store(self, dataset_path):
        original_init(self, dataset_path)
        self.store = None
        self.buildings = {1: FakeBuilding(FakeMains(pd.Series([5.0])))}

    monkeypatch.setattr(FakeDataSet, "__init__", init_without_store)

    stats = calculate_multi_building_mains_stats(
        "dataset.h5",
        building_ids=[1],
        start_time="2020-01-01",
        end_time="2020-01-02",
    )

    assert stats["data_points"] == 1
    assert stats["mean"] == 5.0
