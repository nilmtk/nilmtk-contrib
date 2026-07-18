from datetime import datetime, timezone
from itertools import product
import json
import math

import pytest


torch = pytest.importorskip("torch")

from nilmtk_contrib.torch._lbm import solve_structured_relaxation  # noqa: E402
from nilmtk_contrib.torch._lbm_training import (  # noqa: E402
    ApplianceTrainingWindow,
    _fit_summary,
    _markov_reward_moments,
    assert_disjoint_sources,
    fit_lbm_appliance,
)
from lbm_training_data import (  # noqa: E402
    FINGERPRINT,
    POWER_WINDOWS,
    source_window as _source,
    tensor as _tensor,
    training_windows as _windows,
)


def test_fitter_builds_a_kernel_ready_appliance_with_complete_metadata():
    result = fit_lbm_appliance(_windows())
    appliance = result.appliance

    assert _tensor(appliance.state_means).tolist() == pytest.approx(
        [0.3902439024, 97.6923076923]
    )
    assert _tensor(appliance.initial_probabilities).tolist() == pytest.approx(
        [10 / 11, 1 / 11]
    )
    assert torch.all(_tensor(appliance.initial_probabilities) > 0)
    assert torch.allclose(
        _tensor(appliance.transition_probabilities).sum(dim=1),
        torch.ones(2, dtype=torch.float64),
    )
    assert torch.all(_tensor(appliance.transition_probabilities) > 0)
    assert appliance.off_state == 0
    assert _tensor(appliance.cycle_probabilities).tolist() == pytest.approx(
        [1 / 3, 1 / 3, 1 / 3]
    )
    assert len(appliance.summaries) == 2
    assert result.num_samples == 54
    assert result.window_length == 6
    assert result.sample_period_seconds == 60
    assert sum(result.state_counts) == 54
    assert result.initial_counts == (9, 0)
    assert sum(sum(row) for row in result.transition_counts) == 45
    assert result.cycle_counts == (3, 3, 3)
    assert result.emission_variance > 0

    metadata = result.metadata()
    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata["schema_version"] == 1
    assert metadata["initial_counts"] == [9, 0]
    assert sum(sum(row) for row in metadata["transition_counts"]) == 45
    assert metadata["off_state"] == 0
    assert metadata["evaluation_sources"] == []
    assert metadata["allow_temporal_evaluation"] is False
    assert metadata["sources"][0]["data_fingerprint"] == FINGERPRINT
    assert [summary["name"] for summary in metadata["summaries"]] == [
        "energy_wh",
        "duration_minutes",
    ]
    assert [summary["weight"] for summary in metadata["summaries"]] == [1.0, 1.0]

    inferred = solve_structured_relaxation(
        POWER_WINDOWS[3],
        [appliance],
        observation_variance=result.emission_variance + 1.0,
        max_iterations=5,
    )
    assert inferred.prediction.shape == (6,)
    assert inferred.cycle_posteriors[0] is not None


def test_transition_counts_do_not_cross_source_window_boundaries():
    result = fit_lbm_appliance(_windows(), pseudocount=1.0)
    means = _tensor(result.appliance.state_means)
    expected_counts = torch.zeros((2, 2), dtype=torch.float64)
    for power in POWER_WINDOWS:
        states = torch.argmin(
            torch.abs(_tensor(power)[:, None] - means[None, :]),
            dim=1,
        )
        for previous, current in zip(states[:-1], states[1:]):
            expected_counts[previous, current] += 1
    expected = (expected_counts + 1.0) / (expected_counts + 1.0).sum(
        dim=1,
        keepdim=True,
    )

    assert torch.allclose(_tensor(result.appliance.transition_probabilities), expected)


def test_fitting_is_invariant_to_caller_window_order():
    forward = fit_lbm_appliance(_windows())
    reverse = fit_lbm_appliance(tuple(reversed(_windows())))

    assert forward.metadata() == reverse.metadata()


def test_markov_reward_moments_match_complete_path_enumeration():
    initial = _tensor([0.6, 0.4])
    transition = _tensor([[0.7, 0.3], [0.2, 0.8]])
    rewards = _tensor([0.0, 2.0])
    mean, variance = _markov_reward_moments(initial, transition, rewards, 3)

    outcomes = []
    for path in product(range(2), repeat=3):
        probability = float(initial[path[0]])
        for previous, current in zip(path[:-1], path[1:]):
            probability *= float(transition[previous, current])
        outcomes.append((probability, sum(float(rewards[state]) for state in path)))
    expected_mean = sum(probability * value for probability, value in outcomes)
    expected_variance = sum(
        probability * (value - expected_mean) ** 2 for probability, value in outcomes
    )

    assert mean == pytest.approx(expected_mean, abs=1e-12)
    assert variance == pytest.approx(expected_variance, abs=1e-12)


def test_overlap_guard_rejects_evaluation_building_by_default():
    training = _source(0)
    overlapping = _source(
        50,
        start=datetime(2026, 1, 1, 0, 3, tzinfo=timezone.utc),
        end=datetime(2026, 1, 1, 0, 9, tzinfo=timezone.utc),
        data_fingerprint="sha256:" + "b" * 64,
    )
    disjoint_same_building = _source(1)
    different_building = _source(0, building="2")

    with pytest.raises(ValueError, match="Training/evaluation leakage"):
        assert_disjoint_sources([training], [overlapping])

    with pytest.raises(ValueError, match="Training/evaluation leakage"):
        assert_disjoint_sources([training], [disjoint_same_building])

    assert_disjoint_sources([training], [different_building])


def test_disjoint_same_building_time_requires_explicit_temporal_opt_in():
    training = _source(0)
    evaluation = _source(1)
    overlapping = _source(
        2,
        start=datetime(2026, 1, 1, 0, 3, tzinfo=timezone.utc),
        end=datetime(2026, 1, 1, 0, 9, tzinfo=timezone.utc),
    )

    assert_disjoint_sources(
        [training],
        [evaluation],
        allow_temporal_split=True,
    )
    with pytest.raises(ValueError, match="Training/evaluation leakage"):
        assert_disjoint_sources(
            [training],
            [overlapping],
            allow_temporal_split=True,
        )
    fit_lbm_appliance(
        _windows(),
        evaluation_sources=[_source(20)],
        allow_temporal_evaluation=True,
    )


def test_metadata_records_sorted_evaluation_sources_and_training_policy():
    evaluation = [_source(0, building="3"), _source(0, building="2")]

    result = fit_lbm_appliance(
        _windows(),
        evaluation_sources=evaluation,
        minimum_windows_per_cycle=2,
        kmeans_max_iterations=50,
    )
    metadata = result.metadata()

    assert [source["building"] for source in metadata["evaluation_sources"]] == [
        "2",
        "3",
    ]
    assert metadata["minimum_windows_per_cycle"] == 2
    assert metadata["kmeans_max_iterations"] == 50
    assert metadata["allow_temporal_evaluation"] is False


def test_matching_fingerprint_or_uri_blocks_dataset_alias_leakage():
    aliases = [
        _source(20, dataset="REDD alias"),
        _source(
            20,
            dataset="REDD alias",
            data_fingerprint="sha256:" + "b" * 64,
        ),
    ]

    for aliased_evaluation in aliases:
        with pytest.raises(ValueError, match="Training/evaluation leakage"):
            fit_lbm_appliance(_windows(), evaluation_sources=[aliased_evaluation])


def test_fit_rejects_overlapping_evaluation_source():
    evaluation = _source(0, data_fingerprint="sha256:" + "b" * 64)

    with pytest.raises(ValueError, match="Training/evaluation leakage"):
        fit_lbm_appliance(_windows(), evaluation_sources=[evaluation])


def test_fit_rejects_overlapping_or_duplicate_training_evidence():
    duplicated = _windows() + (_windows()[0],)

    with pytest.raises(ValueError, match="Duplicate training evidence"):
        fit_lbm_appliance(duplicated)


def test_fit_rejects_missing_population_cycle_categories():
    sparse = tuple(
        ApplianceTrainingWindow(_source(index), power)
        for index, power in enumerate(POWER_WINDOWS[:3] + POWER_WINDOWS[6:])
    )

    with pytest.raises(ValueError, match="underspecified"):
        fit_lbm_appliance(sparse)


@pytest.mark.parametrize(
    ("source_kwargs", "message"),
    [
        ({"dataset": " "}, "dataset must be a non-empty"),
        ({"data_fingerprint": "sha256:bad"}, "data_fingerprint"),
        ({"start": "2026-01-01T00:00:00"}, "explicit UTC offset"),
        (
            {
                "start": "2026-01-01T01:00:00+00:00",
                "end": "2026-01-01T00:00:00+00:00",
            },
            "later than start",
        ),
        ({"sample_period_seconds": 0}, "must be positive"),
        ({"building": []}, "building must be"),
    ],
)
def test_source_provenance_is_strictly_validated(source_kwargs, message):
    with pytest.raises(ValueError, match=message):
        _source(0, **source_kwargs)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda windows: (), "non-empty"),
        (
            lambda windows: (
                ApplianceTrainingWindow(windows[0].source, [0.0, math.nan]),
            ),
            "finite",
        ),
        (
            lambda windows: (ApplianceTrainingWindow(windows[0].source, [0.0, -1.0]),),
            "non-negative",
        ),
        (
            lambda windows: (ApplianceTrainingWindow(windows[0].source, [0.0, 1.0]),),
            "timestamps span",
        ),
        (
            lambda windows: windows[:-1]
            + (
                ApplianceTrainingWindow(
                    _source(8, sample_period_seconds=30),
                    windows[-1].power,
                ),
            ),
            "same sample period",
        ),
    ],
)
def test_fit_rejects_malformed_training_windows(mutator, message):
    with pytest.raises(ValueError, match=message):
        fit_lbm_appliance(mutator(_windows()))


def test_fit_rejects_insufficient_state_support():
    constant = tuple(
        ApplianceTrainingWindow(_source(index), [1.0] * 6) for index in range(3)
    )

    with pytest.raises(ValueError, match="unique values"):
        fit_lbm_appliance(constant, minimum_windows_per_cycle=1)


def test_fit_rejects_numerical_overflow_instead_of_persisting_nan():
    extreme = tuple(
        ApplianceTrainingWindow(
            _source(index),
            [1e308, 0.0, 1e308, 0.0, 1e308, 0.0],
        )
        for index in range(3)
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        fit_lbm_appliance(extreme, minimum_windows_per_cycle=1)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"num_states": 1}, ValueError, "at least two"),
        ({"pseudocount": 0.0}, ValueError, "positive"),
        ({"kmeans_max_iterations": True}, TypeError, "integer"),
        (
            {"allow_temporal_evaluation": "yes"},
            TypeError,
            "must be a boolean",
        ),
        (
            {"evaluation_sources": [object()]},
            TypeError,
            "SourceWindow instances",
        ),
    ],
)
def test_fit_rejects_invalid_runtime_contracts(kwargs, error, message):
    with pytest.raises(error, match=message):
        fit_lbm_appliance(_windows(), **kwargs)


def test_summary_fit_rejects_a_nonconvex_population_to_hmm_ratio():
    with pytest.raises(ValueError, match="would not be convex"):
        _fit_summary(
            "energy_wh",
            _tensor([0.0, 100.0]),
            torch.tensor([0, 0], dtype=torch.long),
            1,
            _tensor([0.0, 1.0]),
            _tensor([0.5, 0.5]),
            _tensor([[0.5, 0.5], [0.5, 0.5]]),
            1,
            1e-6,
        )


def test_training_path_has_no_classical_solver_or_dataframe_dependency():
    source = (
        __import__("pathlib").Path(__file__).parents[1]
        / "nilmtk_contrib"
        / "torch"
        / "_lbm_training.py"
    ).read_text(encoding="utf-8")
    for dependency in ("cvxpy", "hmmlearn", "mosek", "numpy", "pandas", "scipy"):
        assert f"import {dependency}" not in source
        assert f"from {dependency}" not in source
