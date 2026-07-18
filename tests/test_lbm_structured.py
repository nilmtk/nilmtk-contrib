from itertools import product
import math

import pytest


torch = pytest.importorskip("torch")

from nilmtk_contrib.torch._lbm import (  # noqa: E402
    GaussianRatioSummary,
    StructuredAppliance,
    _minimum_moment_coupling,
    _path_vertex,
    _prepare_summary,
    solve_structured_relaxation,
)


def _two_state_appliance(**overrides):
    params = {
        "state_means": [0.0, 10.0],
        "initial_probabilities": [0.5, 0.5],
        "transition_probabilities": [[0.5, 0.5], [0.5, 0.5]],
    }
    params.update(overrides)
    return StructuredAppliance(**params)


def _assert_hmm_polytope_feasible(result, appliance_index, *, off_state=0):
    states = result.states[appliance_index]
    transitions = result.transitions[appliance_index]
    assert torch.all(states >= -1e-12)
    assert torch.all(transitions >= -1e-12)
    assert torch.allclose(
        states.sum(dim=1),
        torch.ones(states.shape[0], dtype=states.dtype, device=states.device),
        atol=1e-11,
        rtol=0,
    )
    if states.shape[0] > 1:
        assert torch.allclose(
            transitions.sum(dim=2),
            states[:-1],
            atol=1e-11,
            rtol=0,
        )
        assert torch.allclose(
            transitions.sum(dim=1),
            states[1:],
            atol=1e-11,
            rtol=0,
        )

    category = result.cycle_posteriors[appliance_index]
    if category is not None:
        assert torch.all(category >= -1e-12)
        assert torch.isclose(
            category.sum(),
            torch.tensor(1.0, dtype=category.dtype, device=category.device),
            atol=1e-11,
            rtol=0,
        )
        off_to_on = (
            transitions[:, off_state, :].sum()
            - transitions[:, off_state, off_state].sum()
        )
        expected_cycles = torch.dot(
            category,
            torch.arange(
                category.numel(), dtype=category.dtype, device=category.device
            ),
        )
        assert torch.isclose(off_to_on, expected_cycles, atol=1e-11, rtol=0)


def test_cycle_aware_viterbi_returns_a_feasible_annotated_path():
    emissions = torch.zeros((4, 2), dtype=torch.float64)
    transitions = torch.zeros((3, 2, 2), dtype=torch.float64)
    cycle_costs = torch.tensor([10.0, 0.0, 10.0], dtype=torch.float64)

    states, flows, category = _path_vertex(
        emissions,
        transitions,
        off_state=0,
        cycle_costs=cycle_costs,
    )

    assert category.tolist() == [0.0, 1.0, 0.0]
    assert torch.allclose(flows.sum(dim=2), states[:-1])
    assert torch.allclose(flows.sum(dim=1), states[1:])
    assert torch.isclose(
        flows[:, 0, 1].sum(),
        torch.tensor(1.0, dtype=flows.dtype),
    )


def test_cycle_oracle_uses_the_papers_expected_count_constraint():
    path_weights, category_weights = _minimum_moment_coupling(
        torch.tensor([10.0, 0.0, 10.0], dtype=torch.float64),
        torch.tensor([0.0, 10.0, 0.0], dtype=torch.float64),
    )

    assert path_weights.tolist() == [0.0, 1.0, 0.0]
    assert category_weights.tolist() == [0.5, 0.0, 0.5]
    counts = torch.arange(3, dtype=torch.float64)
    assert torch.dot(path_weights, counts) == torch.dot(category_weights, counts)


def test_cycle_moment_oracle_beats_an_independent_simplex_grid():
    path_costs = torch.tensor([1.2, -0.3, 2.1, 0.4], dtype=torch.float64)
    category_costs = torch.tensor([0.8, 0.1, -0.2], dtype=torch.float64)
    path_weights, category_weights = _minimum_moment_coupling(
        path_costs,
        category_costs,
    )
    path_counts = torch.arange(path_costs.numel(), dtype=torch.float64)
    category_counts = torch.arange(category_costs.numel(), dtype=torch.float64)

    assert torch.isclose(path_weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.isclose(
        category_weights.sum(),
        torch.tensor(1.0, dtype=torch.float64),
    )
    assert torch.all(path_weights >= 0)
    assert torch.all(category_weights >= 0)
    assert torch.isclose(
        torch.dot(path_weights, path_counts),
        torch.dot(category_weights, category_counts),
        atol=1e-12,
        rtol=0,
    )

    denominator = 10
    best_grid_value = math.inf
    path_grid = [
        weights
        for weights in product(range(denominator + 1), repeat=4)
        if sum(weights) == denominator
    ]
    category_grid = [
        weights
        for weights in product(range(denominator + 1), repeat=3)
        if sum(weights) == denominator
    ]
    for raw_path in path_grid:
        path_mean_numerator = sum(
            cycle * weight for cycle, weight in enumerate(raw_path)
        )
        for raw_category in category_grid:
            category_mean_numerator = sum(
                cycle * weight for cycle, weight in enumerate(raw_category)
            )
            if path_mean_numerator != category_mean_numerator:
                continue
            value = (
                sum(weight * float(cost) for weight, cost in zip(raw_path, path_costs))
                / denominator
                + sum(
                    weight * float(cost)
                    for weight, cost in zip(raw_category, category_costs)
                )
                / denominator
            )
            best_grid_value = min(best_grid_value, value)

    oracle_value = float(
        torch.dot(path_weights, path_costs)
        + torch.dot(category_weights, category_costs)
    )
    assert oracle_value <= best_grid_value + 1e-12


def test_gaussian_ratio_completion_matches_integral_cycle_objective():
    summary = GaussianRatioSummary(
        state_weights=[0.0, 1.0],
        conditional_means=[0.0, 4.0, 8.0],
        conditional_variance=2.0,
        induced_mean=3.0,
        induced_variance=10.0,
    )
    prepared = _prepare_summary(
        summary,
        time_points=2,
        states=2,
        cycles=3,
        device=torch.device("cpu"),
        dtype=torch.float64,
        label="summary",
    )

    for cycle in range(3):
        for statistic in (-2.0, 0.0, 5.0, 11.0):
            direct = 0.5 * (
                (statistic - summary.conditional_means[cycle]) ** 2
                / summary.conditional_variance
                - (statistic - summary.induced_mean) ** 2 / summary.induced_variance
            )
            completed = 0.5 * prepared.precision * (
                statistic - float(prepared.cycle_centers[cycle])
            ) ** 2 + float(prepared.cycle_offsets[cycle])
            assert completed == pytest.approx(direct, abs=1e-12)


def test_uniform_hmm_reaches_the_analytic_relaxed_optimum():
    result = solve_structured_relaxation(
        [2.0, 8.0],
        [_two_state_appliance()],
        observation_variance=1.0,
        melding_weight=0.0,
        max_iterations=100,
        absolute_gap_tolerance=1e-10,
        relative_gap_tolerance=1e-10,
    )

    assert result.converged
    assert result.duality_gap <= 1e-8
    assert result.prediction.tolist() == pytest.approx([2.0, 8.0], abs=1e-8)
    assert result.objective == pytest.approx(2 * math.log(2), abs=1e-8)
    assert all(
        later <= earlier + 1e-11
        for earlier, later in zip(
            result.objective_history,
            result.objective_history[1:],
        )
    )
    _assert_hmm_polytope_feasible(result, 0)


def test_one_sample_zero_cycle_problem_handles_empty_transition_flow():
    appliance = _two_state_appliance(cycle_probabilities=[1.0])

    result = solve_structured_relaxation(
        [5.0],
        [appliance],
        observation_variance=1.0,
        melding_weight=0.5,
    )

    assert result.transitions[0].shape == (0, 2, 2)
    assert result.cycle_posteriors[0].tolist() == [1.0]
    _assert_hmm_polytope_feasible(result, 0)


def test_zero_melding_weight_reduces_to_afhmm_without_cycle_restrictions():
    aggregate = [0.0, 10.0, 0.0, 10.0]
    common = {
        "state_means": [0.0, 10.0],
        "initial_probabilities": [0.5, 0.5],
        "transition_probabilities": [[0.5, 0.5], [0.5, 0.5]],
    }

    afhmm = solve_structured_relaxation(
        aggregate,
        [StructuredAppliance(**common)],
        observation_variance=1.0,
        melding_weight=0.0,
    )
    lbm_disabled = solve_structured_relaxation(
        aggregate,
        [StructuredAppliance(**common, cycle_probabilities=[1.0])],
        observation_variance=1.0,
        melding_weight=0.0,
    )

    assert lbm_disabled.cycle_posteriors[0] is None
    assert lbm_disabled.objective == pytest.approx(afhmm.objective, abs=1e-12)
    assert torch.allclose(lbm_disabled.states[0], afhmm.states[0])
    assert lbm_disabled.prediction.tolist() == pytest.approx(aggregate, abs=1e-10)


def test_lbm_relaxation_preserves_flow_and_cycle_constraints():
    energy_summary = GaussianRatioSummary(
        state_weights=[0.0, 100.0],
        conditional_means=[0.0, 200.0, 400.0],
        conditional_variance=400.0,
        induced_mean=150.0,
        induced_variance=4000.0,
    )
    appliance = StructuredAppliance(
        state_means=[0.0, 100.0],
        initial_probabilities=[0.95, 0.05],
        transition_probabilities=[[0.9, 0.1], [0.1, 0.9]],
        off_state=0,
        cycle_probabilities=[0.05, 0.9, 0.05],
        summaries=(energy_summary,),
    )

    result = solve_structured_relaxation(
        [0.0, 0.0, 100.0, 100.0, 0.0],
        [appliance],
        observation_variance=25.0,
        melding_weight=0.8,
        max_iterations=2000,
        absolute_gap_tolerance=1e-6,
        relative_gap_tolerance=5e-5,
    )

    assert result.converged
    assert result.duality_gap <= 5e-4
    target = torch.tensor(
        [0.0, 0.0, 100.0, 100.0, 0.0],
        dtype=torch.float64,
    )
    assert torch.mean(torch.abs(result.prediction - target)) < 2.0
    assert all(
        later <= earlier + 1e-10
        for earlier, later in zip(
            result.objective_history,
            result.objective_history[1:],
        )
    )
    _assert_hmm_polytope_feasible(result, 0)


def test_two_appliances_can_mix_plain_hmm_and_cycle_aware_contracts():
    plain = _two_state_appliance()
    cycle_aware = StructuredAppliance(
        state_means=[0.0, 5.0],
        initial_probabilities=[0.5, 0.5],
        transition_probabilities=[[0.5, 0.5], [0.5, 0.5]],
        cycle_probabilities=[0.2, 0.6, 0.2],
    )

    result = solve_structured_relaxation(
        [0.0, 5.0, 15.0, 10.0, 0.0],
        [plain, cycle_aware],
        observation_variance=1.0,
        melding_weight=0.2,
        relative_gap_tolerance=5e-3,
    )

    assert result.converged
    assert result.cycle_posteriors[0] is None
    assert result.cycle_posteriors[1] is not None
    assert (
        torch.mean(
            torch.abs(
                result.prediction
                - torch.tensor([0.0, 5.0, 15.0, 10.0, 0.0], dtype=torch.float64)
            )
        )
        < 0.1
    )
    _assert_hmm_polytope_feasible(result, 0)
    _assert_hmm_polytope_feasible(result, 1)


def test_solver_is_deterministic_and_respects_iteration_cap():
    kwargs = {
        "aggregate": [3.0, 6.0, 9.0],
        "appliances": [_two_state_appliance()],
        "observation_variance": 1.0,
        "melding_weight": 0.0,
        "max_iterations": 1,
        "absolute_gap_tolerance": 0.0,
        "relative_gap_tolerance": 0.0,
    }

    first = solve_structured_relaxation(**kwargs)
    second = solve_structured_relaxation(**kwargs)

    assert first.iterations == 1
    assert not first.converged
    assert first.objective == second.objective
    assert first.duality_gap == second.duality_gap
    assert torch.equal(first.states[0], second.states[0])
    _assert_hmm_polytope_feasible(first, 0)


@pytest.mark.parametrize(
    ("appliance", "message"),
    [
        (
            _two_state_appliance(initial_probabilities=[1.0, 0.0]),
            "strictly positive",
        ),
        (
            _two_state_appliance(transition_probabilities=[[0.4, 0.4], [0.5, 0.5]]),
            "sum to one",
        ),
        (
            _two_state_appliance(off_state=2),
            "off_state",
        ),
        (
            _two_state_appliance(cycle_probabilities=[0.2, 0.3, 0.5]),
            "unreachable cycle counts",
        ),
    ],
)
def test_solver_rejects_invalid_hmm_contracts(appliance, message):
    with pytest.raises(ValueError, match=message):
        solve_structured_relaxation(
            [1.0, 2.0, 3.0],
            [appliance],
            observation_variance=1.0,
        )


def test_solver_rejects_a_nonconvex_gaussian_ratio():
    appliance = _two_state_appliance(
        cycle_probabilities=[0.5, 0.5],
        summaries=(
            GaussianRatioSummary(
                state_weights=[0.0, 10.0],
                conditional_means=[0.0, 10.0],
                conditional_variance=10.0,
                induced_mean=5.0,
                induced_variance=10.0,
            ),
        ),
    )

    with pytest.raises(ValueError, match="remains convex"):
        solve_structured_relaxation(
            [0.0, 1.0],
            [appliance],
            observation_variance=1.0,
        )


def test_solver_rejects_a_non_sequence_summary_collection():
    appliance = _two_state_appliance(summaries=None)

    with pytest.raises(TypeError, match="summaries must be a sequence"):
        solve_structured_relaxation(
            [0.0, 1.0],
            [appliance],
            observation_variance=1.0,
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"aggregate": []}, ValueError, "non-empty"),
        ({"aggregate": [1.0, -1.0]}, ValueError, "non-negative"),
        ({"observation_variance": 0.0}, ValueError, "positive"),
        ({"melding_weight": 1.1}, ValueError, "must not exceed one"),
        ({"max_iterations": True}, TypeError, "must be an integer"),
        ({"device": "meta"}, ValueError, "only CPU and CUDA"),
    ],
)
def test_solver_rejects_invalid_runtime_inputs(kwargs, error, message):
    params = {
        "aggregate": [1.0, 2.0],
        "appliances": [_two_state_appliance()],
        "observation_variance": 1.0,
    }
    params.update(kwargs)
    with pytest.raises(error, match=message):
        solve_structured_relaxation(**params)


def test_solver_rejects_numerical_overflow_instead_of_false_convergence():
    with pytest.raises(RuntimeError, match="non-finite objective or prediction"):
        solve_structured_relaxation(
            [1e308],
            [_two_state_appliance()],
            observation_variance=1.0,
        )


def test_structured_kernel_has_no_generic_solver_runtime_dependency():
    source = (
        __import__("pathlib").Path(__file__).parents[1]
        / "nilmtk_contrib"
        / "torch"
        / "_lbm.py"
    ).read_text(encoding="utf-8")
    for dependency in ("cvxpy", "mosek", "scipy"):
        assert f"import {dependency}" not in source
        assert f"from {dependency}" not in source


def test_structured_relaxation_runs_on_cuda_when_requested(gpu_tests_enabled):
    if not gpu_tests_enabled:
        pytest.skip("Set NILMTK_CONTRIB_RUN_GPU_TESTS=1 to run GPU checks.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable.")

    result = solve_structured_relaxation(
        torch.tensor([2.0, 8.0], device="cuda"),
        [_two_state_appliance()],
        observation_variance=1.0,
        melding_weight=0.0,
        device="cuda",
    )

    assert result.prediction.device.type == "cuda"
    _assert_hmm_polytope_feasible(result, 0)
