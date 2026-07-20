import inspect
from itertools import pairwise

import pytest


torch = pytest.importorskip("torch")

from nilmtk_contrib.torch._coordinate_factorial_hmm import (  # noqa: E402
    factorial_hmm_coordinate_viterbi,
)
from nilmtk_contrib.torch._factorial_hmm import (  # noqa: E402
    factorial_hmm_path_score,
    factorial_hmm_viterbi,
)
from nilmtk_contrib.torch._hmm import gaussian_hmm_viterbi  # noqa: E402


def _parameters(*, dtype=torch.float64, device="cpu"):
    means = (
        torch.tensor([0.0, 80.0], dtype=dtype, device=device),
        torch.tensor([0.0, 30.0], dtype=dtype, device=device),
    )
    initial = (
        torch.tensor([0.9, 0.1], dtype=dtype, device=device),
        torch.tensor([0.8, 0.2], dtype=dtype, device=device),
    )
    transition = (
        torch.tensor([[0.85, 0.15], [0.2, 0.8]], dtype=dtype, device=device),
        torch.tensor([[0.75, 0.25], [0.3, 0.7]], dtype=dtype, device=device),
    )
    return means, initial, transition


def test_known_path_matches_exact_fhmm_and_reports_convergence():
    parameters = _parameters()
    observations = torch.tensor([0.0, 30.0, 110.0, 80.0], dtype=torch.float64)

    approximate = factorial_hmm_coordinate_viterbi(
        observations, *parameters, noise_std=2.0
    )
    exact = factorial_hmm_viterbi(observations, *parameters, noise_std=2.0)

    assert approximate.converged
    assert approximate.iterations == len(approximate.score_history) - 1
    assert approximate.state_indices.tolist() == [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
    ]
    assert torch.equal(approximate.state_indices, exact.state_indices)
    assert torch.equal(approximate.appliance_power, exact.appliance_power)
    assert torch.equal(
        approximate.aggregate_power, approximate.appliance_power.sum(dim=1)
    )
    assert approximate.score == pytest.approx(exact.score, abs=1e-12)


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_exact_score_upper_bounds_monotonic_coordinate_score(seed):
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    means = (
        torch.tensor([0.0, 45.0], dtype=dtype),
        torch.tensor([0.0, 20.0], dtype=dtype),
    )
    initial = tuple(
        torch.softmax(torch.randn(2, generator=generator, dtype=dtype), 0)
        for _ in means
    )
    transition = tuple(
        torch.softmax(torch.randn(2, 2, generator=generator, dtype=dtype), 1)
        for _ in means
    )
    observations = torch.randn(6, generator=generator, dtype=dtype) * 18 + 30

    approximate = factorial_hmm_coordinate_viterbi(
        observations,
        means,
        initial,
        transition,
        noise_std=7.0,
        max_iterations=12,
    )
    exact = factorial_hmm_viterbi(
        observations, means, initial, transition, noise_std=7.0
    )
    rescored = factorial_hmm_path_score(
        approximate.state_indices,
        observations,
        means,
        initial,
        transition,
        noise_std=7.0,
    )

    assert all(right >= left for left, right in pairwise(approximate.score_history))
    assert approximate.score == pytest.approx(float(rescored), abs=1e-12)
    assert approximate.score <= exact.score + 1e-12


def test_single_appliance_is_exact_gaussian_hmm_inference():
    dtype = torch.float64
    observations = torch.tensor([0.0, 0.0, 75.0, 75.0, 0.0], dtype=dtype)
    means = (torch.tensor([75.0, 0.0], dtype=dtype),)
    initial = (torch.tensor([0.2, 0.8], dtype=dtype),)
    transition = (torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=dtype),)

    approximate = factorial_hmm_coordinate_viterbi(
        observations, means, initial, transition, noise_std=3.0
    )
    exact = gaussian_hmm_viterbi(
        observations, means[0], initial[0], transition[0], noise_std=3.0
    )

    assert approximate.converged
    assert torch.equal(approximate.state_indices[:, 0], exact.states)
    assert torch.equal(approximate.appliance_power[:, 0], exact.state_power)
    assert approximate.score == pytest.approx(exact.score, abs=1e-12)


def test_state_label_permutations_do_not_change_power_or_score():
    means, initial, transition = _parameters()
    observations = torch.tensor([0.0, 30.0, 110.0, 80.0], dtype=torch.float64)
    expected = factorial_hmm_coordinate_viterbi(
        observations, means, initial, transition, noise_std=3.0
    )
    orders = (torch.tensor([1, 0]), torch.tensor([1, 0]))
    permuted = (
        tuple(value[order] for value, order in zip(means, orders, strict=True)),
        tuple(value[order] for value, order in zip(initial, orders, strict=True)),
        tuple(
            value[order][:, order]
            for value, order in zip(transition, orders, strict=True)
        ),
    )

    actual = factorial_hmm_coordinate_viterbi(
        observations, *permuted, noise_std=3.0
    )

    assert torch.equal(actual.state_indices, expected.state_indices)
    assert torch.equal(actual.appliance_power, expected.appliance_power)
    assert actual.score == pytest.approx(expected.score, abs=1e-12)


def test_large_factorial_model_avoids_cartesian_state_construction():
    dtype = torch.float64
    appliance_count = 10
    means = tuple(torch.arange(4, dtype=dtype) * 10 for _ in range(appliance_count))
    initial = tuple(torch.tensor([0.7, 0.1, 0.1, 0.1], dtype=dtype) for _ in means)
    transition_matrix = torch.full((4, 4), 0.1 / 3, dtype=dtype)
    transition_matrix.diagonal().fill_(0.9)
    transition = tuple(transition_matrix.clone() for _ in means)
    observations = torch.tensor([0.0, 10.0, 20.0, 10.0], dtype=dtype)

    result = factorial_hmm_coordinate_viterbi(
        observations,
        means,
        initial,
        transition,
        noise_std=4.0,
        max_iterations=4,
    )

    assert result.state_indices.shape == (4, appliance_count)
    assert result.appliance_power.shape == (4, appliance_count)
    assert torch.isfinite(result.aggregate_power).all()
    with pytest.raises(ValueError, match="1048576 joint states"):
        factorial_hmm_viterbi(
            observations,
            means,
            initial,
            transition,
            noise_std=4.0,
            max_joint_states=512,
        )


def test_iteration_limit_reports_unconfirmed_convergence():
    parameters = _parameters()
    observations = torch.tensor([0.0, 30.0, 110.0, 80.0], dtype=torch.float64)

    result = factorial_hmm_coordinate_viterbi(
        observations, *parameters, noise_std=2.0, max_iterations=1
    )

    assert result.iterations == 1
    assert len(result.score_history) == 2
    assert not result.converged
    assert result.score_history[1] > result.score_history[0]


@pytest.mark.parametrize(
    ("maximum", "error", "message"),
    [
        (True, TypeError, "integer"),
        (2.5, TypeError, "integer"),
        (0, ValueError, "positive"),
    ],
)
def test_iteration_limit_must_be_a_positive_integer(maximum, error, message):
    parameters = _parameters()
    with pytest.raises(error, match=message):
        factorial_hmm_coordinate_viterbi(
            torch.zeros(2, dtype=torch.float64),
            *parameters,
            noise_std=1.0,
            max_iterations=maximum,
        )


def test_shared_validation_rejects_invalid_observations_and_parameters():
    means, initial, transition = _parameters()
    cases = [
        (torch.empty(0, dtype=torch.float64), means, initial, transition, "nonempty"),
        (
            torch.tensor([0.0, float("nan")], dtype=torch.float64),
            means,
            initial,
            transition,
            "finite",
        ),
        (
            torch.zeros(2, dtype=torch.float32),
            means,
            initial,
            transition,
            "dtype",
        ),
        (
            torch.zeros(2, dtype=torch.float64),
            (torch.tensor([-1.0, 2.0], dtype=torch.float64), means[1]),
            initial,
            transition,
            "nonnegative",
        ),
    ]
    for observations, case_means, case_initial, case_transition, message in cases:
        with pytest.raises(ValueError, match=message):
            factorial_hmm_coordinate_viterbi(
                observations,
                case_means,
                case_initial,
                case_transition,
                noise_std=2.0,
            )


def test_exact_ties_are_deterministic_and_do_not_cycle():
    dtype = torch.float64
    means = (torch.tensor([0.0, 0.0], dtype=dtype),)
    initial = (torch.tensor([0.5, 0.5], dtype=dtype),)
    transition = (torch.full((2, 2), 0.5, dtype=dtype),)
    observations = torch.zeros(3, dtype=dtype)

    first = factorial_hmm_coordinate_viterbi(
        observations, means, initial, transition, noise_std=1.0
    )
    second = factorial_hmm_coordinate_viterbi(
        observations, means, initial, transition, noise_std=1.0
    )

    assert first.converged
    assert first.iterations == 1
    assert first.state_indices.tolist() == [[0], [0], [0]]
    assert torch.equal(first.state_indices, second.state_indices)
    assert first.score_history == second.score_history


def test_approximation_has_no_cartesian_numpy_or_solver_dependency():
    import nilmtk_contrib.torch._coordinate_factorial_hmm as module

    source = inspect.getsource(module)

    for dependency in (
        "cartesian_prod",
        "numpy",
        "pandas",
        "cvxpy",
        "hmmlearn",
        "scipy",
    ):
        assert dependency not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_and_cuda_coordinate_decoding_agree():
    cpu_parameters = _parameters(dtype=torch.float32)
    observations = torch.tensor([0.0, 30.0, 110.0, 80.0])
    cpu = factorial_hmm_coordinate_viterbi(
        observations, *cpu_parameters, noise_std=3.0
    )
    cuda_parameters = tuple(
        tuple(value.cuda() for value in group) for group in cpu_parameters
    )
    cuda = factorial_hmm_coordinate_viterbi(
        observations.cuda(), *cuda_parameters, noise_std=3.0
    )

    assert torch.equal(cpu.state_indices, cuda.state_indices.cpu())
    assert torch.allclose(cpu.appliance_power, cuda.appliance_power.cpu())
    assert cpu.score == pytest.approx(cuda.score, abs=1e-5)
