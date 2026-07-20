from itertools import product
import inspect

import pytest


torch = pytest.importorskip("torch")

from nilmtk_contrib.torch._factorial_hmm import (  # noqa: E402
    canonicalize_factorial_hmm,
    factorial_hmm_path_score,
    factorial_hmm_viterbi,
)


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


def _enumerate_paths(observations, means, initial, transition, *, noise_std):
    appliance_count = len(means)
    state_counts = [value.numel() for value in means]
    joint_states = tuple(product(*(range(count) for count in state_counts)))
    candidates = []
    for flattened in product(joint_states, repeat=observations.numel()):
        path = torch.tensor(
            flattened,
            dtype=torch.int64,
            device=observations.device,
        ).reshape(observations.numel(), appliance_count)
        try:
            score = factorial_hmm_path_score(
                path,
                observations,
                means,
                initial,
                transition,
                noise_std=noise_std,
            )
        except ValueError:
            continue
        candidates.append((path, float(score)))
    return candidates


def test_canonicalization_sorts_states_and_all_probability_axes_together():
    means = (torch.tensor([70.0, 0.0, 20.0], dtype=torch.float64),)
    initial = (torch.tensor([0.1, 0.6, 0.3], dtype=torch.float64),)
    transition = (
        torch.tensor(
            [
                [0.7, 0.1, 0.2],
                [0.3, 0.6, 0.1],
                [0.2, 0.3, 0.5],
            ],
            dtype=torch.float64,
        ),
    )

    result = canonicalize_factorial_hmm(means, initial, transition)

    assert result.state_means[0].tolist() == [0.0, 20.0, 70.0]
    assert result.initial_probabilities[0].tolist() == [0.6, 0.3, 0.1]
    assert result.transition_probabilities[0].tolist() == [
        [0.6, 0.1, 0.3],
        [0.3, 0.5, 0.2],
        [0.1, 0.2, 0.7],
    ]
    assert torch.allclose(
        result.transition_probabilities[0].sum(1),
        torch.ones(3, dtype=torch.float64),
    )


def test_viterbi_recovers_a_known_two_appliance_path_and_power():
    means, initial, transition = _parameters()
    observations = torch.tensor([0.0, 30.0, 110.0, 80.0], dtype=torch.float64)

    result = factorial_hmm_viterbi(
        observations,
        means,
        initial,
        transition,
        noise_std=2.0,
    )

    assert result.state_indices.tolist() == [[0, 0], [0, 1], [1, 1], [1, 0]]
    assert result.appliance_power.tolist() == [
        [0.0, 0.0],
        [0.0, 30.0],
        [80.0, 30.0],
        [80.0, 0.0],
    ]
    assert torch.equal(result.aggregate_power, result.appliance_power.sum(dim=1))
    assert result.aggregate_power.tolist() == observations.tolist()


@pytest.mark.parametrize("seed", [2, 7, 19])
def test_viterbi_matches_exhaustive_enumeration_on_small_random_models(seed):
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    means = (
        torch.tensor([0.0, 50.0], dtype=dtype),
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
    observations = torch.randn(4, generator=generator, dtype=dtype) * 20 + 35
    candidates = _enumerate_paths(
        observations, means, initial, transition, noise_std=8.0
    )
    expected_path, expected_score = max(candidates, key=lambda item: item[1])

    result = factorial_hmm_viterbi(
        observations,
        means,
        initial,
        transition,
        noise_std=8.0,
    )

    assert result.score == pytest.approx(expected_score, abs=1e-12)
    assert torch.equal(result.state_indices, expected_path)
    actual_score = factorial_hmm_path_score(
        result.state_indices,
        observations,
        means,
        initial,
        transition,
        noise_std=8.0,
    )
    assert result.score == pytest.approx(float(actual_score), abs=1e-12)


def test_state_label_permutations_cannot_change_decoding_or_score():
    means, initial, transition = _parameters()
    observations = torch.tensor([0.0, 30.0, 110.0, 80.0], dtype=torch.float64)
    expected = factorial_hmm_viterbi(
        observations, means, initial, transition, noise_std=3.0
    )
    orders = (torch.tensor([1, 0]), torch.tensor([1, 0]))
    permuted_means = tuple(value[order] for value, order in zip(means, orders))
    permuted_initial = tuple(value[order] for value, order in zip(initial, orders))
    permuted_transition = tuple(
        value[order][:, order] for value, order in zip(transition, orders)
    )

    actual = factorial_hmm_viterbi(
        observations,
        permuted_means,
        permuted_initial,
        permuted_transition,
        noise_std=3.0,
    )

    assert torch.equal(actual.state_indices, expected.state_indices)
    assert torch.equal(actual.appliance_power, expected.appliance_power)
    assert actual.score == pytest.approx(expected.score, abs=1e-12)


def test_zero_probability_transitions_are_safe_and_forbidden_paths_fail_cleanly():
    dtype = torch.float64
    means = (torch.tensor([0.0, 10.0], dtype=dtype),)
    initial = (torch.tensor([1.0, 0.0], dtype=dtype),)
    transition = (torch.eye(2, dtype=dtype),)
    observations = torch.zeros(3, dtype=dtype)

    result = factorial_hmm_viterbi(
        observations, means, initial, transition, noise_std=1.0
    )

    assert result.state_indices.flatten().tolist() == [0, 0, 0]
    assert result.score == pytest.approx(result.score)
    forbidden = torch.tensor([[0], [1], [1]], dtype=torch.int64)
    with pytest.raises(ValueError, match="forbidden"):
        factorial_hmm_path_score(
            forbidden,
            observations,
            means,
            initial,
            transition,
            noise_std=1.0,
        )


def test_joint_state_guard_precedes_quadratic_allocation():
    dtype = torch.float64
    means = tuple(torch.arange(4, dtype=dtype) for _ in range(5))
    initial = tuple(torch.full((4,), 0.25, dtype=dtype) for _ in means)
    transition = tuple(torch.full((4, 4), 0.25, dtype=dtype) for _ in means)

    with pytest.raises(ValueError, match="1024 joint states"):
        factorial_hmm_viterbi(
            torch.zeros(2, dtype=dtype),
            means,
            initial,
            transition,
            noise_std=1.0,
            max_joint_states=512,
        )


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        (lambda m, i, t: ((), i, t), ValueError, "at least one appliance"),
        (lambda m, i, t: (m, i[:1], t), ValueError, "one entry per appliance"),
        (lambda m, i, t: (m, i, t[:1]), ValueError, "one entry per appliance"),
        (
            lambda m, i, t: (m, (torch.tensor([0.8, 0.8]), i[1]), t),
            ValueError,
            "sum to one",
        ),
        (
            lambda m, i, t: (m, i, (torch.tensor([[0.7, 0.2], [0.3, 0.7]]), t[1])),
            ValueError,
            "Every row",
        ),
        (
            lambda m, i, t: ((torch.tensor([-1.0, 2.0]), m[1]), i, t),
            ValueError,
            "nonnegative",
        ),
    ],
)
def test_parameter_validation_rejects_invalid_hmms(mutate, error, message):
    means, initial, transition = _parameters(dtype=torch.float32)
    arguments = mutate(means, initial, transition)

    with pytest.raises(error, match=message):
        canonicalize_factorial_hmm(*arguments)


def test_parameter_tensor_shape_dtype_and_finiteness_contracts():
    means, initial, transition = _parameters()
    cases = [
        (([0.0, 1.0], means[1]), initial, transition, TypeError, "torch.Tensor"),
        (
            (torch.tensor([0, 1]), means[1]),
            initial,
            transition,
            TypeError,
            "torch.float32 or torch.float64",
        ),
        (
            (torch.tensor([0.0, float("nan")], dtype=torch.float64), means[1]),
            initial,
            transition,
            ValueError,
            "finite",
        ),
        (
            (torch.tensor([0.0], dtype=torch.float64), means[1]),
            initial,
            transition,
            ValueError,
            "at least two states",
        ),
        (
            means,
            (initial[0].reshape(1, 2), initial[1]),
            transition,
            ValueError,
            "shape",
        ),
        (
            means,
            (torch.tensor([1.1, -0.1], dtype=torch.float64), initial[1]),
            transition,
            ValueError,
            "nonnegative",
        ),
        (
            means,
            initial,
            (transition[0][:, :1], transition[1]),
            ValueError,
            "shape",
        ),
        (
            means,
            initial,
            (
                torch.tensor([[1.1, -0.1], [0.2, 0.8]], dtype=torch.float64),
                transition[1],
            ),
            ValueError,
            "nonnegative",
        ),
        (
            (means[0], means[1].float()),
            initial,
            transition,
            ValueError,
            "dtype",
        ),
        (
            means,
            (initial[0].float(), initial[1]),
            transition,
            ValueError,
            "dtype",
        ),
    ]
    for case_means, case_initial, case_transition, error, message in cases:
        with pytest.raises(error, match=message):
            canonicalize_factorial_hmm(case_means, case_initial, case_transition)


@pytest.mark.parametrize(
    ("maximum", "error", "message"),
    [
        (True, TypeError, "integer"),
        (1.5, TypeError, "integer"),
        (0, ValueError, "positive"),
    ],
)
def test_joint_state_limit_is_a_positive_integer(maximum, error, message):
    means, initial, transition = _parameters()

    with pytest.raises(error, match=message):
        factorial_hmm_viterbi(
            torch.zeros(2, dtype=torch.float64),
            means,
            initial,
            transition,
            noise_std=1.0,
            max_joint_states=maximum,
        )


def test_observation_and_state_path_contracts():
    means, initial, transition = _parameters()
    observations = torch.zeros(2, dtype=torch.float64)
    decode_cases = [
        (torch.zeros(1, 2, dtype=torch.float64), ValueError, "one-dimensional"),
        (torch.empty(0, dtype=torch.float64), ValueError, "nonempty"),
        (torch.zeros(2, dtype=torch.float32), ValueError, "dtype"),
    ]
    for values, error, message in decode_cases:
        with pytest.raises(error, match=message):
            factorial_hmm_viterbi(values, means, initial, transition, noise_std=1.0)
    with pytest.raises(TypeError, match="positive finite"):
        factorial_hmm_viterbi(observations, means, initial, transition, noise_std=True)

    path_cases = [
        ([[0, 0], [0, 0]], TypeError, "torch.Tensor"),
        (torch.zeros(2, 2), TypeError, "integer"),
        (torch.zeros(2, dtype=torch.int64), TypeError, "two-dimensional"),
        (torch.zeros(1, 2, dtype=torch.int64), ValueError, "shape"),
        (torch.tensor([[0, 0], [2, 0]]), ValueError, "between 0 and 1"),
    ]
    for path, error, message in path_cases:
        with pytest.raises(error, match=message):
            factorial_hmm_path_score(
                path,
                observations,
                means,
                initial,
                transition,
                noise_std=1.0,
            )


def test_numerically_impossible_decode_fails_instead_of_returning_infinity():
    means, initial, transition = _parameters(dtype=torch.float32)

    with pytest.raises(ValueError, match="No valid"):
        factorial_hmm_viterbi(
            torch.full((2,), torch.finfo(torch.float32).max),
            means,
            initial,
            transition,
            noise_std=1e-30,
        )


@pytest.mark.parametrize("noise_std", [0.0, -1.0, float("inf"), float("nan")])
def test_noise_scale_must_be_positive_and_finite(noise_std):
    means, initial, transition = _parameters()

    with pytest.raises(ValueError, match="positive finite"):
        factorial_hmm_viterbi(
            torch.zeros(2, dtype=torch.float64),
            means,
            initial,
            transition,
            noise_std=noise_std,
        )


def test_decode_is_deterministic_under_exact_ties():
    dtype = torch.float64
    means = (torch.tensor([0.0, 0.0], dtype=dtype),)
    initial = (torch.tensor([0.5, 0.5], dtype=dtype),)
    transition = (torch.full((2, 2), 0.5, dtype=dtype),)
    observations = torch.zeros(3, dtype=dtype)

    first = factorial_hmm_viterbi(
        observations, means, initial, transition, noise_std=1.0
    )
    second = factorial_hmm_viterbi(
        observations, means, initial, transition, noise_std=1.0
    )

    assert first.state_indices.tolist() == [[0], [0], [0]]
    assert torch.equal(first.state_indices, second.state_indices)
    assert first.score == second.score


def test_oracle_has_no_numpy_solver_or_classical_hmm_dependency():
    import nilmtk_contrib.torch._factorial_hmm as module

    source = inspect.getsource(module)

    for dependency in ("numpy", "pandas", "cvxpy", "hmmlearn", "scipy"):
        assert dependency not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_and_cuda_decoding_agree():
    cpu_parameters = _parameters(dtype=torch.float32)
    observations = torch.tensor([0.0, 30.0, 110.0, 80.0])
    cpu = factorial_hmm_viterbi(observations, *cpu_parameters, noise_std=3.0)
    cuda_parameters = tuple(
        tuple(value.cuda() for value in group) for group in cpu_parameters
    )
    cuda = factorial_hmm_viterbi(observations.cuda(), *cuda_parameters, noise_std=3.0)

    assert torch.equal(cpu.state_indices, cuda.state_indices.cpu())
    assert torch.allclose(cpu.appliance_power, cuda.appliance_power.cpu())
    assert cpu.score == pytest.approx(cuda.score, abs=1e-5)
