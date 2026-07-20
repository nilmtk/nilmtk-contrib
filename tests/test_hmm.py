from itertools import product
import inspect

import pytest


torch = pytest.importorskip("torch")

from nilmtk_contrib.torch._hmm import (  # noqa: E402
    canonicalize_gaussian_hmm,
    gaussian_hmm_path_score,
    gaussian_hmm_viterbi,
    gaussian_log_emissions,
    hmm_path_score,
    hmm_viterbi,
)


def _log_scores(*, dtype=torch.float64, device="cpu"):
    emission = torch.tensor(
        [[0.7, -0.2], [0.5, 0.1], [-0.4, 0.9], [0.0, 0.8]],
        dtype=dtype,
        device=device,
    )
    initial = torch.log(torch.tensor([0.8, 0.2], dtype=dtype, device=device))
    transition = torch.log(
        torch.tensor([[0.75, 0.25], [0.3, 0.7]], dtype=dtype, device=device)
    )
    return emission, initial, transition


def _gaussian_parameters(*, dtype=torch.float64, device="cpu"):
    return (
        torch.tensor([0.0, 100.0], dtype=dtype, device=device),
        torch.tensor([0.9, 0.1], dtype=dtype, device=device),
        torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=dtype, device=device),
    )


def test_exact_hmm_matches_exhaustive_path_enumeration():
    scores = _log_scores()
    candidates = []
    for labels in product(range(2), repeat=4):
        states = torch.tensor(labels, dtype=torch.int64)
        candidates.append((states, float(hmm_path_score(states, *scores))))
    expected_states, expected_score = max(candidates, key=lambda item: item[1])

    actual = hmm_viterbi(*scores)

    assert actual.score == pytest.approx(expected_score, abs=1e-12)
    assert torch.equal(actual.states, expected_states)


def test_path_score_has_the_standard_initial_emission_transition_order():
    emission, initial, transition = _log_scores()
    states = torch.tensor([0, 0, 1, 1])

    actual = hmm_path_score(states, emission, initial, transition)
    expected = (
        initial[0]
        + emission[0, 0]
        + transition[0, 0]
        + emission[1, 0]
        + transition[0, 1]
        + emission[2, 1]
        + transition[1, 1]
        + emission[3, 1]
    )

    assert float(actual) == pytest.approx(float(expected), abs=1e-12)


def test_path_score_is_differentiable_with_finite_gradients():
    emission, initial, transition = _log_scores()
    emission.requires_grad_()
    initial.requires_grad_()
    transition.requires_grad_()

    score = hmm_path_score(torch.tensor([0, 0, 1, 1]), emission, initial, transition)
    (-score).backward()

    for value in (emission, initial, transition):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_gaussian_hmm_recovers_known_state_path_and_power():
    parameters = _gaussian_parameters()
    observations = torch.tensor([0.0, 1.0, 99.0, 100.0], dtype=torch.float64)

    result = gaussian_hmm_viterbi(observations, *parameters, noise_std=2.0)

    assert result.states.tolist() == [0, 0, 1, 1]
    assert result.state_power.tolist() == [0.0, 0.0, 100.0, 100.0]
    assert result.score == pytest.approx(
        float(
            gaussian_hmm_path_score(
                result.states,
                observations,
                *parameters,
                noise_std=2.0,
            )
        ),
        abs=1e-12,
    )


def test_gaussian_state_label_permutation_cannot_change_decode():
    means, initial, transition = _gaussian_parameters()
    observations = torch.tensor([0.0, 1.0, 99.0, 100.0], dtype=torch.float64)
    expected = gaussian_hmm_viterbi(
        observations, means, initial, transition, noise_std=2.0
    )
    order = torch.tensor([1, 0])

    actual = gaussian_hmm_viterbi(
        observations,
        means[order],
        initial[order],
        transition[order][:, order],
        noise_std=2.0,
    )

    assert torch.equal(actual.states, expected.states)
    assert torch.equal(actual.state_power, expected.state_power)
    assert actual.score == pytest.approx(expected.score, abs=1e-12)


def test_gaussian_canonicalization_permutates_every_probability_axis():
    means = torch.tensor([30.0, -2.0, 10.0], dtype=torch.float64)
    initial = torch.tensor([0.1, 0.6, 0.3], dtype=torch.float64)
    transition = torch.tensor(
        [[0.7, 0.1, 0.2], [0.3, 0.6, 0.1], [0.2, 0.3, 0.5]],
        dtype=torch.float64,
    )

    actual = canonicalize_gaussian_hmm(means, initial, transition)

    assert actual.state_means.tolist() == [-2.0, 10.0, 30.0]
    assert actual.initial_probabilities.tolist() == [0.6, 0.3, 0.1]
    assert actual.transition_probabilities.tolist() == [
        [0.6, 0.1, 0.3],
        [0.3, 0.5, 0.2],
        [0.1, 0.2, 0.7],
    ]


def test_zero_probabilities_are_forbidden_without_nan_scores():
    dtype = torch.float64
    emissions = torch.zeros((3, 2), dtype=dtype)
    initial = torch.tensor([0.0, -torch.inf], dtype=dtype)
    transition = torch.tensor([[0.0, -torch.inf], [-torch.inf, 0.0]], dtype=dtype)

    result = hmm_viterbi(emissions, initial, transition)

    assert result.states.tolist() == [0, 0, 0]
    assert result.score == pytest.approx(0.0)
    with pytest.raises(ValueError, match="forbidden"):
        hmm_path_score(torch.tensor([0, 1, 1]), emissions, initial, transition)


def test_impossible_hmm_fails_instead_of_returning_negative_infinity():
    emissions = torch.tensor(
        [[0.0, -torch.inf], [-torch.inf, 0.0]], dtype=torch.float64
    )
    initial = torch.tensor([0.0, -torch.inf], dtype=torch.float64)
    transition = torch.tensor(
        [[0.0, -torch.inf], [-torch.inf, 0.0]], dtype=torch.float64
    )

    with pytest.raises(ValueError, match="No valid HMM path"):
        hmm_viterbi(emissions, initial, transition)


def test_exact_ties_have_a_deterministic_first_state_path():
    scores = (
        torch.zeros((3, 2), dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        torch.zeros((2, 2), dtype=torch.float64),
    )

    first = hmm_viterbi(*scores)
    second = hmm_viterbi(*scores)

    assert first.states.tolist() == [0, 0, 0]
    assert torch.equal(first.states, second.states)
    assert first.score == second.score


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        (lambda e, i, t: (e.tolist(), i, t), TypeError, "torch.Tensor"),
        (lambda e, i, t: (e.long(), i, t), TypeError, "float32 or torch.float64"),
        (lambda e, i, t: (e[0], i, t), ValueError, "shape"),
        (lambda e, i, t: (e, i[:1], t), ValueError, "initial_scores"),
        (lambda e, i, t: (e, i, t[:, :1]), ValueError, "transition_scores"),
        (
            lambda e, i, t: (e, torch.full_like(i, -torch.inf), t),
            ValueError,
            "at least one state",
        ),
        (
            lambda e, i, t: (e.clone().fill_(float("nan")), i, t),
            ValueError,
            "NaN",
        ),
    ],
)
def test_log_score_validation_rejects_malformed_models(mutate, error, message):
    with pytest.raises(error, match=message):
        hmm_viterbi(*mutate(*_log_scores()))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda m, i, t: (m[:1], i, t), "at least two states"),
        (lambda m, i, t: (m, i.reshape(1, 2), t), "shape"),
        (lambda m, i, t: (m, torch.tensor([0.8, 0.8]), t), "sum to one"),
        (
            lambda m, i, t: (m, torch.tensor([1.1, -0.1]), t),
            "nonnegative",
        ),
        (lambda m, i, t: (m, i, t[:, :1]), "shape"),
        (
            lambda m, i, t: (
                m,
                i,
                torch.tensor([[0.8, 0.1], [0.1, 0.9]]),
            ),
            "Every row",
        ),
    ],
)
def test_gaussian_probability_validation_rejects_malformed_models(mutate, message):
    with pytest.raises(ValueError, match=message):
        canonicalize_gaussian_hmm(*mutate(*_gaussian_parameters(dtype=torch.float32)))


@pytest.mark.parametrize("noise_std", [0.0, -1.0, float("inf"), float("nan")])
def test_gaussian_noise_scale_must_be_positive_and_finite(noise_std):
    means, _, _ = _gaussian_parameters()

    with pytest.raises(ValueError, match="positive finite"):
        gaussian_log_emissions(
            torch.zeros(2, dtype=torch.float64), means, noise_std=noise_std
        )


def test_gaussian_emission_contract_rejects_bad_shapes_values_and_scale_types():
    means, _, _ = _gaussian_parameters()
    cases = [
        (torch.zeros(2, dtype=torch.float64), means[:1], 1.0, "two states"),
        (torch.zeros(1, 2, dtype=torch.float64), means, 1.0, "one-dimensional"),
        (
            torch.tensor([0.0, float("nan")], dtype=torch.float64),
            means,
            1.0,
            "finite",
        ),
    ]
    for observations, state_means, scale, message in cases:
        with pytest.raises(ValueError, match=message):
            gaussian_log_emissions(observations, state_means, noise_std=scale)
    with pytest.raises(TypeError, match="positive finite"):
        gaussian_log_emissions(
            torch.zeros(2, dtype=torch.float64), means, noise_std=True
        )


def test_state_path_contract_rejects_wrong_type_shape_length_and_labels():
    scores = _log_scores()
    cases = [
        ([0, 0, 1, 1], TypeError, "torch.Tensor"),
        (torch.zeros(4), TypeError, "integer"),
        (torch.zeros(2, 2, dtype=torch.int64), TypeError, "one-dimensional"),
        (torch.zeros(3, dtype=torch.int64), ValueError, "exactly 4"),
        (torch.tensor([0, 0, 1, 2]), ValueError, "between 0 and 1"),
    ]
    for states, error, message in cases:
        with pytest.raises(error, match=message):
            hmm_path_score(states, *scores)


def test_hmm_core_has_no_numpy_solver_or_probabilistic_package_dependency():
    import nilmtk_contrib.torch._hmm as module

    source = inspect.getsource(module)

    for dependency in (
        "numpy",
        "pandas",
        "cvxpy",
        "hmmlearn",
        "pomegranate",
        "pyro",
        "scipy",
    ):
        assert dependency not in source


def test_factorial_hmm_delegates_decoding_to_the_shared_hmm(monkeypatch):
    import nilmtk_contrib.torch._factorial_hmm as factorial_module

    calls = []
    exact_decoder = factorial_module.hmm_viterbi

    def recording_decoder(*args):
        calls.append(tuple(value.shape for value in args))
        return exact_decoder(*args)

    monkeypatch.setattr(factorial_module, "hmm_viterbi", recording_decoder)
    means, initial, transition = _gaussian_parameters()
    result = factorial_module.factorial_hmm_viterbi(
        torch.tensor([0.0, 100.0], dtype=torch.float64),
        (means,),
        (initial,),
        (transition,),
        noise_std=2.0,
    )

    assert result.state_indices.flatten().tolist() == [0, 1]
    assert calls == [((2, 2), (2,), (2, 2))]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_and_cuda_gaussian_decoding_agree():
    observations = torch.tensor([0.0, 1.0, 99.0, 100.0], dtype=torch.float32)
    parameters = _gaussian_parameters(dtype=torch.float32)
    cpu = gaussian_hmm_viterbi(observations, *parameters, noise_std=2.0)
    cuda_parameters = tuple(value.cuda() for value in parameters)
    cuda = gaussian_hmm_viterbi(observations.cuda(), *cuda_parameters, noise_std=2.0)

    assert torch.equal(cpu.states, cuda.states.cpu())
    assert torch.allclose(cpu.state_power, cuda.state_power.cpu())
    assert cpu.score == pytest.approx(cuda.score, abs=1e-5)
