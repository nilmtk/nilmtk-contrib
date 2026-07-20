from itertools import pairwise

import pytest


torch = pytest.importorskip("torch")

from nilmtk_contrib.torch._coordinate_factorial_hmm import (  # noqa: E402
    factorial_hmm_coordinate_viterbi,
)
from nilmtk_contrib.torch._hmm_fit import (  # noqa: E402
    fit_observed_gaussian_hmm,
)


def _two_state_sequences(*, dtype=torch.float64, device="cpu"):
    return (
        torch.tensor([0.0, 0.0, 100.0, 100.0], dtype=dtype, device=device),
        torch.tensor([0.0, 0.0, 100.0, 100.0], dtype=dtype, device=device),
    )


def test_known_states_produce_expected_smoothed_markov_parameters():
    result = fit_observed_gaussian_hmm(_two_state_sequences(), pseudocount=1.0)

    assert result.converged
    assert result.parameters.state_means.tolist() == [0.0, 100.0]
    assert result.parameters.initial_probabilities.tolist() == [0.75, 0.25]
    assert result.parameters.transition_probabilities.tolist() == [
        [0.5, 0.5],
        [0.25, 0.75],
    ]
    assert [states.tolist() for states in result.state_sequences] == [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ]


def test_sequence_boundaries_are_not_counted_as_transitions():
    sequences = (
        torch.tensor([0.0, 100.0], dtype=torch.float64),
        torch.tensor([0.0, 100.0], dtype=torch.float64),
    )

    result = fit_observed_gaussian_hmm(sequences, pseudocount=1.0)

    assert result.parameters.transition_probabilities.tolist() == [
        [0.25, 0.75],
        [0.5, 0.5],
    ]
    assert result.parameters.initial_probabilities.tolist() == [0.75, 0.25]


def test_three_state_fit_is_canonical_and_loss_is_monotonic():
    sequences = (
        torch.tensor(
            [0.0, 2.0, 1.0, 48.0, 52.0, 50.0, 98.0, 102.0, 100.0],
            dtype=torch.float64,
        ),
    )

    result = fit_observed_gaussian_hmm(sequences, n_states=3)

    assert result.converged
    assert result.iterations == len(result.loss_history) - 1
    assert result.parameters.state_means.tolist() == [1.0, 50.0, 100.0]
    assert all(right <= left for left, right in pairwise(result.loss_history))
    assert result.state_sequences[0].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_probability_contracts_hold_for_sparse_singleton_sequences():
    sequences = tuple(
        torch.tensor([value], dtype=torch.float64) for value in (0.0, 50.0, 100.0)
    )

    result = fit_observed_gaussian_hmm(
        sequences, n_states=3, pseudocount=0.25
    )
    parameters = result.parameters

    assert torch.all(parameters.initial_probabilities > 0)
    assert torch.all(parameters.transition_probabilities > 0)
    assert parameters.initial_probabilities.sum() == pytest.approx(1.0)
    assert torch.allclose(
        parameters.transition_probabilities.sum(dim=1),
        torch.ones(3, dtype=torch.float64),
    )


def test_sequence_order_cannot_change_fitted_parameters():
    first, second = _two_state_sequences()
    expected = fit_observed_gaussian_hmm((first, second))
    actual = fit_observed_gaussian_hmm((second, first))

    assert torch.equal(actual.parameters.state_means, expected.parameters.state_means)
    assert torch.equal(
        actual.parameters.initial_probabilities,
        expected.parameters.initial_probabilities,
    )
    assert torch.equal(
        actual.parameters.transition_probabilities,
        expected.parameters.transition_probabilities,
    )


def test_fit_is_deterministic_without_random_state():
    sequences = (
        torch.tensor([0.0, 1.0, 4.0, 48.0, 51.0, 99.0], dtype=torch.float64),
    )

    first = fit_observed_gaussian_hmm(sequences, n_states=3)
    second = fit_observed_gaussian_hmm(sequences, n_states=3)

    assert torch.equal(first.parameters.state_means, second.parameters.state_means)
    assert torch.equal(first.state_sequences[0], second.state_sequences[0])
    assert first.loss_history == second.loss_history
    assert first.iterations == second.iterations


def test_fitted_appliance_hmms_compose_with_coordinate_fhmm_inference():
    dtype = torch.float64
    appliance_traces = (
        torch.tensor([0.0, 0.0, 80.0, 80.0], dtype=dtype),
        torch.tensor([0.0, 30.0, 30.0, 0.0], dtype=dtype),
    )
    fits = tuple(
        fit_observed_gaussian_hmm((trace,)) for trace in appliance_traces
    )
    result = factorial_hmm_coordinate_viterbi(
        appliance_traces[0] + appliance_traces[1],
        tuple(fit.parameters.state_means for fit in fits),
        tuple(fit.parameters.initial_probabilities for fit in fits),
        tuple(fit.parameters.transition_probabilities for fit in fits),
        noise_std=1.0,
    )

    assert result.converged
    assert torch.equal(
        result.appliance_power,
        torch.stack(appliance_traces, dim=1),
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"n_states": True}, TypeError, "integer"),
        ({"n_states": 1}, ValueError, "at least two"),
        ({"max_iterations": 0}, ValueError, "positive"),
        ({"max_iterations": 1.5}, TypeError, "integer"),
        ({"tolerance": True}, TypeError, "nonnegative"),
        ({"tolerance": -1.0}, ValueError, "nonnegative"),
        ({"tolerance": float("nan")}, ValueError, "nonnegative"),
        ({"pseudocount": 0.0}, ValueError, "positive"),
        ({"pseudocount": float("inf")}, ValueError, "positive"),
    ],
)
def test_hyperparameter_contracts(kwargs, error, message):
    with pytest.raises(error, match=message):
        fit_observed_gaussian_hmm(_two_state_sequences(), **kwargs)


@pytest.mark.parametrize(
    ("sequences", "error", "message"),
    [
        ((), ValueError, "at least one"),
        (([0.0, 1.0],), TypeError, "torch.Tensor"),
        ((torch.tensor([], dtype=torch.float64),), ValueError, "nonempty"),
        ((torch.zeros(1, 2, dtype=torch.float64),), ValueError, "one-dimensional"),
        ((torch.tensor([0, 1]),), TypeError, "torch.float32 or torch.float64"),
        (
            (torch.tensor([0.0, float("nan")], dtype=torch.float64),),
            ValueError,
            "finite",
        ),
        (
            (torch.tensor([-1.0, 0.0], dtype=torch.float64),),
            ValueError,
            "nonnegative",
        ),
        (
            (
                torch.tensor([0.0, 1.0], dtype=torch.float64),
                torch.tensor([0.0, 1.0], dtype=torch.float32),
            ),
            ValueError,
            "dtype",
        ),
    ],
)
def test_sequence_contracts(sequences, error, message):
    with pytest.raises(error, match=message):
        fit_observed_gaussian_hmm(sequences)


def test_insufficient_distinct_power_values_fail_cleanly():
    sequences = (torch.tensor([0.0, 0.0, 100.0], dtype=torch.float64),)

    with pytest.raises(ValueError, match="requires at least 3 distinct"):
        fit_observed_gaussian_hmm(sequences, n_states=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_and_cuda_fits_agree_under_strict_determinism():
    cpu_sequences = _two_state_sequences(dtype=torch.float32)
    torch.use_deterministic_algorithms(True)
    try:
        cpu = fit_observed_gaussian_hmm(cpu_sequences)
        cuda = fit_observed_gaussian_hmm(
            tuple(sequence.cuda() for sequence in cpu_sequences)
        )
    finally:
        torch.use_deterministic_algorithms(False)

    assert torch.equal(cpu.parameters.state_means, cuda.parameters.state_means.cpu())
    assert torch.equal(
        cpu.parameters.initial_probabilities,
        cuda.parameters.initial_probabilities.cpu(),
    )
    assert torch.equal(
        cpu.parameters.transition_probabilities,
        cuda.parameters.transition_probabilities.cpu(),
    )
    assert cpu.loss_history == cuda.loss_history
