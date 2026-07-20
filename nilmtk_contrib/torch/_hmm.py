"""Exact log-space HMM primitives shared by NILM state-space models."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


_FLOAT_DTYPES = frozenset({torch.float32, torch.float64})
_INTEGER_DTYPES = frozenset(
    {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
)


@dataclass(frozen=True)
class HMMViterbiResult:
    """Highest-scoring state path under supplied log scores."""

    states: torch.Tensor
    score: float


@dataclass(frozen=True)
class GaussianHMMParameters:
    """Validated Gaussian HMM parameters ordered by increasing state mean."""

    state_means: torch.Tensor
    initial_probabilities: torch.Tensor
    transition_probabilities: torch.Tensor


@dataclass(frozen=True)
class GaussianHMMViterbiResult:
    """Highest-scoring Gaussian HMM path and reconstructed state means."""

    states: torch.Tensor
    state_power: torch.Tensor
    score: float


def _floating_tensor(name, value, *, reference=None):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.dtype not in _FLOAT_DTYPES or value.is_complex():
        raise TypeError(f"{name} must use torch.float32 or torch.float64.")
    if reference is not None and (
        value.dtype != reference.dtype or value.device != reference.device
    ):
        raise ValueError(
            f"{name} must use dtype {reference.dtype} and device {reference.device}."
        )
    return value


def _log_score_tensor(name, value, *, reference=None):
    value = _floating_tensor(name, value, reference=reference)
    if bool(torch.isnan(value).any()) or bool(torch.isposinf(value).any()):
        raise ValueError(f"{name} must not contain NaN or positive infinity.")
    return value


def _validate_log_scores(emission_scores, initial_scores, transition_scores):
    emissions = _log_score_tensor("emission_scores", emission_scores)
    if emissions.ndim != 2 or emissions.shape[0] < 1 or emissions.shape[1] < 2:
        raise ValueError(
            "emission_scores must have shape (time, states) with time >= 1 "
            "and states >= 2."
        )
    time_points, states = emissions.shape
    initial = _log_score_tensor("initial_scores", initial_scores, reference=emissions)
    transition = _log_score_tensor(
        "transition_scores", transition_scores, reference=emissions
    )
    if tuple(initial.shape) != (states,):
        raise ValueError(f"initial_scores must have shape ({states},).")
    if tuple(transition.shape) != (states, states):
        raise ValueError(f"transition_scores must have shape ({states}, {states}).")
    if not bool(torch.isfinite(initial).any()):
        raise ValueError("initial_scores must allow at least one state.")
    return int(time_points), int(states)


@torch.no_grad()
def hmm_viterbi(
    emission_scores,
    initial_scores,
    transition_scores,
) -> HMMViterbiResult:
    """Decode an exact first-order HMM from log-domain model scores."""

    time_points, states = _validate_log_scores(
        emission_scores, initial_scores, transition_scores
    )
    scores = emission_scores.new_empty((time_points, states))
    backpointers = torch.zeros(
        (time_points, states), dtype=torch.int64, device=emission_scores.device
    )
    scores[0] = initial_scores + emission_scores[0]
    for time in range(1, time_points):
        candidates = scores[time - 1][:, None] + transition_scores
        best_scores, best_sources = torch.max(candidates, dim=0)
        scores[time] = best_scores + emission_scores[time]
        backpointers[time] = best_sources

    final_state = int(torch.argmax(scores[-1]))
    final_score = scores[-1, final_state]
    if not bool(torch.isfinite(final_score)):
        raise ValueError("No valid HMM path reaches the sequence end.")
    decoded = torch.empty(time_points, dtype=torch.int64, device=emission_scores.device)
    decoded[-1] = final_state
    for time in range(time_points - 1, 0, -1):
        decoded[time - 1] = backpointers[time, decoded[time]]
    return HMMViterbiResult(states=decoded, score=float(final_score))


def hmm_path_score(states, emission_scores, initial_scores, transition_scores):
    """Return the log score of one state path under the same HMM convention."""

    time_points, state_count = _validate_log_scores(
        emission_scores, initial_scores, transition_scores
    )
    if not isinstance(states, torch.Tensor):
        raise TypeError("states must be a torch.Tensor.")
    if states.dtype not in _INTEGER_DTYPES or states.ndim != 1:
        raise TypeError("states must be a one-dimensional integer tensor.")
    if states.device != emission_scores.device:
        raise ValueError(f"states must be on {emission_scores.device}.")
    if states.numel() != time_points:
        raise ValueError(f"states must contain exactly {time_points} labels.")
    if bool(((states < 0) | (states >= state_count)).any()):
        raise ValueError(f"states labels must be between 0 and {state_count - 1}.")

    time = torch.arange(time_points, device=emission_scores.device)
    score = initial_scores[states[0]] + emission_scores[time, states].sum()
    if time_points > 1:
        score = score + transition_scores[states[:-1], states[1:]].sum()
    if not bool(torch.isfinite(score)):
        raise ValueError("states describe a path forbidden by the HMM scores.")
    return score


def _validate_probability_vector(name, value):
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise ValueError(f"{name} must contain finite nonnegative values.")
    tolerance = 1e-6 if value.dtype == torch.float32 else 1e-12
    if not bool(
        torch.isclose(value.sum(), value.new_tensor(1.0), atol=tolerance, rtol=0)
    ):
        raise ValueError(f"{name} must sum to one.")


def _validate_transition_probabilities(name, value, states):
    if tuple(value.shape) != (states, states):
        raise ValueError(f"{name} must have shape ({states}, {states}).")
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise ValueError(f"{name} must contain finite nonnegative values.")
    tolerance = 1e-6 if value.dtype == torch.float32 else 1e-12
    expected = torch.ones(states, dtype=value.dtype, device=value.device)
    if not bool(torch.allclose(value.sum(dim=1), expected, atol=tolerance, rtol=0)):
        raise ValueError(f"Every row of {name} must sum to one.")


def canonicalize_gaussian_hmm(
    state_means,
    initial_probabilities,
    transition_probabilities,
) -> GaussianHMMParameters:
    """Validate a Gaussian HMM and sort its states by increasing mean."""

    means = _floating_tensor("state_means", state_means)
    initial = _floating_tensor(
        "initial_probabilities", initial_probabilities, reference=means
    )
    transition = _floating_tensor(
        "transition_probabilities", transition_probabilities, reference=means
    )
    if means.ndim != 1 or means.numel() < 2:
        raise ValueError("state_means must contain at least two states.")
    if not bool(torch.isfinite(means).all()):
        raise ValueError("state_means must contain only finite values.")
    states = int(means.numel())
    if tuple(initial.shape) != (states,):
        raise ValueError(f"initial_probabilities must have shape ({states},).")
    _validate_probability_vector("initial_probabilities", initial)
    _validate_transition_probabilities("transition_probabilities", transition, states)
    order = torch.argsort(means, stable=True)
    return GaussianHMMParameters(
        state_means=means[order],
        initial_probabilities=initial[order],
        transition_probabilities=transition[order][:, order],
    )


def gaussian_log_emissions(observations, state_means, *, noise_std):
    """Return Gaussian log-emission scores for observations and state means."""

    means = _floating_tensor("state_means", state_means)
    values = _floating_tensor("observations", observations, reference=means)
    if means.ndim != 1 or means.numel() < 2:
        raise ValueError("state_means must contain at least two states.")
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("observations must be a nonempty one-dimensional tensor.")
    if not bool(torch.isfinite(means).all()) or not bool(torch.isfinite(values).all()):
        raise ValueError("observations and state_means must contain finite values.")
    if isinstance(noise_std, bool) or not isinstance(noise_std, (int, float)):
        raise TypeError("noise_std must be a positive finite number.")
    if not math.isfinite(noise_std) or noise_std <= 0:
        raise ValueError("noise_std must be a positive finite number.")
    scale = values.new_tensor(noise_std)
    residual = (values[:, None] - means[None, :]) / scale
    return -0.5 * residual.square() - torch.log(scale) - 0.5 * math.log(2.0 * math.pi)


def gaussian_hmm_viterbi(
    observations,
    state_means,
    initial_probabilities,
    transition_probabilities,
    *,
    noise_std,
) -> GaussianHMMViterbiResult:
    """Canonicalize and exactly decode a fixed-variance Gaussian HMM."""

    parameters = canonicalize_gaussian_hmm(
        state_means, initial_probabilities, transition_probabilities
    )
    emissions = gaussian_log_emissions(
        observations, parameters.state_means, noise_std=noise_std
    )
    result = hmm_viterbi(
        emissions,
        torch.log(parameters.initial_probabilities),
        torch.log(parameters.transition_probabilities),
    )
    return GaussianHMMViterbiResult(
        states=result.states,
        state_power=parameters.state_means[result.states],
        score=result.score,
    )


def gaussian_hmm_path_score(
    states,
    observations,
    state_means,
    initial_probabilities,
    transition_probabilities,
    *,
    noise_std,
):
    """Score one canonical-state path through a fixed-variance Gaussian HMM."""

    parameters = canonicalize_gaussian_hmm(
        state_means, initial_probabilities, transition_probabilities
    )
    emissions = gaussian_log_emissions(
        observations, parameters.state_means, noise_std=noise_std
    )
    return hmm_path_score(
        states,
        emissions,
        torch.log(parameters.initial_probabilities),
        torch.log(parameters.transition_probabilities),
    )


__all__ = [
    "GaussianHMMParameters",
    "GaussianHMMViterbiResult",
    "HMMViterbiResult",
    "canonicalize_gaussian_hmm",
    "gaussian_hmm_path_score",
    "gaussian_hmm_viterbi",
    "gaussian_log_emissions",
    "hmm_path_score",
    "hmm_viterbi",
]
