"""Exact factorial-HMM primitives for small models and correctness tests.

This private module is an oracle for the forthcoming PyTorch AFHMM port.  It
uses exact joint-state Viterbi decoding, so callers must explicitly bound the
Cartesian state space.  Transition matrices use the conventional
``[source_state, destination_state]`` orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


_FLOAT_DTYPES = frozenset({torch.float32, torch.float64})
_INTEGER_DTYPES = frozenset(
    {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
)


@dataclass(frozen=True)
class FactorialHMMParameters:
    """Canonical per-appliance HMM parameters ordered by increasing power."""

    state_means: tuple[torch.Tensor, ...]
    initial_probabilities: tuple[torch.Tensor, ...]
    transition_probabilities: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class FactorialHMMViterbiResult:
    """Highest-scoring joint path and its reconstructed appliance powers."""

    state_indices: torch.Tensor
    appliance_power: torch.Tensor
    aggregate_power: torch.Tensor
    score: float


def _floating_tensor(name: str, value) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.dtype not in _FLOAT_DTYPES or value.is_complex():
        raise TypeError(f"{name} must use torch.float32 or torch.float64.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _validate_probability_vector(name: str, value: torch.Tensor) -> None:
    if bool((value < 0).any()):
        raise ValueError(f"{name} must be nonnegative.")
    tolerance = 1e-6 if value.dtype == torch.float32 else 1e-12
    if not bool(
        torch.isclose(value.sum(), value.new_tensor(1.0), atol=tolerance, rtol=0)
    ):
        raise ValueError(f"{name} must sum to one.")


def _validate_transition_matrix(
    name: str,
    value: torch.Tensor,
    states: int,
) -> None:
    if tuple(value.shape) != (states, states):
        raise ValueError(f"{name} must have shape ({states}, {states}).")
    if bool((value < 0).any()):
        raise ValueError(f"{name} must be nonnegative.")
    tolerance = 1e-6 if value.dtype == torch.float32 else 1e-12
    expected = torch.ones(states, dtype=value.dtype, device=value.device)
    if not bool(torch.allclose(value.sum(dim=1), expected, atol=tolerance, rtol=0)):
        raise ValueError(f"Every row of {name} must sum to one.")


def canonicalize_factorial_hmm(
    state_means: Sequence[torch.Tensor],
    initial_probabilities: Sequence[torch.Tensor],
    transition_probabilities: Sequence[torch.Tensor],
) -> FactorialHMMParameters:
    """Validate parameters and sort every appliance state by increasing power."""

    if not isinstance(state_means, (tuple, list)) or not state_means:
        raise ValueError("state_means must contain at least one appliance.")
    appliance_count = len(state_means)
    if len(initial_probabilities) != appliance_count:
        raise ValueError("initial_probabilities must have one entry per appliance.")
    if len(transition_probabilities) != appliance_count:
        raise ValueError("transition_probabilities must have one entry per appliance.")

    canonical_means = []
    canonical_initial = []
    canonical_transition = []
    reference = None
    for appliance, (means_value, initial_value, transition_value) in enumerate(
        zip(state_means, initial_probabilities, transition_probabilities, strict=True)
    ):
        means = _floating_tensor(f"state_means[{appliance}]", means_value)
        initial = _floating_tensor(f"initial_probabilities[{appliance}]", initial_value)
        transition = _floating_tensor(
            f"transition_probabilities[{appliance}]", transition_value
        )
        if means.ndim != 1 or means.numel() < 2:
            raise ValueError(
                f"state_means[{appliance}] must contain at least two states."
            )
        if bool((means < 0).any()):
            raise ValueError(f"state_means[{appliance}] must be nonnegative.")
        if reference is None:
            reference = means
        for name, value in (
            (f"initial_probabilities[{appliance}]", initial),
            (f"transition_probabilities[{appliance}]", transition),
        ):
            if value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError(
                    f"{name} must use dtype {reference.dtype} and device "
                    f"{reference.device}."
                )
        if means.dtype != reference.dtype or means.device != reference.device:
            raise ValueError(
                f"state_means[{appliance}] must use dtype {reference.dtype} and "
                f"device {reference.device}."
            )
        states = int(means.numel())
        if tuple(initial.shape) != (states,):
            raise ValueError(
                f"initial_probabilities[{appliance}] must have shape ({states},)."
            )
        _validate_probability_vector(f"initial_probabilities[{appliance}]", initial)
        _validate_transition_matrix(
            f"transition_probabilities[{appliance}]", transition, states
        )

        order = torch.argsort(means, stable=True)
        canonical_means.append(means[order])
        canonical_initial.append(initial[order])
        canonical_transition.append(transition[order][:, order])

    return FactorialHMMParameters(
        state_means=tuple(canonical_means),
        initial_probabilities=tuple(canonical_initial),
        transition_probabilities=tuple(canonical_transition),
    )


def _joint_model(
    parameters: FactorialHMMParameters,
    *,
    max_joint_states: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(max_joint_states, bool) or not isinstance(max_joint_states, int):
        raise TypeError("max_joint_states must be an integer.")
    if max_joint_states < 1:
        raise ValueError("max_joint_states must be positive.")
    joint_count = math.prod(means.numel() for means in parameters.state_means)
    if joint_count > max_joint_states:
        raise ValueError(
            f"The factorial HMM has {joint_count} joint states, exceeding "
            f"max_joint_states={max_joint_states}."
        )

    reference = parameters.state_means[0]
    ranges = [
        torch.arange(means.numel(), device=reference.device)
        for means in parameters.state_means
    ]
    joint_states = torch.cartesian_prod(*ranges).reshape(joint_count, -1)
    joint_means = reference.new_zeros(joint_count)
    joint_initial = reference.new_zeros(joint_count)
    joint_transition = reference.new_zeros((joint_count, joint_count))
    for appliance, (means, initial, transition) in enumerate(
        zip(
            parameters.state_means,
            parameters.initial_probabilities,
            parameters.transition_probabilities,
            strict=True,
        )
    ):
        states = joint_states[:, appliance]
        joint_means = joint_means + means[states]
        joint_initial = joint_initial + torch.log(initial[states])
        joint_transition = joint_transition + torch.log(
            transition[states[:, None], states[None, :]]
        )
    return joint_states, joint_means, joint_initial, joint_transition


def _validate_observations(
    observations,
    parameters: FactorialHMMParameters,
    noise_std: float,
) -> torch.Tensor:
    values = _floating_tensor("observations", observations)
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("observations must be a nonempty one-dimensional tensor.")
    reference = parameters.state_means[0]
    if values.dtype != reference.dtype or values.device != reference.device:
        raise ValueError(
            f"observations must use dtype {reference.dtype} and device "
            f"{reference.device}."
        )
    if isinstance(noise_std, bool) or not isinstance(noise_std, (int, float)):
        raise TypeError("noise_std must be a positive finite number.")
    if not math.isfinite(noise_std) or noise_std <= 0:
        raise ValueError("noise_std must be a positive finite number.")
    return values


def _emission_scores(
    observations: torch.Tensor,
    joint_means: torch.Tensor,
    noise_std: float,
) -> torch.Tensor:
    scale = observations.new_tensor(noise_std)
    residual = (observations[:, None] - joint_means[None, :]) / scale
    normalizer = torch.log(scale) + 0.5 * math.log(2.0 * math.pi)
    return -0.5 * residual.square() - normalizer


@torch.no_grad()
def factorial_hmm_viterbi(
    observations: torch.Tensor,
    state_means: Sequence[torch.Tensor],
    initial_probabilities: Sequence[torch.Tensor],
    transition_probabilities: Sequence[torch.Tensor],
    *,
    noise_std: float,
    max_joint_states: int = 512,
) -> FactorialHMMViterbiResult:
    """Decode the exact maximum-probability joint appliance state path."""

    parameters = canonicalize_factorial_hmm(
        state_means, initial_probabilities, transition_probabilities
    )
    values = _validate_observations(observations, parameters, noise_std)
    joint_states, joint_means, initial, transition = _joint_model(
        parameters, max_joint_states=max_joint_states
    )
    emissions = _emission_scores(values, joint_means, noise_std)

    time_points, joint_count = emissions.shape
    scores = emissions.new_empty((time_points, joint_count))
    backpointers = torch.zeros(
        (time_points, joint_count), dtype=torch.int64, device=values.device
    )
    scores[0] = initial + emissions[0]
    for time in range(1, time_points):
        candidates = scores[time - 1][:, None] + transition
        best_scores, best_sources = torch.max(candidates, dim=0)
        scores[time] = best_scores + emissions[time]
        backpointers[time] = best_sources

    final_joint_state = int(torch.argmax(scores[-1]))
    final_score = scores[-1, final_joint_state]
    if not bool(torch.isfinite(final_score)):
        raise ValueError("No valid factorial-HMM path reaches the sequence end.")
    decoded_joint = torch.empty(time_points, dtype=torch.int64, device=values.device)
    decoded_joint[-1] = final_joint_state
    for time in range(time_points - 1, 0, -1):
        decoded_joint[time - 1] = backpointers[time, decoded_joint[time]]

    state_indices = joint_states[decoded_joint]
    appliance_power = torch.stack(
        [
            means[state_indices[:, appliance]]
            for appliance, means in enumerate(parameters.state_means)
        ],
        dim=1,
    )
    aggregate_power = appliance_power.sum(dim=1)
    return FactorialHMMViterbiResult(
        state_indices=state_indices,
        appliance_power=appliance_power,
        aggregate_power=aggregate_power,
        score=float(final_score),
    )


def factorial_hmm_path_score(
    state_indices: torch.Tensor,
    observations: torch.Tensor,
    state_means: Sequence[torch.Tensor],
    initial_probabilities: Sequence[torch.Tensor],
    transition_probabilities: Sequence[torch.Tensor],
    *,
    noise_std: float,
) -> torch.Tensor:
    """Return the log score of one labeled per-appliance state path."""

    parameters = canonicalize_factorial_hmm(
        state_means, initial_probabilities, transition_probabilities
    )
    values = _validate_observations(observations, parameters, noise_std)
    if not isinstance(state_indices, torch.Tensor):
        raise TypeError("state_indices must be a torch.Tensor.")
    if state_indices.dtype not in _INTEGER_DTYPES or state_indices.ndim != 2:
        raise TypeError("state_indices must be a two-dimensional integer tensor.")
    expected_shape = (values.numel(), len(parameters.state_means))
    if tuple(state_indices.shape) != expected_shape:
        raise ValueError(f"state_indices must have shape {expected_shape}.")
    if state_indices.device != values.device:
        raise ValueError(f"state_indices must be on {values.device}.")
    for appliance, means in enumerate(parameters.state_means):
        states = state_indices[:, appliance]
        if bool(((states < 0) | (states >= means.numel())).any()):
            raise ValueError(
                f"state_indices for appliance {appliance} must be between 0 and "
                f"{means.numel() - 1}."
            )

    appliance_power = torch.stack(
        [
            means[state_indices[:, appliance]]
            for appliance, means in enumerate(parameters.state_means)
        ],
        dim=1,
    )
    scale = values.new_tensor(noise_std)
    residual = (values - appliance_power.sum(dim=1)) / scale
    score = (
        -0.5 * residual.square() - torch.log(scale) - 0.5 * math.log(2.0 * math.pi)
    ).sum()
    for appliance, (initial, transition) in enumerate(
        zip(
            parameters.initial_probabilities,
            parameters.transition_probabilities,
            strict=True,
        )
    ):
        states = state_indices[:, appliance]
        score = score + torch.log(initial[states[0]])
        if values.numel() > 1:
            score = score + torch.log(transition[states[:-1], states[1:]]).sum()
    if not bool(torch.isfinite(score)):
        raise ValueError("state_indices describe a path forbidden by the model.")
    return score


__all__ = [
    "FactorialHMMParameters",
    "FactorialHMMViterbiResult",
    "canonicalize_factorial_hmm",
    "factorial_hmm_path_score",
    "factorial_hmm_viterbi",
]
