"""Exact factorial-HMM construction over the shared exact HMM decoder.

This private module is an oracle for a forthcoming PyTorch AFHMM port. It
constructs the Cartesian product of appliance states and delegates all path
inference to :mod:`nilmtk_contrib.torch._hmm`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from nilmtk_contrib.torch._hmm import (
    canonicalize_gaussian_hmm,
    gaussian_log_emissions,
    hmm_path_score,
    hmm_viterbi,
)


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


def canonicalize_factorial_hmm(
    state_means: Sequence[torch.Tensor],
    initial_probabilities: Sequence[torch.Tensor],
    transition_probabilities: Sequence[torch.Tensor],
) -> FactorialHMMParameters:
    """Validate appliance HMMs and sort their states by increasing power."""

    if not isinstance(state_means, (tuple, list)) or not state_means:
        raise ValueError("state_means must contain at least one appliance.")
    appliance_count = len(state_means)
    if len(initial_probabilities) != appliance_count:
        raise ValueError("initial_probabilities must have one entry per appliance.")
    if len(transition_probabilities) != appliance_count:
        raise ValueError("transition_probabilities must have one entry per appliance.")

    canonical = []
    reference = None
    for appliance, values in enumerate(
        zip(state_means, initial_probabilities, transition_probabilities, strict=True)
    ):
        try:
            parameters = canonicalize_gaussian_hmm(*values)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"appliance {appliance}: {exc}") from exc
        if bool((parameters.state_means < 0).any()):
            raise ValueError(f"state_means[{appliance}] must be nonnegative.")
        if reference is None:
            reference = parameters.state_means
        elif (
            parameters.state_means.dtype != reference.dtype
            or parameters.state_means.device != reference.device
        ):
            raise ValueError(
                f"state_means[{appliance}] must use dtype {reference.dtype} and "
                f"device {reference.device}."
            )
        canonical.append(parameters)
    return FactorialHMMParameters(
        state_means=tuple(item.state_means for item in canonical),
        initial_probabilities=tuple(item.initial_probabilities for item in canonical),
        transition_probabilities=tuple(
            item.transition_probabilities for item in canonical
        ),
    )


def _joint_model(parameters, *, max_joint_states):
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
    initial_scores = reference.new_zeros(joint_count)
    transition_scores = reference.new_zeros((joint_count, joint_count))
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
        initial_scores = initial_scores + torch.log(initial[states])
        transition_scores = transition_scores + torch.log(
            transition[states[:, None], states[None, :]]
        )
    return joint_states, joint_means, initial_scores, transition_scores


def _validate_state_indices(state_indices, observations, parameters):
    if not isinstance(state_indices, torch.Tensor):
        raise TypeError("state_indices must be a torch.Tensor.")
    if state_indices.dtype not in _INTEGER_DTYPES or state_indices.ndim != 2:
        raise TypeError("state_indices must be a two-dimensional integer tensor.")
    expected_shape = (observations.numel(), len(parameters.state_means))
    if tuple(state_indices.shape) != expected_shape:
        raise ValueError(f"state_indices must have shape {expected_shape}.")
    if state_indices.device != observations.device:
        raise ValueError(f"state_indices must be on {observations.device}.")
    for appliance, means in enumerate(parameters.state_means):
        states = state_indices[:, appliance]
        if bool(((states < 0) | (states >= means.numel())).any()):
            raise ValueError(
                f"state_indices for appliance {appliance} must be between 0 and "
                f"{means.numel() - 1}."
            )


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
    """Construct the joint HMM and decode its exact maximum-probability path."""

    parameters = canonicalize_factorial_hmm(
        state_means, initial_probabilities, transition_probabilities
    )
    joint_states, joint_means, initial, transition = _joint_model(
        parameters, max_joint_states=max_joint_states
    )
    emissions = gaussian_log_emissions(observations, joint_means, noise_std=noise_std)
    decoded = hmm_viterbi(emissions, initial, transition)
    state_indices = joint_states[decoded.states]
    appliance_power = torch.stack(
        [
            means[state_indices[:, appliance]]
            for appliance, means in enumerate(parameters.state_means)
        ],
        dim=1,
    )
    return FactorialHMMViterbiResult(
        state_indices=state_indices,
        appliance_power=appliance_power,
        aggregate_power=appliance_power.sum(dim=1),
        score=decoded.score,
    )


def factorial_hmm_path_score(
    state_indices: torch.Tensor,
    observations: torch.Tensor,
    state_means: Sequence[torch.Tensor],
    initial_probabilities: Sequence[torch.Tensor],
    transition_probabilities: Sequence[torch.Tensor],
    *,
    noise_std: float,
    max_joint_states: int = 512,
) -> torch.Tensor:
    """Score a labeled appliance path through the constructed joint HMM."""

    parameters = canonicalize_factorial_hmm(
        state_means, initial_probabilities, transition_probabilities
    )
    _joint_states, joint_means, initial, transition = _joint_model(
        parameters, max_joint_states=max_joint_states
    )
    emissions = gaussian_log_emissions(observations, joint_means, noise_std=noise_std)
    _validate_state_indices(state_indices, observations, parameters)
    strides = [
        math.prod(means.numel() for means in parameters.state_means[index + 1 :])
        for index in range(len(parameters.state_means))
    ]
    joint_path = sum(
        state_indices[:, appliance] * stride for appliance, stride in enumerate(strides)
    )
    try:
        return hmm_path_score(joint_path, emissions, initial, transition)
    except ValueError as exc:
        raise ValueError(
            "state_indices describe a path forbidden by the model."
        ) from exc


__all__ = [
    "FactorialHMMParameters",
    "FactorialHMMViterbiResult",
    "canonicalize_factorial_hmm",
    "factorial_hmm_path_score",
    "factorial_hmm_viterbi",
]
