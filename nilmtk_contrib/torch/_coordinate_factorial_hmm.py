"""Coordinate-ascent inference for additive factorial hidden Markov models.

The exact factorial decoder constructs the Cartesian product of appliance
states.  This approximation instead holds all but one appliance path fixed and
uses the shared exact HMM decoder to optimize the remaining path.  Every
accepted coordinate update increases the joint log score, and no external
solver is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from nilmtk_contrib.torch._factorial_hmm import canonicalize_factorial_hmm
from nilmtk_contrib.torch._hmm import (
    gaussian_log_emissions,
    hmm_path_score,
    hmm_viterbi,
)


@dataclass(frozen=True)
class CoordinateFactorialHMMViterbiResult:
    """Locally optimal appliance paths and coordinate-ascent diagnostics."""

    state_indices: torch.Tensor
    appliance_power: torch.Tensor
    aggregate_power: torch.Tensor
    score: float
    score_history: tuple[float, ...]
    iterations: int
    converged: bool


def _validate_max_iterations(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_iterations must be an integer.")
    if value < 1:
        raise ValueError("max_iterations must be positive.")
    return value


def _joint_path_score(
    observations,
    state_indices,
    appliance_power,
    initial_probabilities,
    transition_probabilities,
    *,
    noise_std,
):
    scale = observations.new_tensor(noise_std)
    aggregate = appliance_power.sum(dim=1)
    residual = (observations - aggregate) / scale
    score = (
        -0.5 * residual.square()
        - torch.log(scale)
        - 0.5 * math.log(2.0 * math.pi)
    ).sum()
    for appliance, (initial, transition) in enumerate(
        zip(initial_probabilities, transition_probabilities, strict=True)
    ):
        states = state_indices[:, appliance]
        score = score + torch.log(initial[states[0]])
        if states.numel() > 1:
            score = score + torch.log(
                transition[states[:-1], states[1:]]
            ).sum()
    return float(score)


@torch.no_grad()
def factorial_hmm_coordinate_viterbi(
    observations: torch.Tensor,
    state_means: Sequence[torch.Tensor],
    initial_probabilities: Sequence[torch.Tensor],
    transition_probabilities: Sequence[torch.Tensor],
    *,
    noise_std: float,
    max_iterations: int = 20,
) -> CoordinateFactorialHMMViterbiResult:
    """Approximately decode an additive FHMM by deterministic coordinate ascent.

    A valid prior-only Viterbi path initializes every appliance.  Each sweep
    then decodes one appliance against the aggregate residual left by the
    others.  Tied coordinate scores retain the current path, preventing cycles
    while preserving deterministic output.
    """

    max_iterations = _validate_max_iterations(max_iterations)
    parameters = canonicalize_factorial_hmm(
        state_means, initial_probabilities, transition_probabilities
    )
    # Reuse the shared Gaussian emission validator before allocating paths.
    gaussian_log_emissions(
        observations,
        parameters.state_means[0],
        noise_std=noise_std,
    )
    time_points = int(observations.numel())
    appliance_count = len(parameters.state_means)
    state_indices = torch.empty(
        (time_points, appliance_count),
        dtype=torch.int64,
        device=observations.device,
    )

    log_initial = tuple(torch.log(value) for value in parameters.initial_probabilities)
    log_transition = tuple(
        torch.log(value) for value in parameters.transition_probabilities
    )
    for appliance, means in enumerate(parameters.state_means):
        prior_emissions = means.new_zeros((time_points, means.numel()))
        state_indices[:, appliance] = hmm_viterbi(
            prior_emissions,
            log_initial[appliance],
            log_transition[appliance],
        ).states

    appliance_power = torch.stack(
        [
            means[state_indices[:, appliance]]
            for appliance, means in enumerate(parameters.state_means)
        ],
        dim=1,
    )
    aggregate_power = appliance_power.sum(dim=1)
    score_history = [
        _joint_path_score(
            observations,
            state_indices,
            appliance_power,
            parameters.initial_probabilities,
            parameters.transition_probabilities,
            noise_std=noise_std,
        )
    ]
    relative_tolerance = 1e-5 if observations.dtype == torch.float32 else 1e-12
    converged = False

    for iteration in range(1, max_iterations + 1):
        changed = False
        for appliance, means in enumerate(parameters.state_means):
            residual = observations - (
                aggregate_power - appliance_power[:, appliance]
            )
            emissions = gaussian_log_emissions(residual, means, noise_std=noise_std)
            current_score = float(
                hmm_path_score(
                    state_indices[:, appliance],
                    emissions,
                    log_initial[appliance],
                    log_transition[appliance],
                )
            )
            decoded = hmm_viterbi(
                emissions,
                log_initial[appliance],
                log_transition[appliance],
            )
            tolerance = relative_tolerance * max(
                1.0, abs(current_score), abs(decoded.score)
            )
            if decoded.score <= current_score + tolerance:
                continue

            previous_power = appliance_power[:, appliance].clone()
            state_indices[:, appliance] = decoded.states
            appliance_power[:, appliance] = means[decoded.states]
            aggregate_power = (
                aggregate_power
                - previous_power
                + appliance_power[:, appliance]
            )
            changed = True

        score_history.append(
            _joint_path_score(
                observations,
                state_indices,
                appliance_power,
                parameters.initial_probabilities,
                parameters.transition_probabilities,
                noise_std=noise_std,
            )
        )
        if not changed:
            converged = True
            break

    return CoordinateFactorialHMMViterbiResult(
        state_indices=state_indices,
        appliance_power=appliance_power,
        aggregate_power=aggregate_power,
        score=score_history[-1],
        score_history=tuple(score_history),
        iterations=iteration,
        converged=converged,
    )


__all__ = [
    "CoordinateFactorialHMMViterbiResult",
    "factorial_hmm_coordinate_viterbi",
]
