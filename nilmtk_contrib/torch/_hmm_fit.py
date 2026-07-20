"""Deterministic HMM parameter fitting from observed appliance power traces."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from nilmtk_contrib.torch._hmm import (
    GaussianHMMParameters,
    canonicalize_gaussian_hmm,
)


_FLOAT_DTYPES = frozenset({torch.float32, torch.float64})


@dataclass(frozen=True)
class ObservedGaussianHMMFit:
    """Fitted parameters, canonical state paths, and convergence diagnostics."""

    parameters: GaussianHMMParameters
    state_sequences: tuple[torch.Tensor, ...]
    loss_history: tuple[float, ...]
    iterations: int
    converged: bool


def _positive_integer(name, value, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        qualifier = "at least two" if minimum == 2 else "positive"
        raise ValueError(f"{name} must be {qualifier}.")
    return value


def _finite_number(name, value, *, positive):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        qualifier = "positive" if positive else "nonnegative"
        raise TypeError(f"{name} must be a finite {qualifier} number.")
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {qualifier} number.")
    return value


def _validate_sequences(sequences):
    if not isinstance(sequences, (tuple, list)) or not sequences:
        raise ValueError("sequences must contain at least one power trace.")
    validated = []
    reference = None
    for index, values in enumerate(sequences):
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"sequences[{index}] must be a torch.Tensor.")
        if values.dtype not in _FLOAT_DTYPES or values.is_complex():
            raise TypeError(
                f"sequences[{index}] must use torch.float32 or torch.float64."
            )
        if values.ndim != 1 or values.numel() < 1:
            raise ValueError(
                f"sequences[{index}] must be a nonempty one-dimensional tensor."
            )
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"sequences[{index}] must contain only finite values.")
        if bool((values < 0).any()):
            raise ValueError(f"sequences[{index}] must contain nonnegative power.")
        if reference is None:
            reference = values
        elif values.dtype != reference.dtype or values.device != reference.device:
            raise ValueError(
                f"sequences[{index}] must use dtype {reference.dtype} and device "
                f"{reference.device}."
            )
        validated.append(values)
    return tuple(validated)


def _assign_states(values, means):
    distances = (values[:, None] - means[None, :]).square()
    states = torch.argmin(distances, dim=1)
    loss = (values - means[states]).square().sum()
    return states, float(loss)


@torch.no_grad()
def fit_observed_gaussian_hmm(
    sequences: Sequence[torch.Tensor],
    *,
    n_states: int = 2,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    pseudocount: float = 1.0,
) -> ObservedGaussianHMMFit:
    """Fit a discrete-state Gaussian HMM to observed appliance power.

    State means are learned with deterministic one-dimensional Lloyd updates.
    Initial and transition probabilities are then counted from the inferred
    paths, preserving every supplied sequence boundary.  Positive additive
    smoothing keeps all probability vectors and transition rows well-defined.
    """

    sequences = _validate_sequences(sequences)
    n_states = _positive_integer("n_states", n_states, minimum=2)
    max_iterations = _positive_integer("max_iterations", max_iterations)
    tolerance = _finite_number("tolerance", tolerance, positive=False)
    pseudocount = _finite_number("pseudocount", pseudocount, positive=True)
    values = torch.cat(sequences)
    unique_values = torch.unique(values, sorted=True)
    if unique_values.numel() < n_states:
        raise ValueError(
            f"n_states={n_states} requires at least {n_states} distinct power "
            f"values; received {unique_values.numel()}."
        )

    positions = torch.div(
        torch.arange(n_states, device=values.device)
        * (unique_values.numel() - 1),
        n_states - 1,
        rounding_mode="floor",
    )
    means = unique_values[positions]
    states, loss = _assign_states(values, means)
    loss_history = [loss]
    converged = False

    for iteration in range(1, max_iterations + 1):
        updated_means = means.clone()
        for state in range(n_states):
            members = values[states == state]
            if members.numel():
                updated_means[state] = members.mean()
        updated_states, updated_loss = _assign_states(values, updated_means)
        loss_history.append(updated_loss)
        improvement = loss - updated_loss
        threshold = tolerance * max(1.0, abs(loss))
        unchanged = torch.equal(updated_states, states)
        means, states, loss = updated_means, updated_states, updated_loss
        if unchanged or improvement <= threshold:
            converged = True
            break

    order = torch.argsort(means, stable=True)
    inverse_order = torch.empty_like(order)
    inverse_order[order] = torch.arange(n_states, device=values.device)
    means = means[order]
    states = inverse_order[states]
    lengths = [sequence.numel() for sequence in sequences]
    state_sequences = tuple(torch.split(states, lengths))

    initial_counts = means.new_full((n_states,), pseudocount)
    transition_counts = means.new_full((n_states, n_states), pseudocount)
    for sequence_states in state_sequences:
        initial_counts[sequence_states[0]] += 1
        if sequence_states.numel() > 1:
            transitions = sequence_states[:-1] * n_states + sequence_states[1:]
            transition_counts += torch.bincount(
                transitions, minlength=n_states * n_states
            ).reshape(n_states, n_states).to(means.dtype)

    initial_probabilities = initial_counts / initial_counts.sum()
    transition_probabilities = transition_counts / transition_counts.sum(
        dim=1, keepdim=True
    )
    parameters = canonicalize_gaussian_hmm(
        means, initial_probabilities, transition_probabilities
    )
    return ObservedGaussianHMMFit(
        parameters=parameters,
        state_sequences=state_sequences,
        loss_history=tuple(loss_history),
        iterations=iteration,
        converged=converged,
    )


__all__ = ["ObservedGaussianHMMFit", "fit_observed_gaussian_hmm"]
