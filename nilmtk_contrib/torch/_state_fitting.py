"""Shared deterministic finite-state fitting for structured NILM models."""

from collections.abc import Sequence

import torch


def fit_state_means(
    windows: Sequence[torch.Tensor],
    *,
    num_states: int,
    max_iterations: int,
) -> torch.Tensor:
    """Fit ordered one-dimensional centers with deterministic Lloyd steps."""
    values = torch.sort(torch.cat(tuple(windows))).values
    unique = torch.unique(values, sorted=True)
    if unique.numel() < num_states:
        raise ValueError(
            f"Training power contains {unique.numel()} unique values; "
            f"at least num_states={num_states} are required."
        )
    initial_indices = (
        torch.linspace(
            0,
            unique.numel() - 1,
            steps=num_states,
            dtype=torch.float64,
        )
        .round()
        .to(torch.long)
    )
    centers = unique[initial_indices]
    previous_assignments = None
    for _ in range(max_iterations):
        assignments = assign_states(values, centers)
        if previous_assignments is not None and torch.equal(
            assignments, previous_assignments
        ):
            break
        counts = torch.bincount(assignments, minlength=num_states)
        if bool((counts == 0).any()):
            raise RuntimeError("Deterministic state fitting produced an empty cluster.")
        centers = torch.stack(
            [values[assignments == state].mean() for state in range(num_states)]
        )
        if not bool(torch.isfinite(centers).all()):
            raise RuntimeError(
                "State fitting produced non-finite means; rescale the power data."
            )
        previous_assignments = assignments
    else:
        raise RuntimeError("Deterministic state fitting did not converge.")
    return torch.sort(centers).values


def assign_states(values: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    """Assign each scalar to its nearest mean, breaking ties by state order."""
    return torch.argmin(torch.abs(values[:, None] - means[None, :]), dim=1)


__all__ = ["assign_states", "fit_state_means"]
