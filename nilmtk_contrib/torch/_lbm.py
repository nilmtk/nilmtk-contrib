"""Solver-free structured inference primitives for latent Bayesian melding.

The fixed-variance relaxation in Zhong, Goddard, and Sutton (2015) is a
convex quadratic objective over an HMM path polytope.  A linear minimization
oracle over that polytope is Viterbi.  Cycle-aware Viterbi plus a closed-form
equal-moment coupling handles the paper's relaxed latent cycle constraint.
Conditional gradient therefore keeps every iterate feasible and supplies an
optimality gap without depending on a generic conic solver.

This module is deliberately private.  It is the numerical kernel for a future
NILMTK disaggregator, not yet a benchmarkable public model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Sequence

import torch


@dataclass(frozen=True)
class GaussianRatioSummary:
    """One Gaussian population/induced-prior ratio from the LBM objective.

    ``state_weights`` defines a linear summary of relaxed states.  A vector of
    length K is repeated over time; a T-by-K matrix can describe a general
    time-dependent summary.  ``conditional_means[c]`` is the population mean
    conditional on ``c`` off-to-on cycles.

    The population variance must be smaller than the induced variance.  This
    is the condition that leaves a positive quadratic precision after dividing
    the population Gaussian by the induced Gaussian.
    """

    state_weights: torch.Tensor | Sequence[float]
    conditional_means: torch.Tensor | Sequence[float]
    conditional_variance: float
    induced_mean: float
    induced_variance: float
    weight: float = 1.0


@dataclass(frozen=True)
class StructuredAppliance:
    """Fixed HMM and optional population information for one appliance.

    When supplied, ``cycle_probabilities`` must represent consecutive counts
    starting at zero.  For example, three entries mean zero, one, or two
    off-to-on cycles.
    """

    state_means: torch.Tensor | Sequence[float]
    initial_probabilities: torch.Tensor | Sequence[float]
    transition_probabilities: torch.Tensor | Sequence[Sequence[float]]
    off_state: int = 0
    cycle_probabilities: torch.Tensor | Sequence[float] | None = None
    summaries: tuple[GaussianRatioSummary, ...] = ()


@dataclass(frozen=True)
class StructuredRelaxationResult:
    """Certified result of the fixed-variance structured relaxation."""

    states: tuple[torch.Tensor, ...]
    transitions: tuple[torch.Tensor, ...]
    cycle_posteriors: tuple[torch.Tensor | None, ...]
    prediction: torch.Tensor
    objective: float
    duality_gap: float
    iterations: int
    converged: bool
    objective_history: tuple[float, ...]


@dataclass(frozen=True)
class _PreparedSummary:
    state_weights: torch.Tensor
    cycle_centers: torch.Tensor
    cycle_offsets: torch.Tensor
    precision: float
    weight: float


@dataclass(frozen=True)
class _PreparedAppliance:
    state_means: torch.Tensor
    initial_costs: torch.Tensor
    transition_costs: torch.Tensor
    off_state: int
    cycle_costs: torch.Tensor | None
    summaries: tuple[_PreparedSummary, ...]


def _finite_real(name: str, value: Real, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_real(name: str, value: Real) -> float:
    result = _finite_real(name, value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _positive_int(name: str, value: Integral) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _resolve_device(requested) -> torch.device:
    try:
        device = torch.device(requested)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Invalid torch device {requested!r}.") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Structured LBM supports only CPU and CUDA devices.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU.")
        device_count = torch.cuda.device_count()
        if device_count < 1:
            raise RuntimeError(
                "CUDA was reported available but has no visible devices."
            )
        if device.index is not None and device.index >= device_count:
            raise RuntimeError(
                f"CUDA device index {device.index} is unavailable; "
                f"visible device count is {device_count}."
            )
    return device


def _as_finite_tensor(
    name: str,
    value,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} must contain real numeric values.") from exc
    if tensor.is_complex():
        raise TypeError(f"{name} must contain real numeric values.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _probabilities(
    name: str,
    value,
    *,
    expected_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    sum_dimension: int,
) -> torch.Tensor:
    tensor = _as_finite_tensor(name, value, device=device, dtype=dtype)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}.")
    if bool((tensor <= 0).any()):
        raise ValueError(
            f"{name} must be strictly positive; apply an explicit pseudocount "
            "while fitting the HMM."
        )
    totals = tensor.sum(dim=sum_dimension)
    if not torch.allclose(
        totals,
        torch.ones_like(totals),
        rtol=1e-7,
        atol=1e-9,
    ):
        raise ValueError(f"{name} must sum to one along dimension {sum_dimension}.")
    return tensor


def _prepare_summary(
    summary: GaussianRatioSummary,
    *,
    time_points: int,
    states: int,
    cycles: int,
    device: torch.device,
    dtype: torch.dtype,
    label: str,
) -> _PreparedSummary:
    weights = _as_finite_tensor(
        f"{label}.state_weights",
        summary.state_weights,
        device=device,
        dtype=dtype,
    )
    if tuple(weights.shape) == (states,):
        weights = weights.expand(time_points, states)
    elif tuple(weights.shape) != (time_points, states):
        raise ValueError(
            f"{label}.state_weights must have shape ({states},) or "
            f"({time_points}, {states})."
        )

    conditional_means = _as_finite_tensor(
        f"{label}.conditional_means",
        summary.conditional_means,
        device=device,
        dtype=dtype,
    )
    if tuple(conditional_means.shape) != (cycles,):
        raise ValueError(f"{label}.conditional_means must have shape ({cycles},).")
    conditional_variance = _finite_real(
        f"{label}.conditional_variance",
        summary.conditional_variance,
        positive=True,
    )
    induced_mean = _finite_real(f"{label}.induced_mean", summary.induced_mean)
    induced_variance = _finite_real(
        f"{label}.induced_variance",
        summary.induced_variance,
        positive=True,
    )
    if conditional_variance >= induced_variance:
        raise ValueError(
            f"{label}.conditional_variance must be smaller than "
            "induced_variance so the Gaussian ratio remains convex."
        )
    weight = _nonnegative_real(f"{label}.weight", summary.weight)

    precision = 1.0 / conditional_variance - 1.0 / induced_variance
    if not math.isfinite(precision):
        raise ValueError(f"{label} variances produce a non-finite precision.")
    centers = (
        conditional_means / conditional_variance - induced_mean / induced_variance
    ) / precision
    offsets = 0.5 * (
        torch.square(conditional_means) / conditional_variance
        - induced_mean**2 / induced_variance
        - precision * torch.square(centers)
    )
    if not bool(torch.isfinite(centers).all()) or not bool(
        torch.isfinite(offsets).all()
    ):
        raise ValueError(f"{label} parameters produce non-finite Gaussian terms.")
    return _PreparedSummary(weights, centers, offsets, precision, weight)


def _prepare_appliances(
    appliances: Sequence[StructuredAppliance],
    *,
    time_points: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[_PreparedAppliance, ...]:
    if not isinstance(appliances, Sequence) or isinstance(appliances, (str, bytes)):
        raise TypeError("appliances must be a non-empty sequence.")
    if not appliances:
        raise ValueError("appliances must be non-empty.")

    prepared = []
    for index, appliance in enumerate(appliances):
        if not isinstance(appliance, StructuredAppliance):
            raise TypeError(
                f"appliances[{index}] must be a StructuredAppliance instance."
            )
        prefix = f"appliances[{index}]"
        means = _as_finite_tensor(
            f"{prefix}.state_means",
            appliance.state_means,
            device=device,
            dtype=dtype,
        )
        if means.ndim != 1 or means.numel() < 2:
            raise ValueError(
                f"{prefix}.state_means must be one-dimensional with K >= 2."
            )
        if bool((means < 0).any()):
            raise ValueError(f"{prefix}.state_means must be non-negative.")
        state_count = int(means.numel())
        initial = _probabilities(
            f"{prefix}.initial_probabilities",
            appliance.initial_probabilities,
            expected_shape=(state_count,),
            device=device,
            dtype=dtype,
            sum_dimension=0,
        )
        transition = _probabilities(
            f"{prefix}.transition_probabilities",
            appliance.transition_probabilities,
            expected_shape=(state_count, state_count),
            device=device,
            dtype=dtype,
            sum_dimension=1,
        )
        if (
            isinstance(appliance.off_state, bool)
            or not isinstance(appliance.off_state, Integral)
            or not 0 <= int(appliance.off_state) < state_count
        ):
            raise ValueError(f"{prefix}.off_state must index one of the K states.")

        cycle_costs = None
        if not isinstance(appliance.summaries, Sequence) or isinstance(
            appliance.summaries,
            (str, bytes),
        ):
            raise TypeError(f"{prefix}.summaries must be a sequence.")
        summary_count = len(appliance.summaries)
        if appliance.cycle_probabilities is None:
            if summary_count:
                raise ValueError(
                    f"{prefix}.cycle_probabilities are required when summaries exist."
                )
            summaries = ()
        else:
            raw_cycles = _as_finite_tensor(
                f"{prefix}.cycle_probabilities",
                appliance.cycle_probabilities,
                device=device,
                dtype=dtype,
            )
            if raw_cycles.ndim != 1 or raw_cycles.numel() < 1:
                raise ValueError(
                    f"{prefix}.cycle_probabilities must be one-dimensional."
                )
            cycles = int(raw_cycles.numel())
            cycle_probabilities = _probabilities(
                f"{prefix}.cycle_probabilities",
                raw_cycles,
                expected_shape=(cycles,),
                device=device,
                dtype=dtype,
                sum_dimension=0,
            )
            maximum_reachable = time_points // 2
            if cycles - 1 > maximum_reachable:
                raise ValueError(
                    f"{prefix}.cycle_probabilities include unreachable cycle counts; "
                    f"at most {maximum_reachable} are possible in {time_points} samples."
                )
            cycle_costs = -torch.log(cycle_probabilities)
            prepared_summaries = []
            for summary_index, summary in enumerate(appliance.summaries):
                if not isinstance(summary, GaussianRatioSummary):
                    raise TypeError(
                        f"{prefix}.summaries[{summary_index}] must be a "
                        "GaussianRatioSummary instance."
                    )
                prepared_summaries.append(
                    _prepare_summary(
                        summary,
                        time_points=time_points,
                        states=state_count,
                        cycles=cycles,
                        device=device,
                        dtype=dtype,
                        label=f"{prefix}.summaries[{summary_index}]",
                    )
                )
            summaries = tuple(prepared_summaries)
        prepared.append(
            _PreparedAppliance(
                means,
                -torch.log(initial),
                -torch.log(transition),
                int(appliance.off_state),
                cycle_costs,
                summaries,
            )
        )
    return tuple(prepared)


def _vertices_from_path(
    path: Sequence[int],
    *,
    states: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    time_points = len(path)
    path_tensor = torch.tensor(path, device=device, dtype=torch.long)
    state_vertex = torch.nn.functional.one_hot(
        path_tensor,
        num_classes=states,
    ).to(dtype=dtype)
    transition_vertex = torch.zeros(
        (max(time_points - 1, 0), states, states),
        device=device,
        dtype=dtype,
    )
    if time_points > 1:
        indices = torch.arange(time_points - 1, device=device)
        transition_vertex[indices, path_tensor[:-1], path_tensor[1:]] = 1.0
    return state_vertex, transition_vertex


def _cycle_viterbi_tables(
    emission_costs: torch.Tensor,
    transition_costs: torch.Tensor,
    *,
    off_state: int,
    cycle_count: int,
):
    """Return best terminal costs and backpointers for every cycle count."""
    time_points, states = emission_costs.shape
    device = emission_costs.device
    dtype = emission_costs.dtype
    infinity = torch.tensor(math.inf, device=device, dtype=dtype)
    costs = torch.full((cycle_count, states), infinity, device=device, dtype=dtype)
    costs[0] = emission_costs[0]
    predecessor_states = []
    predecessor_cycles = []
    state_indices = torch.arange(states, device=device)
    cycle_indices = torch.arange(cycle_count, device=device)
    increments = (
        (state_indices[:, None] == off_state) & (state_indices[None, :] != off_state)
    ).to(dtype=torch.long)
    source_cycles = cycle_indices[:, None, None] - increments[None, :, :]
    valid_sources = source_cycles >= 0
    source_cycles_for_gather = source_cycles.clamp_min(0)
    previous_states_for_gather = state_indices[None, :, None].expand(
        cycle_count,
        states,
        states,
    )
    for time_index in range(1, time_points):
        transition_at_time = (
            transition_costs
            if transition_costs.ndim == 2
            else transition_costs[time_index - 1]
        )
        candidates = (
            costs[source_cycles_for_gather, previous_states_for_gather]
            + transition_at_time[None, :, :]
            + emission_costs[time_index][None, None, :]
        )
        candidates = torch.where(valid_sources, candidates, infinity)
        costs, previous_state_table = torch.min(candidates, dim=1)
        previous_cycle_table = torch.gather(
            source_cycles,
            1,
            previous_state_table[:, None, :],
        ).squeeze(1)
        predecessor_states.append(previous_state_table)
        predecessor_cycles.append(previous_cycle_table)
    return costs, predecessor_states, predecessor_cycles


def _backtrack_cycle_path(
    final_cycle: int,
    final_state: int,
    predecessor_states: Sequence[torch.Tensor],
    predecessor_cycles: Sequence[torch.Tensor],
) -> list[int]:
    path = [final_state]
    cycle_path = final_cycle
    for state_table, cycle_table in zip(
        reversed(predecessor_states),
        reversed(predecessor_cycles),
    ):
        previous_state = int(state_table[cycle_path, final_state].item())
        previous_cycle = int(cycle_table[cycle_path, final_state].item())
        if previous_state < 0 or previous_cycle < 0:
            raise RuntimeError("Cycle-aware Viterbi backtracking failed.")
        final_state = previous_state
        cycle_path = previous_cycle
        path.append(final_state)
    path.reverse()
    return path


def _minimum_moment_coupling(
    path_costs: torch.Tensor,
    category_costs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the two-simplex, equal-mean cycle LP by enumerating its vertices."""
    path_count = int(path_costs.numel())
    category_count = int(category_costs.numel())
    path_values = path_costs.detach().cpu().tolist()
    category_values = category_costs.detach().cpu().tolist()
    best_value = math.inf
    best_support = None

    def consider(value, support):
        nonlocal best_value, best_support
        if value < best_value:
            best_value = value
            best_support = support

    shared_count = min(path_count, category_count)
    for cycle in range(shared_count):
        consider(
            path_values[cycle] + category_values[cycle],
            ("matched", cycle),
        )

    for category in range(category_count):
        for lower_path in range(category):
            for upper_path in range(category + 1, path_count):
                lower_weight = (upper_path - category) / (upper_path - lower_path)
                value = (
                    lower_weight * path_values[lower_path]
                    + (1.0 - lower_weight) * path_values[upper_path]
                    + category_values[category]
                )
                consider(
                    value,
                    (
                        "two_paths",
                        lower_path,
                        upper_path,
                        lower_weight,
                        category,
                    ),
                )

    for path_cycle in range(path_count):
        for lower_category in range(path_cycle):
            for upper_category in range(path_cycle + 1, category_count):
                lower_weight = (upper_category - path_cycle) / (
                    upper_category - lower_category
                )
                value = (
                    path_values[path_cycle]
                    + lower_weight * category_values[lower_category]
                    + (1.0 - lower_weight) * category_values[upper_category]
                )
                consider(
                    value,
                    (
                        "two_categories",
                        path_cycle,
                        lower_category,
                        upper_category,
                        lower_weight,
                    ),
                )

    if best_support is None:
        raise RuntimeError("No feasible expected-cycle coupling exists.")
    path_weights = torch.zeros_like(path_costs)
    category_weights = torch.zeros_like(category_costs)
    if best_support[0] == "matched":
        cycle = best_support[1]
        path_weights[cycle] = 1.0
        category_weights[cycle] = 1.0
    elif best_support[0] == "two_paths":
        _, lower_path, upper_path, lower_weight, category = best_support
        path_weights[lower_path] = lower_weight
        path_weights[upper_path] = 1.0 - lower_weight
        category_weights[category] = 1.0
    else:
        _, path_cycle, lower_category, upper_category, lower_weight = best_support
        path_weights[path_cycle] = 1.0
        category_weights[lower_category] = lower_weight
        category_weights[upper_category] = 1.0 - lower_weight
    return path_weights, category_weights


def _path_vertex(
    emission_costs: torch.Tensor,
    transition_costs: torch.Tensor,
    *,
    off_state: int,
    cycle_costs: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return a minimum-cost path with an integral cycle annotation."""
    time_points, states = emission_costs.shape
    device = emission_costs.device
    dtype = emission_costs.dtype

    if cycle_costs is None:
        costs = emission_costs[0]
        previous_states = []
        for time_index in range(1, time_points):
            transition_at_time = (
                transition_costs
                if transition_costs.ndim == 2
                else transition_costs[time_index - 1]
            )
            candidates = costs[:, None] + transition_at_time
            costs, predecessor = torch.min(candidates, dim=0)
            costs = costs + emission_costs[time_index]
            previous_states.append(predecessor)
        final_state = int(torch.argmin(costs).item())
        path = [final_state]
        for predecessor in reversed(previous_states):
            final_state = int(predecessor[final_state].item())
            path.append(final_state)
        path.reverse()
        category = None
    else:
        costs, predecessor_states, predecessor_cycles = _cycle_viterbi_tables(
            emission_costs,
            transition_costs,
            off_state=off_state,
            cycle_count=int(cycle_costs.numel()),
        )
        terminal = costs + cycle_costs[:, None]
        flat_index = int(torch.argmin(terminal).item())
        final_cycle = flat_index // states
        final_state = flat_index % states
        if not math.isfinite(float(terminal[final_cycle, final_state])):
            raise RuntimeError(
                "No finite HMM path satisfies the supplied cycle support."
            )
        path = _backtrack_cycle_path(
            final_cycle,
            final_state,
            predecessor_states,
            predecessor_cycles,
        )
        category = torch.zeros_like(cycle_costs)
        category[final_cycle] = 1.0

    state_vertex, transition_vertex = _vertices_from_path(
        path,
        states=states,
        device=device,
        dtype=dtype,
    )
    return state_vertex, transition_vertex, category


def _moment_coupled_vertex(
    emission_costs: torch.Tensor,
    transition_costs: torch.Tensor,
    *,
    off_state: int,
    category_costs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the paper's expected-cycle linear oracle without a generic LP."""
    time_points, states = emission_costs.shape
    maximum_cycles = time_points // 2
    terminal_costs, predecessor_states, predecessor_cycles = _cycle_viterbi_tables(
        emission_costs,
        transition_costs,
        off_state=off_state,
        cycle_count=maximum_cycles + 1,
    )
    path_costs, final_states = torch.min(terminal_costs, dim=1)
    if not bool(torch.isfinite(path_costs).all()):
        raise RuntimeError("A reachable cycle count has no finite HMM path.")
    path_weights, category_weights = _minimum_moment_coupling(
        path_costs,
        category_costs,
    )

    state_vertex = torch.zeros_like(emission_costs)
    transition_vertex = torch.zeros(
        (max(time_points - 1, 0), states, states),
        device=emission_costs.device,
        dtype=emission_costs.dtype,
    )
    active_cycles = torch.nonzero(path_weights > 0, as_tuple=False).flatten()
    for cycle_tensor in active_cycles:
        cycle = int(cycle_tensor.item())
        path = _backtrack_cycle_path(
            cycle,
            int(final_states[cycle].item()),
            predecessor_states,
            predecessor_cycles,
        )
        path_states, path_transitions = _vertices_from_path(
            path,
            states=states,
            device=emission_costs.device,
            dtype=emission_costs.dtype,
        )
        weight = path_weights[cycle]
        state_vertex = state_vertex + weight * path_states
        transition_vertex = transition_vertex + weight * path_transitions
    return state_vertex, transition_vertex, category_weights


def _objective(
    aggregate: torch.Tensor,
    appliances: tuple[_PreparedAppliance, ...],
    states: Sequence[torch.Tensor],
    transitions: Sequence[torch.Tensor],
    categories: Sequence[torch.Tensor | None],
    *,
    observation_variance: float,
    melding_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = torch.zeros_like(aggregate)
    objective = torch.zeros((), device=aggregate.device, dtype=aggregate.dtype)
    for appliance, state, transition, category in zip(
        appliances,
        states,
        transitions,
        categories,
    ):
        prediction = prediction + state @ appliance.state_means
        objective = objective + torch.dot(state[0], appliance.initial_costs)
        objective = objective + torch.sum(
            transition * appliance.transition_costs[None, :, :]
        )
        if category is not None:
            objective = objective + melding_weight * torch.dot(
                category,
                appliance.cycle_costs,
            )
            for summary in appliance.summaries:
                statistic = torch.sum(state * summary.state_weights)
                center = torch.dot(category, summary.cycle_centers)
                objective = objective + (
                    melding_weight
                    * summary.weight
                    * torch.dot(category, summary.cycle_offsets)
                )
                objective = objective + (
                    0.5
                    * melding_weight
                    * summary.weight
                    * summary.precision
                    * torch.square(statistic - center)
                )
    residual = aggregate - prediction
    objective = (
        objective + 0.5 * torch.sum(torch.square(residual)) / observation_variance
    )
    return objective, prediction


def _linear_oracle(
    appliances: tuple[_PreparedAppliance, ...],
    state_gradients: Sequence[torch.Tensor],
    transition_gradients: Sequence[torch.Tensor],
    category_gradients: Sequence[torch.Tensor | None],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor | None]]:
    state_vertices = []
    transition_vertices = []
    category_vertices = []
    for appliance, state_gradient, transition_gradient, category_gradient in zip(
        appliances,
        state_gradients,
        transition_gradients,
        category_gradients,
    ):
        if category_gradient is None:
            state_vertex, transition_vertex, category_vertex = _path_vertex(
                state_gradient,
                transition_gradient,
                off_state=appliance.off_state,
                cycle_costs=None,
            )
        else:
            state_vertex, transition_vertex, category_vertex = _moment_coupled_vertex(
                state_gradient,
                transition_gradient,
                off_state=appliance.off_state,
                category_costs=category_gradient,
            )
        state_vertices.append(state_vertex)
        transition_vertices.append(transition_vertex)
        category_vertices.append(category_vertex)
    return state_vertices, transition_vertices, category_vertices


def _value_and_oracle(
    aggregate: torch.Tensor,
    appliances: tuple[_PreparedAppliance, ...],
    states: Sequence[torch.Tensor],
    transitions: Sequence[torch.Tensor],
    categories: Sequence[torch.Tensor | None],
    *,
    observation_variance: float,
    melding_weight: float,
):
    differentiable_states = [state.detach().requires_grad_(True) for state in states]
    differentiable_transitions = [
        transition.detach().requires_grad_(True) for transition in transitions
    ]
    differentiable_categories = [
        None if category is None else category.detach().requires_grad_(True)
        for category in categories
    ]
    differentiable_variables = (
        differentiable_states
        + differentiable_transitions
        + [category for category in differentiable_categories if category is not None]
    )
    objective, prediction = _objective(
        aggregate,
        appliances,
        differentiable_states,
        differentiable_transitions,
        differentiable_categories,
        observation_variance=observation_variance,
        melding_weight=melding_weight,
    )
    if not bool(torch.isfinite(objective)) or not bool(
        torch.isfinite(prediction).all()
    ):
        raise RuntimeError(
            "Structured LBM produced a non-finite objective or prediction; "
            "rescale the power data and variance parameters."
        )
    gradients = torch.autograd.grad(objective, differentiable_variables)
    if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise RuntimeError(
            "Structured LBM produced a non-finite gradient; rescale the power "
            "data and variance parameters."
        )
    appliance_count = len(appliances)
    state_gradients = gradients[:appliance_count]
    transition_gradients = gradients[appliance_count : 2 * appliance_count]
    remaining_gradients = iter(gradients[2 * appliance_count :])
    category_gradients = [
        None if category is None else next(remaining_gradients)
        for category in differentiable_categories
    ]
    vertices = _linear_oracle(
        appliances,
        state_gradients,
        transition_gradients,
        category_gradients,
    )
    state_vertices, transition_vertices, category_vertices = vertices

    gap = torch.zeros_like(objective)
    for gradient, value, vertex in zip(state_gradients, states, state_vertices):
        gap = gap + torch.sum(gradient * (value - vertex))
    for gradient, value, vertex in zip(
        transition_gradients,
        transitions,
        transition_vertices,
    ):
        gap = gap + torch.sum(gradient * (value - vertex))
    for gradient, value, vertex in zip(
        category_gradients,
        categories,
        category_vertices,
    ):
        if gradient is not None:
            gap = gap + torch.sum(gradient * (value - vertex))
    return (
        objective.detach(),
        prediction.detach(),
        gap.detach(),
        state_vertices,
        transition_vertices,
        category_vertices,
    )


def solve_structured_relaxation(
    aggregate,
    appliances: Sequence[StructuredAppliance],
    *,
    observation_variance: float,
    melding_weight: float = 0.5,
    max_iterations: int = 2000,
    absolute_gap_tolerance: float = 1e-6,
    relative_gap_tolerance: float = 5e-5,
    device: str | torch.device | None = None,
) -> StructuredRelaxationResult:
    """Solve the fixed-variance AFHMM+LBM relaxation with PyTorch only.

    The objective is convex under the validated Gaussian-ratio condition.  The
    returned ``duality_gap`` is the Frank--Wolfe certificate: for this convex
    problem it upper-bounds the difference from the relaxed optimum. Inferred
    relaxed cycle weights are returned as ``cycle_posteriors`` to distinguish
    them from the fitted prior probabilities supplied with each appliance.
    """
    observation_variance = _finite_real(
        "observation_variance",
        observation_variance,
        positive=True,
    )
    melding_weight = _nonnegative_real("melding_weight", melding_weight)
    if melding_weight > 1:
        raise ValueError("melding_weight must not exceed one.")
    max_iterations = _positive_int("max_iterations", max_iterations)
    absolute_gap_tolerance = _nonnegative_real(
        "absolute_gap_tolerance",
        absolute_gap_tolerance,
    )
    relative_gap_tolerance = _nonnegative_real(
        "relative_gap_tolerance",
        relative_gap_tolerance,
    )

    if device is None and isinstance(aggregate, torch.Tensor):
        requested_device = aggregate.device
    else:
        requested_device = "cpu" if device is None else device
    resolved_device = _resolve_device(requested_device)
    aggregate_tensor = _as_finite_tensor(
        "aggregate",
        aggregate,
        device=resolved_device,
        dtype=torch.float64,
    )
    if aggregate_tensor.ndim != 1 or aggregate_tensor.numel() < 1:
        raise ValueError("aggregate must be a non-empty one-dimensional sequence.")
    if bool((aggregate_tensor < 0).any()):
        raise ValueError("aggregate must be non-negative.")

    time_points = int(aggregate_tensor.numel())
    prepared = _prepare_appliances(
        appliances,
        time_points=time_points,
        device=resolved_device,
        dtype=aggregate_tensor.dtype,
    )

    states = []
    transitions = []
    categories = []
    for appliance in prepared:
        initial_emissions = torch.zeros(
            (time_points, appliance.state_means.numel()),
            device=resolved_device,
            dtype=aggregate_tensor.dtype,
        )
        initial_emissions[0] = appliance.initial_costs
        category_costs = (
            None
            if appliance.cycle_costs is None or melding_weight == 0
            else melding_weight * appliance.cycle_costs
        )
        state, transition, category = _path_vertex(
            initial_emissions,
            appliance.transition_costs,
            off_state=appliance.off_state,
            cycle_costs=category_costs,
        )
        states.append(state)
        transitions.append(transition)
        categories.append(category)

    history = []
    iterations = 0
    while True:
        (
            objective,
            prediction,
            raw_gap,
            state_vertices,
            transition_vertices,
            category_vertices,
        ) = _value_and_oracle(
            aggregate_tensor,
            prepared,
            states,
            transitions,
            categories,
            observation_variance=observation_variance,
            melding_weight=melding_weight,
        )
        objective_value = float(objective)
        gap_value = float(raw_gap)
        numerical_scale = max(1.0, abs(objective_value))
        if gap_value < -1e-9 * numerical_scale:
            raise RuntimeError(
                "The structured linear oracle returned an invalid negative gap."
            )
        gap_value = max(0.0, gap_value)
        history.append(objective_value)
        tolerance = absolute_gap_tolerance + relative_gap_tolerance * numerical_scale
        converged = gap_value <= tolerance
        if converged or iterations >= max_iterations:
            return StructuredRelaxationResult(
                tuple(state.detach() for state in states),
                tuple(transition.detach() for transition in transitions),
                tuple(
                    None if category is None else category.detach()
                    for category in categories
                ),
                prediction,
                objective_value,
                gap_value,
                iterations,
                converged,
                tuple(history),
            )

        vertex_objective, _ = _objective(
            aggregate_tensor,
            prepared,
            state_vertices,
            transition_vertices,
            category_vertices,
            observation_variance=observation_variance,
            melding_weight=melding_weight,
        )
        vertex_objective_value = float(vertex_objective)
        if not math.isfinite(vertex_objective_value):
            raise RuntimeError(
                "Structured LBM produced a non-finite line-search objective; "
                "rescale the power data and variance parameters."
            )
        curvature = 2.0 * (vertex_objective_value - objective_value + gap_value)
        if not math.isfinite(curvature):
            raise RuntimeError(
                "Structured LBM produced non-finite line-search curvature; "
                "rescale the power data and variance parameters."
            )
        curvature_tolerance = 1e-10 * max(
            1.0,
            abs(vertex_objective_value),
            abs(objective_value),
            gap_value,
        )
        if curvature < -curvature_tolerance:
            raise RuntimeError(
                "The validated LBM objective was unexpectedly non-convex."
            )
        if curvature <= curvature_tolerance:
            step_size = 1.0
        else:
            step_size = min(1.0, gap_value / curvature)

        states = [
            state + step_size * (vertex - state)
            for state, vertex in zip(states, state_vertices)
        ]
        transitions = [
            transition + step_size * (vertex - transition)
            for transition, vertex in zip(transitions, transition_vertices)
        ]
        categories = [
            None if category is None else category + step_size * (vertex - category)
            for category, vertex in zip(categories, category_vertices)
        ]
        iterations += 1


__all__ = [
    "GaussianRatioSummary",
    "StructuredAppliance",
    "StructuredRelaxationResult",
    "solve_structured_relaxation",
]
