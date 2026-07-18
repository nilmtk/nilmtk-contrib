"""Leakage-safe, deterministic training primitives for structured LBM.

The public NILMTK wrapper will adapt timestamped chunks to these private data
contracts. Keeping the fitter independent of pandas and NILMTK makes the
scientific inputs easy to test and keeps the PyTorch runtime self-contained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import math
import re
from typing import Sequence

import torch

from nilmtk_contrib.torch._lbm import (
    GaussianRatioSummary,
    StructuredAppliance,
    _finite_real,
    _positive_int,
)


_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


def _nonempty_string(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _timestamp(name: str, value: str) -> datetime:
    value = _nonempty_string(name, value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO 8601 timestamp.") from exc
    if parsed.utcoffset() is None:
        raise ValueError(f"{name} must include an explicit UTC offset.")
    return parsed


@dataclass(frozen=True)
class SourceWindow:
    """Provenance for one dense, half-open dataset window ``[start, end)``."""

    dataset: str
    dataset_version: str
    data_uri: str
    data_fingerprint: str
    building: str | int
    start: str
    end: str
    sample_period_seconds: int

    def __post_init__(self):
        for field in ("dataset", "dataset_version", "data_uri"):
            object.__setattr__(
                self,
                field,
                _nonempty_string(field, getattr(self, field)),
            )
        if isinstance(self.building, bool) or not isinstance(self.building, (str, int)):
            raise ValueError("building must be a non-empty string or integer.")
        if isinstance(self.building, str):
            object.__setattr__(
                self, "building", _nonempty_string("building", self.building)
            )
        if not isinstance(self.data_fingerprint, str) or not _SHA256.fullmatch(
            self.data_fingerprint
        ):
            raise ValueError(
                "data_fingerprint must be lowercase sha256:<64 hexadecimal digits>."
            )
        object.__setattr__(self, "start", _nonempty_string("start", self.start))
        object.__setattr__(self, "end", _nonempty_string("end", self.end))
        start = _timestamp("start", self.start)
        end = _timestamp("end", self.end)
        if end.timestamp() <= start.timestamp():
            raise ValueError("end must be later than start.")
        object.__setattr__(
            self,
            "sample_period_seconds",
            _positive_int("sample_period_seconds", self.sample_period_seconds),
        )

    @property
    def interval(self) -> tuple[float, float]:
        return (
            _timestamp("start", self.start).timestamp(),
            _timestamp("end", self.end).timestamp(),
        )


@dataclass(frozen=True)
class ApplianceTrainingWindow:
    """One dense appliance-power window and its complete provenance."""

    source: SourceWindow
    power: torch.Tensor | Sequence[float]


@dataclass(frozen=True)
class LBMTrainingResult:
    """Fitted structured appliance plus auditable training diagnostics."""

    appliance: StructuredAppliance
    sources: tuple[SourceWindow, ...]
    evaluation_sources: tuple[SourceWindow, ...]
    num_samples: int
    window_length: int
    sample_period_seconds: int
    state_counts: tuple[int, ...]
    initial_counts: tuple[int, ...]
    transition_counts: tuple[tuple[int, ...], ...]
    cycle_counts: tuple[int, ...]
    emission_variance: float
    pseudocount: float
    minimum_windows_per_cycle: int
    variance_floor: float
    kmeans_max_iterations: int
    allow_temporal_evaluation: bool

    def metadata(self) -> dict:
        """Return a JSON-safe record sufficient to audit the fitted inputs."""
        summaries = []
        for name, summary in zip(
            ("energy_wh", "duration_minutes"), self.appliance.summaries
        ):
            summaries.append(
                {
                    "name": name,
                    "state_weights": torch.as_tensor(
                        summary.state_weights, dtype=torch.float64
                    ).tolist(),
                    "conditional_means": torch.as_tensor(
                        summary.conditional_means, dtype=torch.float64
                    ).tolist(),
                    "conditional_variance": summary.conditional_variance,
                    "induced_mean": summary.induced_mean,
                    "induced_variance": summary.induced_variance,
                    "weight": summary.weight,
                }
            )
        return {
            "schema_version": 1,
            "algorithm": "deterministic-1d-kmeans+hmm-counts",
            "sources": [asdict(source) for source in self.sources],
            "evaluation_sources": [
                asdict(source) for source in self.evaluation_sources
            ],
            "num_samples": self.num_samples,
            "window_length": self.window_length,
            "sample_period_seconds": self.sample_period_seconds,
            "state_means": torch.as_tensor(
                self.appliance.state_means, dtype=torch.float64
            ).tolist(),
            "initial_probabilities": torch.as_tensor(
                self.appliance.initial_probabilities, dtype=torch.float64
            ).tolist(),
            "transition_probabilities": torch.as_tensor(
                self.appliance.transition_probabilities, dtype=torch.float64
            ).tolist(),
            "off_state": self.appliance.off_state,
            "cycle_probabilities": torch.as_tensor(
                self.appliance.cycle_probabilities, dtype=torch.float64
            ).tolist(),
            "state_counts": list(self.state_counts),
            "initial_counts": list(self.initial_counts),
            "transition_counts": [list(row) for row in self.transition_counts],
            "cycle_counts": list(self.cycle_counts),
            "emission_variance": self.emission_variance,
            "pseudocount": self.pseudocount,
            "minimum_windows_per_cycle": self.minimum_windows_per_cycle,
            "variance_floor": self.variance_floor,
            "kmeans_max_iterations": self.kmeans_max_iterations,
            "allow_temporal_evaluation": self.allow_temporal_evaluation,
            "summaries": summaries,
        }


def _same_source_entity(first: SourceWindow, second: SourceWindow) -> bool:
    same_building = str(first.building) == str(second.building)
    same_dataset = first.dataset.casefold() == second.dataset.casefold()
    same_fingerprint = first.data_fingerprint == second.data_fingerprint
    same_uri = first.data_uri == second.data_uri
    return same_building and (same_dataset or same_fingerprint or same_uri)


def _windows_overlap(first: SourceWindow, second: SourceWindow) -> bool:
    if not _same_source_entity(first, second):
        return False
    first_start, first_end = first.interval
    second_start, second_end = second.interval
    return first_start < second_end and second_start < first_end


def assert_disjoint_sources(
    training_sources: Sequence[SourceWindow],
    evaluation_sources: Sequence[SourceWindow],
    *,
    allow_temporal_split: bool = False,
) -> None:
    """Reject evaluation-building leakage, with explicit temporal-split opt-in."""
    if not isinstance(allow_temporal_split, bool):
        raise TypeError("allow_temporal_split must be a boolean.")
    for training in training_sources:
        if not isinstance(training, SourceWindow):
            raise TypeError("training_sources must contain SourceWindow instances.")
        for evaluation in evaluation_sources:
            if not isinstance(evaluation, SourceWindow):
                raise TypeError(
                    "evaluation_sources must contain SourceWindow instances."
                )
            same_building = _same_source_entity(training, evaluation)
            if same_building and (
                not allow_temporal_split or _windows_overlap(training, evaluation)
            ):
                raise ValueError(
                    "Training/evaluation leakage: training data uses evaluation "
                    "building "
                    f"{training.dataset} building {training.building}."
                )


def _as_power(index: int, power) -> torch.Tensor:
    try:
        values = torch.as_tensor(power, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"windows[{index}].power must contain numeric values.") from exc
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError(
            f"windows[{index}].power must be non-empty and one-dimensional."
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"windows[{index}].power must contain only finite values.")
    if bool((values < 0).any()):
        raise ValueError(f"windows[{index}].power must be non-negative.")
    return values.clone()


def _prepare_windows(
    windows: Sequence[ApplianceTrainingWindow],
    evaluation_sources: Sequence[SourceWindow],
    *,
    allow_temporal_evaluation: bool,
) -> tuple[tuple[SourceWindow, torch.Tensor], ...]:
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)):
        raise TypeError("windows must be a non-empty sequence.")
    if not windows:
        raise ValueError("windows must be non-empty.")

    prepared = []
    for index, window in enumerate(windows):
        if not isinstance(window, ApplianceTrainingWindow):
            raise TypeError(
                f"windows[{index}] must be an ApplianceTrainingWindow instance."
            )
        if not isinstance(window.source, SourceWindow):
            raise TypeError(f"windows[{index}].source must be a SourceWindow.")
        values = _as_power(index, window.power)
        start, end = window.source.interval
        expected_seconds = values.numel() * window.source.sample_period_seconds
        if not math.isclose(end - start, expected_seconds, abs_tol=1e-6, rel_tol=0):
            raise ValueError(
                f"windows[{index}] timestamps span {end - start:g} seconds but "
                f"{values.numel()} samples require {expected_seconds:g} seconds."
            )
        prepared.append((window.source, values))

    periods = {source.sample_period_seconds for source, _ in prepared}
    lengths = {int(values.numel()) for _, values in prepared}
    if len(periods) != 1:
        raise ValueError("All training windows must use the same sample period.")
    if len(lengths) != 1:
        raise ValueError(
            "All population windows must contain the same number of samples."
        )

    sources = [source for source, _ in prepared]
    for index, source in enumerate(sources):
        for previous in sources[:index]:
            if _windows_overlap(previous, source):
                raise ValueError(
                    "Duplicate training evidence: training windows overlap for "
                    f"{source.dataset} building {source.building}."
                )
    assert_disjoint_sources(
        sources,
        evaluation_sources,
        allow_temporal_split=allow_temporal_evaluation,
    )
    return tuple(
        sorted(
            prepared,
            key=lambda pair: _source_sort_key(pair[0]),
        )
    )


def _source_sort_key(source: SourceWindow):
    return (
        source.dataset.casefold(),
        source.dataset_version,
        source.data_fingerprint,
        str(source.building),
        source.interval,
    )


def _fit_state_means(
    windows: Sequence[torch.Tensor],
    *,
    num_states: int,
    max_iterations: int,
) -> torch.Tensor:
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
        assignments = torch.argmin(torch.abs(values[:, None] - centers[None, :]), dim=1)
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


def _assign_states(values: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    return torch.argmin(torch.abs(values[:, None] - means[None, :]), dim=1)


def _smoothed_probabilities(counts: torch.Tensor, pseudocount: float) -> torch.Tensor:
    smoothed = counts.to(torch.float64) + pseudocount
    return smoothed / smoothed.sum(dim=-1, keepdim=True)


def _markov_reward_moments(
    initial: torch.Tensor,
    transition: torch.Tensor,
    rewards: torch.Tensor,
    time_points: int,
) -> tuple[float, float]:
    """Return exact mean and variance of an additive finite-state reward."""
    probabilities = initial
    first_moment = initial * rewards
    second_moment = initial * torch.square(rewards)
    for _ in range(1, time_points):
        propagated_first = first_moment @ transition
        probabilities = probabilities @ transition
        second_moment = (
            second_moment @ transition
            + 2.0 * rewards * propagated_first
            + torch.square(rewards) * probabilities
        )
        first_moment = propagated_first + rewards * probabilities
    mean = float(first_moment.sum())
    raw_variance = float(second_moment.sum()) - mean**2
    if not math.isfinite(mean) or not math.isfinite(raw_variance):
        raise RuntimeError(
            "HMM reward moments became non-finite; rescale the power data."
        )
    tolerance = 1e-10 * max(1.0, mean**2, abs(float(second_moment.sum())))
    if raw_variance < -tolerance:
        raise RuntimeError("HMM reward variance became unexpectedly negative.")
    variance = max(0.0, raw_variance)
    return mean, variance


def _fit_summary(
    name: str,
    observed: torch.Tensor,
    cycle_assignments: torch.Tensor,
    cycle_categories: int,
    state_weights: torch.Tensor,
    initial: torch.Tensor,
    transition: torch.Tensor,
    window_length: int,
    variance_floor: float,
) -> GaussianRatioSummary:
    conditional_means = torch.stack(
        [
            observed[cycle_assignments == cycle].mean()
            for cycle in range(cycle_categories)
        ]
    )
    residuals = observed - conditional_means[cycle_assignments]
    degrees_of_freedom = observed.numel() - cycle_categories
    if degrees_of_freedom <= 0:
        raise ValueError(f"{name} needs more windows than fitted cycle categories.")
    conditional_variance = max(
        variance_floor,
        float(torch.dot(residuals, residuals)) / degrees_of_freedom,
    )
    if not bool(torch.isfinite(conditional_means).all()) or not math.isfinite(
        conditional_variance
    ):
        raise RuntimeError(
            f"{name} population statistics became non-finite; rescale the data."
        )
    induced_mean, induced_variance = _markov_reward_moments(
        initial,
        transition,
        state_weights,
        window_length,
    )
    if induced_variance <= conditional_variance:
        raise ValueError(
            f"{name} conditional variance {conditional_variance:g} must be smaller "
            f"than induced HMM variance {induced_variance:g}; the LBM ratio would "
            "not be convex."
        )
    return GaussianRatioSummary(
        state_weights=state_weights,
        conditional_means=conditional_means,
        conditional_variance=conditional_variance,
        induced_mean=induced_mean,
        induced_variance=induced_variance,
    )


def fit_lbm_appliance(
    windows: Sequence[ApplianceTrainingWindow],
    *,
    num_states: int = 2,
    pseudocount: float = 1.0,
    minimum_windows_per_cycle: int = 2,
    variance_floor: float = 1e-6,
    kmeans_max_iterations: int = 100,
    evaluation_sources: Sequence[SourceWindow] = (),
    allow_temporal_evaluation: bool = False,
) -> LBMTrainingResult:
    """Fit one private LBM appliance contract from dense population windows."""
    num_states = _positive_int("num_states", num_states)
    if num_states < 2:
        raise ValueError("num_states must be at least two.")
    pseudocount = _finite_real("pseudocount", pseudocount, positive=True)
    minimum_windows_per_cycle = _positive_int(
        "minimum_windows_per_cycle", minimum_windows_per_cycle
    )
    variance_floor = _finite_real("variance_floor", variance_floor, positive=True)
    kmeans_max_iterations = _positive_int(
        "kmeans_max_iterations", kmeans_max_iterations
    )
    if not isinstance(allow_temporal_evaluation, bool):
        raise TypeError("allow_temporal_evaluation must be a boolean.")
    if not isinstance(evaluation_sources, Sequence) or isinstance(
        evaluation_sources, (str, bytes)
    ):
        raise TypeError(
            "evaluation_sources must be a sequence of SourceWindow instances."
        )
    evaluation_sources = tuple(evaluation_sources)
    prepared = _prepare_windows(
        windows,
        evaluation_sources,
        allow_temporal_evaluation=allow_temporal_evaluation,
    )
    sources = tuple(source for source, _ in prepared)
    powers = tuple(values for _, values in prepared)
    window_length = int(powers[0].numel())
    sample_period = sources[0].sample_period_seconds

    state_means = _fit_state_means(
        powers,
        num_states=num_states,
        max_iterations=kmeans_max_iterations,
    )
    state_windows = tuple(_assign_states(values, state_means) for values in powers)
    state_counts = sum(
        (torch.bincount(states, minlength=num_states) for states in state_windows),
        start=torch.zeros(num_states, dtype=torch.long),
    )
    initial_counts = torch.bincount(
        torch.stack([states[0] for states in state_windows]),
        minlength=num_states,
    )
    transition_counts = torch.zeros((num_states, num_states), dtype=torch.long)
    for states in state_windows:
        flat_transitions = states[:-1] * num_states + states[1:]
        transition_counts += torch.bincount(
            flat_transitions,
            minlength=num_states * num_states,
        ).reshape(num_states, num_states)
    initial = _smoothed_probabilities(initial_counts, pseudocount)
    transition = _smoothed_probabilities(transition_counts, pseudocount)

    off_state = int(torch.argmin(state_means))
    cycles = torch.tensor(
        [
            int(((states[:-1] == off_state) & (states[1:] != off_state)).sum())
            for states in state_windows
        ],
        dtype=torch.long,
    )
    cycle_categories = int(cycles.max()) + 1
    cycle_counts = torch.bincount(cycles, minlength=cycle_categories)
    sparse_categories = torch.nonzero(
        cycle_counts < minimum_windows_per_cycle,
        as_tuple=False,
    ).flatten()
    if sparse_categories.numel():
        categories = ", ".join(str(int(value)) for value in sparse_categories)
        raise ValueError(
            "Population windows are underspecified: cycle categories "
            f"{categories} have fewer than minimum_windows_per_cycle="
            f"{minimum_windows_per_cycle}."
        )
    cycle_probabilities = _smoothed_probabilities(cycle_counts, pseudocount)

    energy = torch.tensor(
        [float(values.sum()) * sample_period / 3600.0 for values in powers],
        dtype=torch.float64,
    )
    duration = torch.tensor(
        [
            float((states != off_state).sum()) * sample_period / 60.0
            for states in state_windows
        ],
        dtype=torch.float64,
    )
    if not bool(torch.isfinite(energy).all()) or not bool(
        torch.isfinite(duration).all()
    ):
        raise RuntimeError(
            "Population summaries became non-finite; rescale the power data."
        )
    energy_weights = state_means * sample_period / 3600.0
    duration_weights = torch.full_like(state_means, sample_period / 60.0)
    duration_weights[off_state] = 0.0
    summaries = (
        _fit_summary(
            "energy_wh",
            energy,
            cycles,
            cycle_categories,
            energy_weights,
            initial,
            transition,
            window_length,
            variance_floor,
        ),
        _fit_summary(
            "duration_minutes",
            duration,
            cycles,
            cycle_categories,
            duration_weights,
            initial,
            transition,
            window_length,
            variance_floor,
        ),
    )
    all_power = torch.cat(powers)
    all_states = torch.cat(state_windows)
    emission_residuals = all_power - state_means[all_states]
    emission_variance = max(
        variance_floor,
        float(torch.mean(torch.square(emission_residuals))),
    )
    if not math.isfinite(emission_variance):
        raise RuntimeError(
            "Emission variance became non-finite; rescale the power data."
        )
    appliance = StructuredAppliance(
        state_means=state_means,
        initial_probabilities=initial,
        transition_probabilities=transition,
        off_state=off_state,
        cycle_probabilities=cycle_probabilities,
        summaries=summaries,
    )
    return LBMTrainingResult(
        appliance=appliance,
        sources=sources,
        evaluation_sources=tuple(sorted(evaluation_sources, key=_source_sort_key)),
        num_samples=int(all_power.numel()),
        window_length=window_length,
        sample_period_seconds=sample_period,
        state_counts=tuple(int(value) for value in state_counts),
        initial_counts=tuple(int(value) for value in initial_counts),
        transition_counts=tuple(
            tuple(int(value) for value in row) for row in transition_counts
        ),
        cycle_counts=tuple(int(value) for value in cycle_counts),
        emission_variance=emission_variance,
        pseudocount=pseudocount,
        minimum_windows_per_cycle=minimum_windows_per_cycle,
        variance_floor=variance_floor,
        kmeans_max_iterations=kmeans_max_iterations,
        allow_temporal_evaluation=allow_temporal_evaluation,
    )


__all__ = [
    "ApplianceTrainingWindow",
    "LBMTrainingResult",
    "SourceWindow",
    "assert_disjoint_sources",
    "fit_lbm_appliance",
]
