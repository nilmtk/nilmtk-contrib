"""Supervised explicit-duration hidden semi-Markov NILM baseline."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
import math

import numpy as np
import pandas as pd
import torch
from nilmtk.disaggregate import Disaggregator

from nilmtk_contrib.torch._base import resolve_torch_device
from nilmtk_contrib.torch._data import power_vector
from nilmtk_contrib.torch._semi_markov import semi_markov_viterbi
from nilmtk_contrib.torch._state_fitting import assign_states, fit_state_means
from nilmtk_contrib.utils.checkpoints import unsupported_persistence
from nilmtk_contrib.utils.model import initialize_runtime
from nilmtk_contrib.utils.params import (
    DEFAULT_ALIASES,
    get_param,
    validate_positive_int,
)


@dataclass(frozen=True)
class _HSMMParameters:
    state_means: tuple[float, ...]
    aggregate_means: tuple[float, ...]
    aggregate_variances: tuple[float, ...]
    initial_probabilities: tuple[float, ...]
    transition_probabilities: tuple[tuple[float, ...], ...]
    duration_probabilities: tuple[tuple[float, ...], ...]
    state_counts: tuple[int, ...]
    initial_counts: tuple[int, ...]
    transition_counts: tuple[tuple[int, ...], ...]
    duration_counts: tuple[tuple[int, ...], ...]
    num_samples: int
    num_chunks: int
    num_segments: int
    right_censored_segments: int


def _positive_finite(name, value) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number.")
    return float(value)


def _as_power_tensor(frame, label) -> torch.Tensor:
    values = power_vector(frame, label)
    if bool((values < 0).any()):
        raise ValueError(f"{label} must be non-negative.")
    return torch.as_tensor(values, dtype=torch.float64, device="cpu").clone()


def _frame_index(frame, length):
    index = getattr(frame, "index", None)
    return index if index is not None else pd.RangeIndex(length)


def _aligned_windows(main_frames, target_frames, appliance_name):
    if len(main_frames) != len(target_frames):
        raise ValueError(
            f"{appliance_name!r} has {len(target_frames)} chunks but mains has "
            f"{len(main_frames)}."
        )
    mains = []
    targets = []
    for index, (main_frame, target_frame) in enumerate(zip(main_frames, target_frames)):
        main = _as_power_tensor(main_frame, f"mains chunk {index}")
        target = _as_power_tensor(
            target_frame, f"{appliance_name!r} target chunk {index}"
        )
        if main.numel() != target.numel():
            raise ValueError(
                f"{appliance_name!r} target chunk {index} length does not match mains."
            )
        main_index = getattr(main_frame, "index", None)
        target_index = getattr(target_frame, "index", None)
        if (
            main_index is not None
            and target_index is not None
            and not main_index.equals(target_index)
        ):
            raise ValueError(
                f"{appliance_name!r} target chunk {index} index does not match mains."
            )
        mains.append(main)
        targets.append(target)
    return tuple(mains), tuple(targets)


def _runs(states: torch.Tensor) -> tuple[tuple[int, int], ...]:
    labels = states.tolist()
    runs = []
    start = 0
    for end in range(1, len(labels) + 1):
        if end == len(labels) or labels[end] != labels[start]:
            runs.append((labels[start], end - start))
            start = end
    return tuple(runs)


def _fit_hsmm(
    mains_windows,
    target_windows,
    *,
    num_states,
    max_duration,
    pseudocount,
    variance_floor,
    kmeans_max_iterations,
) -> _HSMMParameters:
    state_means = fit_state_means(
        target_windows,
        num_states=num_states,
        max_iterations=kmeans_max_iterations,
    )
    state_windows = tuple(
        assign_states(values, state_means) for values in target_windows
    )
    state_counts = sum(
        (torch.bincount(states, minlength=num_states) for states in state_windows),
        start=torch.zeros(num_states, dtype=torch.long),
    )
    initial_counts = torch.zeros(num_states, dtype=torch.long)
    transition_counts = torch.zeros((num_states, num_states), dtype=torch.long)
    duration_counts = torch.zeros((num_states, max_duration), dtype=torch.long)
    num_segments = 0
    right_censored_segments = 0
    for states in state_windows:
        runs = _runs(states)
        initial_counts[runs[0][0]] += 1
        num_segments += len(runs)
        for run_index, (state, duration) in enumerate(runs):
            if duration >= max_duration:
                right_censored_segments += int(duration > max_duration)
                duration = max_duration
            duration_counts[state, duration - 1] += 1
            if run_index:
                transition_counts[runs[run_index - 1][0], state] += 1

    initial_prior = pseudocount / num_states
    initial_probabilities = initial_counts.to(torch.float64) + initial_prior
    initial_probabilities /= initial_probabilities.sum()

    off_diagonal = ~torch.eye(num_states, dtype=torch.bool)
    transition_prior = pseudocount / (num_states - 1)
    transition_probabilities = transition_counts.to(torch.float64)
    transition_probabilities[off_diagonal] += transition_prior
    transition_probabilities /= transition_probabilities.sum(dim=1, keepdim=True)

    duration_prior = pseudocount / max_duration
    duration_probabilities = duration_counts.to(torch.float64) + duration_prior
    duration_probabilities /= duration_probabilities.sum(dim=1, keepdim=True)

    all_mains = torch.cat(mains_windows)
    all_states = torch.cat(state_windows)
    aggregate_means = torch.stack(
        [all_mains[all_states == state].mean() for state in range(num_states)]
    )
    aggregate_variances = torch.stack(
        [
            torch.square(all_mains[all_states == state] - aggregate_means[state]).mean()
            for state in range(num_states)
        ]
    ).clamp_min(variance_floor)
    if not bool(torch.isfinite(aggregate_means).all()) or not bool(
        torch.isfinite(aggregate_variances).all()
    ):
        raise RuntimeError("HSMM emission fitting produced non-finite parameters.")

    return _HSMMParameters(
        state_means=tuple(float(value) for value in state_means),
        aggregate_means=tuple(float(value) for value in aggregate_means),
        aggregate_variances=tuple(float(value) for value in aggregate_variances),
        initial_probabilities=tuple(float(value) for value in initial_probabilities),
        transition_probabilities=tuple(
            tuple(float(value) for value in row) for row in transition_probabilities
        ),
        duration_probabilities=tuple(
            tuple(float(value) for value in row) for row in duration_probabilities
        ),
        state_counts=tuple(int(value) for value in state_counts),
        initial_counts=tuple(int(value) for value in initial_counts),
        transition_counts=tuple(
            tuple(int(value) for value in row) for row in transition_counts
        ),
        duration_counts=tuple(
            tuple(int(value) for value in row) for row in duration_counts
        ),
        num_samples=int(all_mains.numel()),
        num_chunks=len(mains_windows),
        num_segments=num_segments,
        right_censored_segments=right_censored_segments,
    )


def _decode_block(values, model: _HSMMParameters, device) -> np.ndarray:
    observations = torch.as_tensor(values, dtype=torch.float64, device=device)
    state_means = torch.tensor(model.state_means, dtype=torch.float64, device=device)
    aggregate_means = torch.tensor(
        model.aggregate_means, dtype=torch.float64, device=device
    )
    aggregate_variances = torch.tensor(
        model.aggregate_variances, dtype=torch.float64, device=device
    )
    initial_scores = torch.log(
        torch.tensor(model.initial_probabilities, dtype=torch.float64, device=device)
    )
    transition_scores = torch.log(
        torch.tensor(model.transition_probabilities, dtype=torch.float64, device=device)
    )
    duration_scores = torch.log(
        torch.tensor(model.duration_probabilities, dtype=torch.float64, device=device)
    )
    emission_scores = -0.5 * (
        torch.square(observations[:, None] - aggregate_means[None, :])
        / aggregate_variances[None, :]
        + torch.log(2 * torch.pi * aggregate_variances)[None, :]
    )
    path = semi_markov_viterbi(
        emission_scores,
        initial_scores,
        transition_scores,
        duration_scores,
    )
    return state_means[path.states].to(dtype=torch.float32).cpu().numpy()


class HSMM(Disaggregator):
    """Explicit-duration Gaussian HSMM fitted from aligned appliance labels."""

    MODEL_NAME = "HSMM"

    def __init__(self, params=None):
        params = {} if params is None else params
        if not isinstance(params, Mapping):
            raise TypeError("params must be a mapping or None.")
        super().__init__()
        self.num_states = validate_positive_int(
            "num_states", get_param(params, "num_states", 2)
        )
        if self.num_states < 2:
            raise ValueError("num_states must be at least two.")
        self.max_duration = validate_positive_int(
            "max_duration", get_param(params, "max_duration", 720)
        )
        self.pseudocount = _positive_finite(
            "pseudocount", get_param(params, "pseudocount", 1.0)
        )
        self.variance_floor = _positive_finite(
            "variance_floor", get_param(params, "variance_floor", 1.0)
        )
        self.kmeans_max_iterations = validate_positive_int(
            "kmeans_max_iterations",
            get_param(params, "kmeans_max_iterations", 100),
        )
        self.device = resolve_torch_device(get_param(params, "device"))
        if self.device.type == "mps":
            raise ValueError("HSMM supports CPU and CUDA; float64 MPS is unsupported.")
        self.seed = get_param(params, "seed")
        self.verbose = get_param(params, "verbose", False)
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, Integral)
        ):
            raise ValueError("seed must be an integer or None.")
        if not isinstance(self.verbose, bool):
            raise ValueError("verbose must be a boolean.")
        self.chunk_wise_training = False
        self.save_model_path = get_param(
            params,
            "save_model_path",
            aliases=DEFAULT_ALIASES["save_model_path"],
        )
        self.load_model_path = get_param(
            params,
            "pretrained_model_path",
            aliases=DEFAULT_ALIASES["pretrained_model_path"],
        )
        self.models = OrderedDict()
        initialize_runtime(
            self,
            {"seed": self.seed, "verbose": self.verbose},
            backends=("python", "numpy", "torch"),
        )
        if self.load_model_path:
            self.load_model(self.load_model_path)

    def save_model(self, *_, **__):
        unsupported_persistence(self.MODEL_NAME)

    def load_model(self, *_, **__):
        unsupported_persistence(self.MODEL_NAME)

    def partial_fit(
        self,
        train_main,
        train_appliances,
        do_preprocessing=True,
        current_epoch=0,
        **_,
    ):
        del current_epoch
        if not isinstance(do_preprocessing, bool):
            raise ValueError("do_preprocessing must be a boolean.")
        main_frames = list(train_main)
        if not main_frames:
            raise ValueError("Training requires at least one mains chunk.")
        try:
            appliance_entries = list(train_appliances)
        except TypeError as exc:
            raise TypeError(
                "train_appliances must contain (name, frames) pairs."
            ) from exc
        if not appliance_entries:
            raise ValueError("Training requires at least one appliance.")
        fitted = OrderedDict()
        for entry in appliance_entries:
            try:
                appliance_name, frames = entry
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Each appliance must be a (name, frames) pair."
                ) from exc
            if not isinstance(appliance_name, str) or not appliance_name.strip():
                raise ValueError("Appliance names must be non-empty strings.")
            if appliance_name != appliance_name.strip():
                raise ValueError(
                    "Appliance names must not have surrounding whitespace."
                )
            if appliance_name in fitted:
                raise ValueError(f"Duplicate appliance name {appliance_name!r}.")
            if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
                raise TypeError(
                    f"Training frames for {appliance_name!r} must be a sequence."
                )
            mains, targets = _aligned_windows(main_frames, list(frames), appliance_name)
            fitted[appliance_name] = _fit_hsmm(
                mains,
                targets,
                num_states=self.num_states,
                max_duration=self.max_duration,
                pseudocount=self.pseudocount,
                variance_floor=self.variance_floor,
                kmeans_max_iterations=self.kmeans_max_iterations,
            )
        previous_models = self.models
        self.models = fitted
        try:
            if self.save_model_path:
                self.save_model(self.save_model_path)
        except Exception:
            self.models = previous_models
            raise

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        if not isinstance(do_preprocessing, bool):
            raise ValueError("do_preprocessing must be a boolean.")
        models = self.models if model is None else model
        if not isinstance(models, Mapping) or not models:
            raise RuntimeError("HSMM requires a trained model mapping.")
        for appliance_name, fitted in models.items():
            if not isinstance(appliance_name, str) or not appliance_name.strip():
                raise ValueError(
                    "HSMM model appliance names must be non-empty strings."
                )
            if not isinstance(fitted, _HSMMParameters):
                raise TypeError("HSMM model values must be fitted HSMM parameters.")
            if len(fitted.state_means) != self.num_states:
                raise ValueError("HSMM model state count does not match configuration.")
            if any(
                len(probabilities) != self.max_duration
                for probabilities in fitted.duration_probabilities
            ):
                raise ValueError(
                    "HSMM model duration cap does not match configuration."
                )
        results = []
        for chunk_index, frame in enumerate(test_main_list):
            values = power_vector(frame, f"mains chunk {chunk_index}", allow_empty=True)
            if bool((values < 0).any()):
                raise ValueError(f"mains chunk {chunk_index} must be non-negative.")
            output_index = _frame_index(frame, len(values))
            columns = OrderedDict()
            for appliance_name, fitted in models.items():
                blocks = [
                    _decode_block(
                        values[start : start + self.max_duration],
                        fitted,
                        self.device,
                    )
                    for start in range(0, len(values), self.max_duration)
                ]
                prediction = (
                    np.concatenate(blocks).astype(np.float32, copy=False)
                    if blocks
                    else np.empty(0, dtype=np.float32)
                )
                columns[appliance_name] = pd.Series(prediction, index=output_index)
            results.append(pd.DataFrame(columns, index=output_index))
        return results


__all__ = ["HSMM"]
