"""Supervised explicit-duration hidden semi-Markov NILM baseline."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from numbers import Integral, Real
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nilmtk_contrib.torch._base import TorchStateSpaceDisaggregator
from nilmtk_contrib.torch._data import power_vector
from nilmtk_contrib.torch._semi_markov import semi_markov_viterbi
from nilmtk_contrib.torch._state_space_data import (
    aligned_power_windows,
    frame_index,
)
from nilmtk_contrib.torch._state_fitting import assign_states, fit_state_means
from nilmtk_contrib.utils.checkpoints import load_json_strict, save_json_atomic
from nilmtk_contrib.utils.params import get_param, validate_positive_int


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


_ARTIFACT_FILENAME = "hsmm.json"
_ARTIFACT_SCHEMA_VERSION = 1
_CONFIG_FIELDS = (
    "num_states",
    "max_duration",
    "pseudocount",
    "variance_floor",
    "kmeans_max_iterations",
)
_PARAMETER_FIELDS = {field.name for field in fields(_HSMMParameters)}


def _positive_finite(name, value) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number.")
    return float(value)


def _positive_integer(name, value) -> int:
    return validate_positive_int(name, value)


def _count_vector(name, values, length) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} counts.")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must contain non-negative integers.")
        result.append(int(value))
    return tuple(result)


def _finite_vector(name, values, length, *, positive=False, non_negative=False):
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} values.")
    result = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or (positive and value <= 0)
            or (non_negative and value < 0)
        ):
            qualifier = (
                "positive finite"
                if positive
                else "non-negative finite"
                if non_negative
                else "finite"
            )
            raise ValueError(f"{name} must contain {qualifier} values.")
        result.append(float(value))
    return tuple(result)


def _matrix(name, values, rows, columns, converter):
    if not isinstance(values, (list, tuple)) or len(values) != rows:
        raise ValueError(f"{name} must contain exactly {rows} rows.")
    return tuple(
        converter(f"{name}[{index}]", row, columns)
        for index, row in enumerate(values)
    )


def _validate_fitted_model(model, *, num_states, max_duration) -> None:
    if not isinstance(model, _HSMMParameters):
        raise TypeError("HSMM model values must be fitted HSMM parameters.")
    state_means = _finite_vector(
        "state_means", model.state_means, num_states, non_negative=True
    )
    if any(left >= right for left, right in zip(state_means, state_means[1:])):
        raise ValueError("state_means must be strictly increasing.")
    _finite_vector(
        "aggregate_means", model.aggregate_means, num_states, non_negative=True
    )
    _finite_vector(
        "aggregate_variances", model.aggregate_variances, num_states, positive=True
    )
    initial = _finite_vector(
        "initial_probabilities", model.initial_probabilities, num_states, positive=True
    )
    transitions = _matrix(
        "transition_probabilities",
        model.transition_probabilities,
        num_states,
        num_states,
        lambda name, row, length: _finite_vector(
            name, row, length, non_negative=True
        ),
    )
    durations = _matrix(
        "duration_probabilities",
        model.duration_probabilities,
        num_states,
        max_duration,
        lambda name, row, length: _finite_vector(name, row, length, positive=True),
    )
    if not math.isclose(sum(initial), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("initial_probabilities must sum to one.")
    for state, row in enumerate(transitions):
        if row[state] != 0 or any(
            probability <= 0
            for other, probability in enumerate(row)
            if other != state
        ):
            raise ValueError(
                "transition_probabilities must have a zero diagonal and positive "
                "off-diagonal values."
            )
        if not math.isclose(sum(row), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("transition_probabilities rows must sum to one.")
    if any(
        not math.isclose(sum(row), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        for row in durations
    ):
        raise ValueError("duration_probabilities rows must sum to one.")

    state_counts = _count_vector("state_counts", model.state_counts, num_states)
    if any(count == 0 for count in state_counts):
        raise ValueError("state_counts must show at least one sample in every state.")
    initial_counts = _count_vector("initial_counts", model.initial_counts, num_states)
    transition_counts = _matrix(
        "transition_counts",
        model.transition_counts,
        num_states,
        num_states,
        _count_vector,
    )
    duration_counts = _matrix(
        "duration_counts",
        model.duration_counts,
        num_states,
        max_duration,
        _count_vector,
    )
    num_samples = _positive_integer("num_samples", model.num_samples)
    num_chunks = _positive_integer("num_chunks", model.num_chunks)
    num_segments = _positive_integer("num_segments", model.num_segments)
    right_censored = model.right_censored_segments
    if (
        isinstance(right_censored, bool)
        or not isinstance(right_censored, Integral)
        or not 0 <= right_censored <= num_segments
    ):
        raise ValueError(
            "right_censored_segments must be an integer between zero and num_segments."
        )
    if any(row[state] != 0 for state, row in enumerate(transition_counts)):
        raise ValueError("transition_counts must have a zero diagonal.")
    if num_segments < num_chunks:
        raise ValueError("num_segments must be at least num_chunks.")
    expected_totals = (
        (sum(state_counts), num_samples, "state_counts"),
        (sum(initial_counts), num_chunks, "initial_counts"),
        (
            sum(map(sum, transition_counts)),
            num_segments - num_chunks,
            "transition_counts",
        ),
        (sum(map(sum, duration_counts)), num_segments, "duration_counts"),
    )
    for observed, expected, name in expected_totals:
        if observed != expected:
            raise ValueError(f"{name} total is inconsistent with fitted metadata.")


def _parameters_from_payload(payload, *, num_states, max_duration):
    if not isinstance(payload, Mapping) or set(payload) != _PARAMETER_FIELDS:
        raise ValueError("HSMM artifact has invalid fitted-parameter fields.")
    try:
        parameters = _HSMMParameters(
            state_means=tuple(payload["state_means"]),
            aggregate_means=tuple(payload["aggregate_means"]),
            aggregate_variances=tuple(payload["aggregate_variances"]),
            initial_probabilities=tuple(payload["initial_probabilities"]),
            transition_probabilities=tuple(
                tuple(row) for row in payload["transition_probabilities"]
            ),
            duration_probabilities=tuple(
                tuple(row) for row in payload["duration_probabilities"]
            ),
            state_counts=tuple(payload["state_counts"]),
            initial_counts=tuple(payload["initial_counts"]),
            transition_counts=tuple(tuple(row) for row in payload["transition_counts"]),
            duration_counts=tuple(tuple(row) for row in payload["duration_counts"]),
            num_samples=payload["num_samples"],
            num_chunks=payload["num_chunks"],
            num_segments=payload["num_segments"],
            right_censored_segments=payload["right_censored_segments"],
        )
    except (TypeError, KeyError) as exc:
        raise ValueError("HSMM artifact fitted parameters are malformed.") from exc
    _validate_fitted_model(
        parameters, num_states=num_states, max_duration=max_duration
    )
    return parameters


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

    fitted = _HSMMParameters(
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
    _validate_fitted_model(
        fitted, num_states=num_states, max_duration=max_duration
    )
    return fitted


def _decode_sequence(values, model: _HSMMParameters, device) -> np.ndarray:
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


class HSMM(TorchStateSpaceDisaggregator):
    """Explicit-duration Gaussian HSMM fitted from aligned appliance labels."""

    MODEL_NAME = "HSMM"

    def __init__(self, params=None):
        super().__init__(params)
        params = {} if params is None else params
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
        if self.device.type == "mps":
            raise ValueError("HSMM supports CPU and CUDA; float64 MPS is unsupported.")
        if self.chunk_wise_training:
            raise ValueError("HSMM does not support chunk_wise_training.")
        if self.load_model_path:
            self.load_model(self.load_model_path)

    def _validate_model_record(self, appliance_name, model):
        del appliance_name
        _validate_fitted_model(
            model,
            num_states=self.num_states,
            max_duration=self.max_duration,
        )

    def save_model(self, path=None):
        target = path if path is not None else self.save_model_path
        if target is None:
            raise ValueError("HSMM save_model requires a checkpoint directory.")
        models = self.require_models()
        serialized_models = {}
        for appliance_name, fitted in models.items():
            serialized_models[appliance_name] = asdict(fitted)
        payload = {
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "model_class": self.MODEL_NAME,
            "config": {name: getattr(self, name) for name in _CONFIG_FIELDS},
            "models": serialized_models,
        }
        save_json_atomic(Path(target) / _ARTIFACT_FILENAME, payload)

    def load_model(self, path=None):
        source = path if path is not None else self.load_model_path
        if source is None:
            raise ValueError("HSMM load_model requires a checkpoint directory.")
        artifact_path = Path(source) / _ARTIFACT_FILENAME
        payload = load_json_strict(artifact_path, description="HSMM artifact")
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "model_class",
            "config",
            "models",
        }:
            raise ValueError("HSMM artifact has invalid top-level fields.")
        if (
            isinstance(payload["schema_version"], bool)
            or not isinstance(payload["schema_version"], Integral)
            or payload["schema_version"] != _ARTIFACT_SCHEMA_VERSION
        ):
            raise ValueError("HSMM artifact has an unsupported schema_version.")
        if payload["model_class"] != self.MODEL_NAME:
            raise ValueError("HSMM artifact model_class does not match HSMM.")
        config = payload["config"]
        if not isinstance(config, Mapping) or set(config) != set(_CONFIG_FIELDS):
            raise ValueError("HSMM artifact has invalid configuration fields.")
        num_states = _positive_integer("num_states", config["num_states"])
        if num_states < 2:
            raise ValueError("num_states must be at least two.")
        max_duration = _positive_integer("max_duration", config["max_duration"])
        pseudocount = _positive_finite("pseudocount", config["pseudocount"])
        variance_floor = _positive_finite(
            "variance_floor", config["variance_floor"]
        )
        kmeans_max_iterations = _positive_integer(
            "kmeans_max_iterations", config["kmeans_max_iterations"]
        )
        model_payloads = payload["models"]
        if not isinstance(model_payloads, Mapping) or not model_payloads:
            raise ValueError("HSMM artifact must contain at least one appliance model.")
        loaded = OrderedDict()
        for appliance_name, fitted_payload in model_payloads.items():
            if (
                not isinstance(appliance_name, str)
                or not appliance_name.strip()
                or appliance_name != appliance_name.strip()
                or appliance_name in loaded
            ):
                raise ValueError("HSMM artifact has an invalid appliance name.")
            loaded[appliance_name] = _parameters_from_payload(
                fitted_payload,
                num_states=num_states,
                max_duration=max_duration,
            )
        self.num_states = num_states
        self.max_duration = max_duration
        self.pseudocount = pseudocount
        self.variance_floor = variance_floor
        self.kmeans_max_iterations = kmeans_max_iterations
        self.models = loaded

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
            mains, targets = aligned_power_windows(
                main_frames, list(frames), appliance_name
            )
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
        models = self.require_models(model)
        results = []
        for chunk_index, frame in enumerate(test_main_list):
            values = power_vector(frame, f"mains chunk {chunk_index}", allow_empty=True)
            if bool((values < 0).any()):
                raise ValueError(f"mains chunk {chunk_index} must be non-negative.")
            output_index = frame_index(frame, len(values))
            columns = OrderedDict()
            for appliance_name, fitted in models.items():
                prediction = (
                    _decode_sequence(values, fitted, self.device)
                    if len(values)
                    else np.empty(0, dtype=np.float32)
                )
                columns[appliance_name] = pd.Series(prediction, index=output_index)
            results.append(pd.DataFrame(columns, index=output_index))
        return results


__all__ = ["HSMM"]
