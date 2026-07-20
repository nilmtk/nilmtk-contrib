"""Solver-free discriminative sparse coding for NILM."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nilmtk_contrib.torch._base import TorchStateSpaceDisaggregator
from nilmtk_contrib.torch._data import power_vector
from nilmtk_contrib.utils.checkpoints import save_json_atomic
from nilmtk_contrib.utils.params import get_param, validate_positive_int


@dataclass(frozen=True)
class SparseCodeResult:
    """Result and stopping certificate for non-negative sparse coding."""

    codes: torch.Tensor
    objective: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class _DSCParameters:
    reconstruction_dictionary: tuple[tuple[float, ...], ...]
    discriminative_dictionary: tuple[tuple[float, ...], ...]
    training_windows: int
    reconstruction_objective: float
    reconstruction_iterations: int
    reconstruction_converged: bool
    activation_error: float


_ARTIFACT_FILENAME = "dsc.json"
_ARTIFACT_SCHEMA_VERSION = 1
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_CONFIG_FIELDS = (
    "shape",
    "n_components",
    "sparsity_coefficient",
    "dictionary_iterations",
    "discriminative_iterations",
    "sparse_code_iterations",
    "tolerance",
    "discriminative_learning_rate",
    "enforce_aggregate",
)
_PARAMETER_FIELDS = {field.name for field in fields(_DSCParameters)}


def _positive_finite(name, value) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number.")
    return float(value)


def _non_negative_finite(name, value) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative finite number.")
    return float(value)


def _positive_integer(name, value) -> int:
    return validate_positive_int(name, value)


def _non_negative_integer(name, value) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _validate_tensor(name, value, *, non_negative=True) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if value.is_complex() or value.dtype not in {
        torch.float32,
        torch.float64,
    }:
        raise TypeError(f"{name} must use float32 or float64 real values.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite values.")
    if non_negative and bool((value < 0).any()):
        raise ValueError(f"{name} must contain non-negative values.")
    return value


def _sparse_objective(
    dictionary: torch.Tensor,
    observations: torch.Tensor,
    codes: torch.Tensor,
    sparsity_coefficient: float,
) -> torch.Tensor:
    residual = observations - dictionary @ codes
    return 0.5 * torch.square(residual).sum() + sparsity_coefficient * codes.sum()


@torch.no_grad()
def nonnegative_sparse_code(
    dictionary: torch.Tensor,
    observations: torch.Tensor,
    *,
    sparsity_coefficient: float,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
) -> SparseCodeResult:
    """Solve non-negative LASSO with monotone accelerated proximal gradient.

    The minimized objective is ``0.5 ||X - D A||² + alpha ||A||₁`` with
    ``A >= 0``. The positive soft-thresholding proximal map is exact, and the
    step is derived from the spectral norm of ``D`` rather than user tuning.
    """

    dictionary = _validate_tensor("dictionary", dictionary)
    observations = _validate_tensor("observations", observations)
    if dictionary.device != observations.device:
        raise ValueError("dictionary and observations must be on the same device.")
    if dictionary.dtype != observations.dtype:
        raise ValueError("dictionary and observations must have the same dtype.")
    if dictionary.shape[0] != observations.shape[0]:
        raise ValueError("dictionary and observations must have the same row count.")
    if dictionary.shape[1] == 0 or observations.shape[1] == 0:
        raise ValueError("dictionary and observations must have at least one column.")
    sparsity_coefficient = _non_negative_finite(
        "sparsity_coefficient", sparsity_coefficient
    )
    max_iterations = _positive_integer("max_iterations", max_iterations)
    tolerance = _positive_finite("tolerance", tolerance)

    spectral_norm = torch.linalg.matrix_norm(dictionary, ord=2)
    lipschitz = torch.square(spectral_norm)
    if not bool(torch.isfinite(lipschitz)) or float(lipschitz) <= 0:
        raise ValueError("dictionary must have a positive finite spectral norm.")
    step = lipschitz.reciprocal()
    threshold = step * sparsity_coefficient
    codes = torch.zeros(
        (dictionary.shape[1], observations.shape[1]),
        dtype=dictionary.dtype,
        device=dictionary.device,
    )
    extrapolated = codes.clone()
    momentum = 1.0
    previous_objective = _sparse_objective(
        dictionary, observations, codes, sparsity_coefficient
    )
    converged = False

    for iteration in range(1, max_iterations + 1):
        gradient = dictionary.T @ (dictionary @ extrapolated - observations)
        candidate = torch.clamp(extrapolated - step * gradient - threshold, min=0)
        candidate_objective = _sparse_objective(
            dictionary, observations, candidate, sparsity_coefficient
        )
        if candidate_objective > previous_objective:
            momentum = 1.0
            gradient = dictionary.T @ (dictionary @ codes - observations)
            candidate = torch.clamp(codes - step * gradient - threshold, min=0)
            candidate_objective = _sparse_objective(
                dictionary, observations, candidate, sparsity_coefficient
            )
        if not bool(torch.isfinite(candidate_objective)) or not bool(
            torch.isfinite(candidate).all()
        ):
            raise RuntimeError("Sparse coding produced non-finite values.")

        change = torch.linalg.vector_norm(candidate - codes)
        scale = 1.0 + torch.linalg.vector_norm(codes)
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum**2))
        extrapolated = candidate + (momentum - 1.0) / next_momentum * (
            candidate - codes
        )
        codes = candidate
        previous_objective = candidate_objective
        momentum = next_momentum
        if float(change) <= tolerance * float(scale):
            converged = True
            break

    return SparseCodeResult(
        codes=codes,
        objective=float(previous_objective),
        iterations=iteration,
        converged=converged,
    )


def _normalize_dictionary(
    dictionary: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    dictionary = torch.clamp(dictionary, min=0)
    norms = torch.linalg.vector_norm(dictionary, dim=0)
    missing = norms <= torch.finfo(dictionary.dtype).eps
    if bool(missing.any()):
        replacement = torch.clamp(fallback, min=0)
        replacement_norm = torch.linalg.vector_norm(replacement)
        if float(replacement_norm) <= torch.finfo(dictionary.dtype).eps:
            replacement = torch.ones_like(replacement)
            replacement_norm = torch.linalg.vector_norm(replacement)
        dictionary[:, missing] = (replacement / replacement_norm)[:, None]
        norms = torch.linalg.vector_norm(dictionary, dim=0)
    return dictionary / norms.clamp_min(torch.finfo(dictionary.dtype).eps)


def _initial_dictionary(observations: torch.Tensor, n_components: int) -> torch.Tensor:
    if not bool((observations > 0).any()):
        raise ValueError("Appliance training data must contain positive power.")
    indices = torch.arange(n_components, device=observations.device)
    indices = indices.remainder(observations.shape[1])
    dictionary = observations[:, indices].clone()
    fallback = observations.mean(dim=1)
    return _normalize_dictionary(dictionary, fallback)


def _learn_dictionary(
    observations: torch.Tensor,
    *,
    n_components: int,
    sparsity_coefficient: float,
    dictionary_iterations: int,
    sparse_code_iterations: int,
    tolerance: float,
) -> tuple[torch.Tensor, SparseCodeResult]:
    dictionary = _initial_dictionary(observations, n_components)
    fallback = observations.mean(dim=1)
    best_dictionary = dictionary.clone()
    best_result = nonnegative_sparse_code(
        dictionary,
        observations,
        sparsity_coefficient=sparsity_coefficient,
        max_iterations=sparse_code_iterations,
        tolerance=tolerance,
    )
    best_objective = best_result.objective

    for _ in range(dictionary_iterations):
        code_result = nonnegative_sparse_code(
            dictionary,
            observations,
            sparsity_coefficient=sparsity_coefficient,
            max_iterations=sparse_code_iterations,
            tolerance=tolerance,
        )
        codes = code_result.codes
        code_norm = torch.linalg.matrix_norm(codes, ord=2)
        lipschitz = torch.square(code_norm)
        if float(lipschitz) <= torch.finfo(dictionary.dtype).eps:
            break
        residual = observations - dictionary @ codes
        candidate = dictionary + residual @ codes.T / lipschitz
        candidate = _normalize_dictionary(candidate, fallback)
        candidate_result = nonnegative_sparse_code(
            candidate,
            observations,
            sparsity_coefficient=sparsity_coefficient,
            max_iterations=sparse_code_iterations,
            tolerance=tolerance,
        )
        if candidate_result.objective < best_objective:
            best_dictionary = candidate.clone()
            best_result = candidate_result
            relative_improvement = (best_objective - candidate_result.objective) / max(
                1.0, abs(best_objective)
            )
            best_objective = candidate_result.objective
            dictionary = candidate
            if relative_improvement <= tolerance:
                break
        else:
            break

    return best_dictionary, best_result


def _fit_discriminative_dictionary(
    aggregate: torch.Tensor,
    reconstruction_dictionary: torch.Tensor,
    target_codes: torch.Tensor,
    *,
    sparsity_coefficient: float,
    discriminative_iterations: int,
    sparse_code_iterations: int,
    tolerance: float,
    learning_rate: float,
) -> tuple[torch.Tensor, float]:
    dictionary = reconstruction_dictionary.clone()
    fallback = dictionary.mean(dim=1)
    validation_windows = int(aggregate.shape[1] * 0.2)
    if validation_windows:
        train_aggregate = aggregate[:, :-validation_windows]
        validation_aggregate = aggregate[:, -validation_windows:]
        train_target_codes = target_codes[:, :-validation_windows]
        validation_target_codes = target_codes[:, -validation_windows:]
    else:
        train_aggregate = validation_aggregate = aggregate
        train_target_codes = validation_target_codes = target_codes
    predicted = nonnegative_sparse_code(
        dictionary,
        train_aggregate,
        sparsity_coefficient=sparsity_coefficient,
        max_iterations=sparse_code_iterations,
        tolerance=tolerance,
    )
    validation_prediction = predicted
    if validation_windows:
        validation_prediction = nonnegative_sparse_code(
            dictionary,
            validation_aggregate,
            sparsity_coefficient=sparsity_coefficient,
            max_iterations=sparse_code_iterations,
            tolerance=tolerance,
        )
    best_error = float(
        torch.mean(torch.abs(validation_prediction.codes - validation_target_codes))
    )
    best_dictionary = dictionary.clone()

    for _ in range(discriminative_iterations):
        predicted_codes = predicted.codes
        candidate = _discriminative_dictionary_step(
            train_aggregate,
            dictionary,
            predicted_codes,
            train_target_codes,
            learning_rate=learning_rate,
            fallback=fallback,
        )
        if candidate is None:
            break
        change = torch.linalg.vector_norm(candidate - dictionary)
        scale = 1.0 + torch.linalg.vector_norm(dictionary)
        dictionary = candidate
        predicted = nonnegative_sparse_code(
            dictionary,
            train_aggregate,
            sparsity_coefficient=sparsity_coefficient,
            max_iterations=sparse_code_iterations,
            tolerance=tolerance,
        )
        validation_prediction = predicted
        if validation_windows:
            validation_prediction = nonnegative_sparse_code(
                dictionary,
                validation_aggregate,
                sparsity_coefficient=sparsity_coefficient,
                max_iterations=sparse_code_iterations,
                tolerance=tolerance,
            )
        activation_error = float(
            torch.mean(torch.abs(validation_prediction.codes - validation_target_codes))
        )
        if activation_error < best_error:
            best_error = activation_error
            best_dictionary = dictionary.clone()
        if float(change) <= tolerance * float(scale):
            break

    return best_dictionary, best_error


def _discriminative_dictionary_step(
    aggregate: torch.Tensor,
    dictionary: torch.Tensor,
    predicted_codes: torch.Tensor,
    target_codes: torch.Tensor,
    *,
    learning_rate: float,
    fallback: torch.Tensor,
) -> torch.Tensor | None:
    """Apply the DSC ``T1 - T2`` basis update used by the legacy implementation."""
    predicted_residual = aggregate - dictionary @ predicted_codes
    target_residual = aggregate - dictionary @ target_codes
    gradient = predicted_residual @ predicted_codes.T - target_residual @ target_codes.T
    if float(torch.linalg.vector_norm(gradient)) <= torch.finfo(dictionary.dtype).eps:
        return None
    return _normalize_dictionary(dictionary - learning_rate * gradient, fallback)


def _as_power_tensor(frame, label, *, device, allow_empty=False) -> torch.Tensor:
    values = power_vector(frame, label, allow_empty=allow_empty)
    if bool((values < 0).any()):
        raise ValueError(f"{label} must be non-negative.")
    return torch.as_tensor(values, dtype=torch.float32, device=device).clone()


def _window_tensor(frames, *, shape, label, device) -> torch.Tensor:
    windows = []
    for index, frame in enumerate(frames):
        values = _as_power_tensor(frame, f"{label} chunk {index}", device=device)
        padding = (-values.numel()) % shape
        if padding:
            values = torch.nn.functional.pad(values, (0, padding))
        windows.append(values.reshape(-1, shape).T)
    if not windows:
        raise ValueError(f"{label} requires at least one chunk.")
    return torch.cat(windows, dim=1)


def _aligned_training_windows(main_frames, train_appliances, *, shape, device):
    mains = _window_tensor(main_frames, shape=shape, label="mains", device=device)
    aligned = OrderedDict()
    for entry in train_appliances:
        try:
            appliance_name, target_frames = entry
        except (TypeError, ValueError) as exc:
            raise ValueError("Each appliance must be a (name, frames) pair.") from exc
        if not isinstance(appliance_name, str) or not appliance_name.strip():
            raise ValueError("Appliance names must be non-empty strings.")
        if appliance_name != appliance_name.strip():
            raise ValueError("Appliance names must not have surrounding whitespace.")
        if appliance_name in aligned:
            raise ValueError(f"Duplicate appliance name {appliance_name!r}.")
        if isinstance(target_frames, (str, bytes)) or not isinstance(
            target_frames, Sequence
        ):
            raise TypeError(
                f"Training frames for {appliance_name!r} must be a sequence."
            )
        target_frames = list(target_frames)
        if len(target_frames) != len(main_frames):
            raise ValueError(
                f"{appliance_name!r} has {len(target_frames)} chunks but mains has "
                f"{len(main_frames)}."
            )
        for index, (main_frame, target_frame) in enumerate(
            zip(main_frames, target_frames)
        ):
            main_values = power_vector(main_frame, f"mains chunk {index}")
            target_values = power_vector(
                target_frame, f"{appliance_name!r} target chunk {index}"
            )
            if len(main_values) != len(target_values):
                raise ValueError(
                    f"{appliance_name!r} target chunk {index} length does not "
                    "match mains."
                )
            main_index = getattr(main_frame, "index", None)
            target_index = getattr(target_frame, "index", None)
            if (
                main_index is not None
                and target_index is not None
                and not main_index.equals(target_index)
            ):
                raise ValueError(
                    f"{appliance_name!r} target chunk {index} index does not "
                    "match mains."
                )
        aligned[appliance_name] = _window_tensor(
            target_frames,
            shape=shape,
            label=f"{appliance_name!r} target",
            device=device,
        )
    if not aligned:
        raise ValueError("Training requires at least one appliance.")
    return mains, aligned


def _matrix_tuple(values: torch.Tensor) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in row)
        for row in values.detach().to(device="cpu", dtype=torch.float64)
    )


def _validate_dictionary_payload(name, values, *, shape, n_components):
    if not isinstance(values, (list, tuple)) or len(values) != shape:
        raise ValueError(f"{name} must contain exactly {shape} rows.")
    rows = []
    for row in values:
        if not isinstance(row, (list, tuple)) or len(row) != n_components:
            raise ValueError(f"{name} rows must contain exactly {n_components} values.")
        rows.append(
            tuple(_non_negative_finite(f"{name} value", value) for value in row)
        )
    matrix = torch.tensor(rows, dtype=torch.float64)
    norms = torch.linalg.vector_norm(matrix, dim=0)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
        raise ValueError(f"{name} columns must have unit norm.")
    return tuple(rows)


def _validate_fitted_model(model, *, shape, n_components) -> None:
    if not isinstance(model, _DSCParameters):
        raise TypeError("DSC model values must be fitted DSC parameters.")
    _validate_dictionary_payload(
        "reconstruction_dictionary",
        model.reconstruction_dictionary,
        shape=shape,
        n_components=n_components,
    )
    _validate_dictionary_payload(
        "discriminative_dictionary",
        model.discriminative_dictionary,
        shape=shape,
        n_components=n_components,
    )
    _positive_integer("training_windows", model.training_windows)
    _non_negative_finite("reconstruction_objective", model.reconstruction_objective)
    _positive_integer("reconstruction_iterations", model.reconstruction_iterations)
    if not isinstance(model.reconstruction_converged, bool):
        raise ValueError("reconstruction_converged must be a boolean.")
    _non_negative_finite("activation_error", model.activation_error)


def _parameters_from_payload(payload, *, shape, n_components):
    if not isinstance(payload, Mapping) or set(payload) != _PARAMETER_FIELDS:
        raise ValueError("DSC artifact has invalid fitted-parameter fields.")
    try:
        parameters = _DSCParameters(
            reconstruction_dictionary=tuple(
                tuple(row) for row in payload["reconstruction_dictionary"]
            ),
            discriminative_dictionary=tuple(
                tuple(row) for row in payload["discriminative_dictionary"]
            ),
            training_windows=payload["training_windows"],
            reconstruction_objective=payload["reconstruction_objective"],
            reconstruction_iterations=payload["reconstruction_iterations"],
            reconstruction_converged=payload["reconstruction_converged"],
            activation_error=payload["activation_error"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("DSC artifact fitted parameters are malformed.") from exc
    _validate_fitted_model(parameters, shape=shape, n_components=n_components)
    return parameters


def _reject_json_constant(value):
    raise ValueError(f"Invalid JSON constant {value!r}.")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


class DSC(TorchStateSpaceDisaggregator):
    """Non-negative discriminative sparse coding with Torch-only optimization."""

    MODEL_NAME = "DSC"

    def __init__(self, params=None):
        super().__init__(params)
        params = {} if params is None else params
        self.shape = _positive_integer("shape", get_param(params, "shape", 120))
        self.n_components = _positive_integer(
            "n_components", get_param(params, "n_components", 10)
        )
        self.sparsity_coefficient = _non_negative_finite(
            "sparsity_coefficient",
            get_param(
                params,
                "sparsity_coefficient",
                20.0,
                aliases=("sparsity_coef",),
            ),
        )
        self.dictionary_iterations = _non_negative_integer(
            "dictionary_iterations", get_param(params, "dictionary_iterations", 20)
        )
        self.discriminative_iterations = _non_negative_integer(
            "discriminative_iterations",
            get_param(
                params,
                "discriminative_iterations",
                20,
                aliases=("iterations",),
            ),
        )
        self.sparse_code_iterations = _positive_integer(
            "sparse_code_iterations",
            get_param(params, "sparse_code_iterations", 100),
        )
        self.tolerance = _positive_finite(
            "tolerance", get_param(params, "tolerance", 1e-5)
        )
        self.discriminative_learning_rate = _positive_finite(
            "discriminative_learning_rate",
            get_param(
                params,
                "discriminative_learning_rate",
                1e-9,
                aliases=("learning_rate",),
            ),
        )
        self.enforce_aggregate = get_param(params, "enforce_aggregate", True)
        if not isinstance(self.enforce_aggregate, bool):
            raise ValueError("enforce_aggregate must be a boolean.")
        if self.device.type == "mps":
            raise ValueError("DSC currently supports CPU and CUDA.")
        if self.chunk_wise_training:
            raise ValueError("DSC does not support chunk_wise_training.")
        if self.load_model_path:
            self.load_model(self.load_model_path)

    @property
    def iterations(self):
        """Legacy name for the discriminative update count."""
        return self.discriminative_iterations

    @iterations.setter
    def iterations(self, value):
        self.discriminative_iterations = _non_negative_integer("iterations", value)

    @property
    def learning_rate(self):
        """Legacy name for the discriminative update step size."""
        return self.discriminative_learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self.discriminative_learning_rate = _positive_finite("learning_rate", value)

    @property
    def sparsity_coef(self):
        """Legacy name for the non-negative LASSO coefficient."""
        return self.sparsity_coefficient

    @sparsity_coef.setter
    def sparsity_coef(self, value):
        self.sparsity_coefficient = _non_negative_finite("sparsity_coef", value)

    def _validate_model_record(self, appliance_name, model):
        del appliance_name
        _validate_fitted_model(
            model,
            shape=self.shape,
            n_components=self.n_components,
        )

    @torch.no_grad()
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
        try:
            main_frames = list(train_main)
        except TypeError as exc:
            raise TypeError("train_main must contain mains chunks.") from exc
        if not main_frames:
            raise ValueError("Training requires at least one mains chunk.")
        try:
            appliance_entries = list(train_appliances)
        except TypeError as exc:
            raise TypeError(
                "train_appliances must contain (name, frames) pairs."
            ) from exc
        mains, targets = _aligned_training_windows(
            main_frames,
            appliance_entries,
            shape=self.shape,
            device=self.device,
        )

        reconstruction_dictionaries = OrderedDict()
        target_codes = OrderedDict()
        reconstruction_results = OrderedDict()
        for appliance_name, observations in targets.items():
            dictionary, code_result = _learn_dictionary(
                observations,
                n_components=self.n_components,
                sparsity_coefficient=self.sparsity_coefficient,
                dictionary_iterations=self.dictionary_iterations,
                sparse_code_iterations=self.sparse_code_iterations,
                tolerance=self.tolerance,
            )
            reconstruction_dictionaries[appliance_name] = dictionary
            target_codes[appliance_name] = code_result.codes
            reconstruction_results[appliance_name] = code_result

        joint_reconstruction = torch.cat(
            tuple(reconstruction_dictionaries.values()), dim=1
        )
        joint_target_codes = torch.cat(tuple(target_codes.values()), dim=0)
        joint_discriminative, activation_error = _fit_discriminative_dictionary(
            mains,
            joint_reconstruction,
            joint_target_codes,
            sparsity_coefficient=self.sparsity_coefficient,
            discriminative_iterations=self.discriminative_iterations,
            sparse_code_iterations=self.sparse_code_iterations,
            tolerance=self.tolerance,
            learning_rate=self.discriminative_learning_rate,
        )

        fitted = OrderedDict()
        start = 0
        for appliance_name, reconstruction in reconstruction_dictionaries.items():
            stop = start + self.n_components
            record = _DSCParameters(
                reconstruction_dictionary=_matrix_tuple(reconstruction),
                discriminative_dictionary=_matrix_tuple(
                    joint_discriminative[:, start:stop]
                ),
                training_windows=int(mains.shape[1]),
                reconstruction_objective=reconstruction_results[
                    appliance_name
                ].objective,
                reconstruction_iterations=reconstruction_results[
                    appliance_name
                ].iterations,
                reconstruction_converged=reconstruction_results[
                    appliance_name
                ].converged,
                activation_error=activation_error,
            )
            _validate_fitted_model(
                record,
                shape=self.shape,
                n_components=self.n_components,
            )
            fitted[appliance_name] = record
            start = stop
        self.models = fitted
        if self.save_model_path:
            self.save_model(self.save_model_path)

    @torch.no_grad()
    def disaggregate_chunk(
        self,
        test_main_list,
        model=None,
        do_preprocessing=True,
    ):
        if not isinstance(do_preprocessing, bool):
            raise ValueError("do_preprocessing must be a boolean.")
        models = self.require_models(model)
        reconstruction = torch.cat(
            tuple(
                torch.tensor(
                    fitted.reconstruction_dictionary,
                    dtype=torch.float32,
                    device=self.device,
                )
                for fitted in models.values()
            ),
            dim=1,
        )
        discriminative = torch.cat(
            tuple(
                torch.tensor(
                    fitted.discriminative_dictionary,
                    dtype=torch.float32,
                    device=self.device,
                )
                for fitted in models.values()
            ),
            dim=1,
        )
        predictions = []
        for chunk_index, frame in enumerate(test_main_list):
            values = _as_power_tensor(
                frame,
                f"mains chunk {chunk_index}",
                device=self.device,
                allow_empty=True,
            )
            frame_index = getattr(frame, "index", pd.RangeIndex(values.numel()))
            if values.numel() == 0:
                predictions.append(
                    pd.DataFrame(
                        {
                            appliance_name: pd.Series(
                                index=frame_index, dtype=np.float32
                            )
                            for appliance_name in models
                        },
                        index=frame_index,
                    )
                )
                continue
            original_length = values.numel()
            padding = (-original_length) % self.shape
            padded = (
                torch.nn.functional.pad(values, (0, padding)) if padding else values
            )
            windows = padded.reshape(-1, self.shape).T
            codes = nonnegative_sparse_code(
                discriminative,
                windows,
                sparsity_coefficient=self.sparsity_coefficient,
                max_iterations=self.sparse_code_iterations,
                tolerance=self.tolerance,
            ).codes
            appliance_predictions = []
            start = 0
            for _appliance_name in models:
                stop = start + self.n_components
                reconstructed = reconstruction[:, start:stop] @ codes[start:stop]
                appliance_predictions.append(
                    reconstructed.T.flatten()[:original_length]
                )
                start = stop
            stacked = torch.stack(appliance_predictions)
            stacked.clamp_(min=0)
            if self.enforce_aggregate:
                total = stacked.sum(dim=0)
                scale = torch.where(
                    total > values,
                    values / total.clamp_min(torch.finfo(total.dtype).eps),
                    torch.ones_like(total),
                )
                stacked *= scale
            result = {
                appliance_name: stacked[index]
                .to(device="cpu", dtype=torch.float32)
                .numpy()
                for index, appliance_name in enumerate(models)
            }
            predictions.append(
                pd.DataFrame(result, index=frame_index, dtype=np.float32)
            )
        return predictions

    def save_model(self, path=None):
        target = path if path is not None else self.save_model_path
        if target is None:
            raise ValueError("DSC save_model requires a checkpoint directory.")
        models = self.require_models()
        payload = {
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "model_class": self.MODEL_NAME,
            "config": {name: getattr(self, name) for name in _CONFIG_FIELDS},
            "models": {
                appliance_name: asdict(fitted)
                for appliance_name, fitted in models.items()
            },
        }
        save_json_atomic(Path(target) / _ARTIFACT_FILENAME, payload)

    def load_model(self, path=None):
        source = path if path is not None else self.load_model_path
        if source is None:
            raise ValueError("DSC load_model requires a checkpoint directory.")
        artifact_path = Path(source) / _ARTIFACT_FILENAME
        try:
            if artifact_path.stat().st_size > _MAX_ARTIFACT_BYTES:
                raise ValueError("DSC artifact exceeds the safety limit.")
            with artifact_path.open(encoding="utf-8") as handle:
                payload = json.load(
                    handle,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_unique_json_object,
                )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Could not load a valid DSC artifact: {exc}") from exc
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "model_class",
            "config",
            "models",
        }:
            raise ValueError("DSC artifact has invalid top-level fields.")
        if (
            isinstance(payload["schema_version"], bool)
            or not isinstance(payload["schema_version"], int)
            or payload["schema_version"] != _ARTIFACT_SCHEMA_VERSION
        ):
            raise ValueError("DSC artifact has an unsupported schema_version.")
        if payload["model_class"] != self.MODEL_NAME:
            raise ValueError("DSC artifact model_class does not match DSC.")
        config = payload["config"]
        if not isinstance(config, Mapping) or set(config) != set(_CONFIG_FIELDS):
            raise ValueError("DSC artifact has invalid configuration fields.")
        shape = _positive_integer("shape", config["shape"])
        n_components = _positive_integer("n_components", config["n_components"])
        validated_config = {
            "shape": shape,
            "n_components": n_components,
            "sparsity_coefficient": _non_negative_finite(
                "sparsity_coefficient", config["sparsity_coefficient"]
            ),
            "dictionary_iterations": _non_negative_integer(
                "dictionary_iterations", config["dictionary_iterations"]
            ),
            "discriminative_iterations": _non_negative_integer(
                "discriminative_iterations", config["discriminative_iterations"]
            ),
            "sparse_code_iterations": _positive_integer(
                "sparse_code_iterations", config["sparse_code_iterations"]
            ),
            "tolerance": _positive_finite("tolerance", config["tolerance"]),
            "discriminative_learning_rate": _positive_finite(
                "discriminative_learning_rate",
                config["discriminative_learning_rate"],
            ),
            "enforce_aggregate": config["enforce_aggregate"],
        }
        if not isinstance(validated_config["enforce_aggregate"], bool):
            raise ValueError("enforce_aggregate must be a boolean.")
        model_payloads = payload["models"]
        if not isinstance(model_payloads, Mapping) or not model_payloads:
            raise ValueError("DSC artifact must contain at least one appliance model.")
        loaded = OrderedDict()
        for appliance_name, fitted_payload in model_payloads.items():
            if (
                not isinstance(appliance_name, str)
                or not appliance_name.strip()
                or appliance_name != appliance_name.strip()
                or appliance_name in loaded
            ):
                raise ValueError("DSC artifact has an invalid appliance name.")
            loaded[appliance_name] = _parameters_from_payload(
                fitted_payload,
                shape=shape,
                n_components=n_components,
            )
        for name, value in validated_config.items():
            setattr(self, name, value)
        self.models = loaded


__all__ = ["DSC", "SparseCodeResult", "nonnegative_sparse_code"]
