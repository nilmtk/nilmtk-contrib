"""Supervised solver-free additive factorial HMM disaggregator."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from itertools import pairwise
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nilmtk_contrib.torch._base import TorchStateSpaceDisaggregator
from nilmtk_contrib.torch._coordinate_factorial_hmm import (
    factorial_hmm_coordinate_viterbi,
)
from nilmtk_contrib.torch._data import power_vector
from nilmtk_contrib.torch._hmm import canonicalize_gaussian_hmm
from nilmtk_contrib.torch._hmm_fit import fit_observed_gaussian_hmm
from nilmtk_contrib.torch._state_space_data import (
    aligned_power_windows,
    frame_index,
)
from nilmtk_contrib.utils.checkpoints import load_json_strict, save_json_atomic
from nilmtk_contrib.utils.params import get_param, validate_positive_int


@dataclass(frozen=True)
class _AFHMMParameters:
    state_means: tuple[float, ...]
    initial_probabilities: tuple[float, ...]
    transition_probabilities: tuple[tuple[float, ...], ...]
    background_mean: float
    fit_loss_history: tuple[float, ...]
    fit_iterations: int
    fit_converged: bool
    num_samples: int
    num_chunks: int


@dataclass(frozen=True)
class TorchAFHMMInferenceDiagnostics:
    """Convergence details for one disaggregated mains chunk."""

    samples: int
    iterations: int
    converged: bool
    score: float | None
    score_history: tuple[float, ...]


_ARTIFACT_FILENAME = "torch_afhmm.json"
_ARTIFACT_SCHEMA_VERSION = 2
_CONFIG_FIELDS = (
    "num_states",
    "pseudocount",
    "kmeans_max_iterations",
    "kmeans_tolerance",
    "noise_std",
    "inference_max_iterations",
    "fail_on_nonconvergence",
)
_PARAMETER_FIELDS = {field.name for field in fields(_AFHMMParameters)}


def _finite_number(name, value, *, positive):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {qualifier} number.")
    return float(value)


def _integer(name, value, *, minimum=1):
    result = validate_positive_int(name, value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _validate_fitted_model(model, *, num_states):
    if not isinstance(model, _AFHMMParameters):
        raise TypeError("TorchAFHMM model values must be fitted HMM parameters.")
    if len(model.state_means) != num_states:
        raise ValueError(f"state_means must contain exactly {num_states} values.")
    if len(model.initial_probabilities) != num_states:
        raise ValueError(
            f"initial_probabilities must contain exactly {num_states} values."
        )
    if len(model.transition_probabilities) != num_states or any(
        len(row) != num_states for row in model.transition_probabilities
    ):
        raise ValueError(
            f"transition_probabilities must have shape ({num_states}, {num_states})."
        )
    try:
        means = torch.tensor(model.state_means, dtype=torch.float64)
        initial = torch.tensor(model.initial_probabilities, dtype=torch.float64)
        transition = torch.tensor(model.transition_probabilities, dtype=torch.float64)
        if not bool((initial > 0).all()) or not bool((transition > 0).all()):
            raise ValueError("TorchAFHMM probabilities must be strictly positive.")
        canonical = canonicalize_gaussian_hmm(means, initial, transition)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Invalid TorchAFHMM HMM parameters: {exc}") from exc
    if bool((canonical.state_means < 0).any()) or any(
        left >= right for left, right in zip(model.state_means, model.state_means[1:])
    ):
        raise ValueError("state_means must be finite, nonnegative, and increasing.")
    _finite_number("background_mean", model.background_mean, positive=False)
    if (
        not isinstance(model.fit_converged, bool)
        or isinstance(model.fit_iterations, bool)
        or not isinstance(model.fit_iterations, Integral)
        or model.fit_iterations < 1
    ):
        raise ValueError("TorchAFHMM fit diagnostics are invalid.")
    if len(model.fit_loss_history) != model.fit_iterations + 1 or any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
        for value in model.fit_loss_history
    ):
        raise ValueError("fit_loss_history is inconsistent with fit_iterations.")
    tolerance = 1e-9
    if any(
        right > left + tolerance * max(1.0, abs(left))
        for left, right in pairwise(model.fit_loss_history)
    ):
        raise ValueError("fit_loss_history must be monotonically nonincreasing.")
    for name, value in (
        ("num_samples", model.num_samples),
        ("num_chunks", model.num_chunks),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if model.num_samples < model.num_chunks:
        raise ValueError("num_samples must be at least num_chunks.")


def _parameters_from_payload(payload, *, num_states):
    if not isinstance(payload, Mapping) or set(payload) != _PARAMETER_FIELDS:
        raise ValueError("TorchAFHMM artifact has invalid fitted-parameter fields.")
    try:
        parameters = _AFHMMParameters(
            state_means=tuple(payload["state_means"]),
            initial_probabilities=tuple(payload["initial_probabilities"]),
            transition_probabilities=tuple(
                tuple(row) for row in payload["transition_probabilities"]
            ),
            background_mean=payload["background_mean"],
            fit_loss_history=tuple(payload["fit_loss_history"]),
            fit_iterations=payload["fit_iterations"],
            fit_converged=payload["fit_converged"],
            num_samples=payload["num_samples"],
            num_chunks=payload["num_chunks"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("TorchAFHMM fitted parameters are malformed.") from exc
    _validate_fitted_model(parameters, num_states=num_states)
    return parameters


def _fitted_record(fit, *, background_mean, num_samples, num_chunks):
    parameters = fit.parameters
    record = _AFHMMParameters(
        state_means=tuple(float(value) for value in parameters.state_means),
        initial_probabilities=tuple(
            float(value) for value in parameters.initial_probabilities
        ),
        transition_probabilities=tuple(
            tuple(float(value) for value in row)
            for row in parameters.transition_probabilities
        ),
        background_mean=background_mean,
        fit_loss_history=fit.loss_history,
        fit_iterations=fit.iterations,
        fit_converged=fit.converged,
        num_samples=num_samples,
        num_chunks=num_chunks,
    )
    _validate_fitted_model(record, num_states=len(record.state_means))
    return record


def _background_mean(mains, training_frames):
    appliance_names = tuple(sorted(training_frames))
    residuals = []
    for chunk_index, main in enumerate(mains):
        modeled = torch.stack(
            [training_frames[name][chunk_index] for name in appliance_names]
        ).sum(dim=0)
        residuals.append(main - modeled)
    mean = float(torch.cat(residuals).mean())
    if not math.isfinite(mean):
        raise ValueError("Training residual background mean must be finite.")
    return max(0.0, mean)


class TorchAFHMM(TorchStateSpaceDisaggregator):
    """Additive FHMM with observed-state fitting and coordinate inference."""

    MODEL_NAME = "TorchAFHMM"

    def __init__(self, params=None):
        super().__init__(params)
        params = {} if params is None else params
        self.num_states = _integer(
            "num_states", get_param(params, "num_states", 2), minimum=2
        )
        self.pseudocount = _finite_number(
            "pseudocount", get_param(params, "pseudocount", 1.0), positive=True
        )
        self.kmeans_max_iterations = _integer(
            "kmeans_max_iterations",
            get_param(params, "kmeans_max_iterations", 100),
        )
        self.kmeans_tolerance = _finite_number(
            "kmeans_tolerance",
            get_param(params, "kmeans_tolerance", 1e-6),
            positive=False,
        )
        self.noise_std = _finite_number(
            "noise_std", get_param(params, "noise_std", 100.0), positive=True
        )
        self.inference_max_iterations = _integer(
            "inference_max_iterations",
            get_param(params, "inference_max_iterations", 20),
        )
        self.fail_on_nonconvergence = get_param(params, "fail_on_nonconvergence", False)
        if not isinstance(self.fail_on_nonconvergence, bool):
            raise ValueError("fail_on_nonconvergence must be a boolean.")
        if self.device.type == "mps":
            raise ValueError(
                "TorchAFHMM supports CPU and CUDA; float64 MPS is unsupported."
            )
        if self.chunk_wise_training:
            raise ValueError("TorchAFHMM does not support chunk_wise_training.")
        self.last_inference_diagnostics = ()
        if self.load_model_path:
            self.load_model(self.load_model_path)

    def _validate_model_record(self, appliance_name, model):
        del appliance_name
        _validate_fitted_model(model, num_states=self.num_states)

    def save_model(self, path=None):
        target = path if path is not None else self.save_model_path
        if target is None:
            raise ValueError("TorchAFHMM save_model requires a checkpoint directory.")
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
            raise ValueError("TorchAFHMM load_model requires a checkpoint directory.")
        payload = load_json_strict(
            Path(source) / _ARTIFACT_FILENAME,
            description="TorchAFHMM artifact",
        )
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "model_class",
            "config",
            "models",
        }:
            raise ValueError("TorchAFHMM artifact has invalid top-level fields.")
        if (
            isinstance(payload["schema_version"], bool)
            or not isinstance(payload["schema_version"], Integral)
            or payload["schema_version"] != _ARTIFACT_SCHEMA_VERSION
        ):
            raise ValueError("TorchAFHMM artifact has an unsupported schema_version.")
        if payload["model_class"] != self.MODEL_NAME:
            raise ValueError("TorchAFHMM artifact model_class does not match.")
        config = payload["config"]
        if not isinstance(config, Mapping) or set(config) != set(_CONFIG_FIELDS):
            raise ValueError("TorchAFHMM artifact has invalid configuration fields.")
        num_states = _integer("num_states", config["num_states"], minimum=2)
        pseudocount = _finite_number(
            "pseudocount", config["pseudocount"], positive=True
        )
        kmeans_max_iterations = _integer(
            "kmeans_max_iterations", config["kmeans_max_iterations"]
        )
        kmeans_tolerance = _finite_number(
            "kmeans_tolerance", config["kmeans_tolerance"], positive=False
        )
        noise_std = _finite_number("noise_std", config["noise_std"], positive=True)
        inference_max_iterations = _integer(
            "inference_max_iterations", config["inference_max_iterations"]
        )
        fail_on_nonconvergence = config["fail_on_nonconvergence"]
        if not isinstance(fail_on_nonconvergence, bool):
            raise ValueError("fail_on_nonconvergence must be a boolean.")
        model_payloads = payload["models"]
        if not isinstance(model_payloads, Mapping) or not model_payloads:
            raise ValueError(
                "TorchAFHMM artifact must contain at least one appliance model."
            )
        loaded = OrderedDict()
        for appliance_name in sorted(model_payloads):
            if (
                not isinstance(appliance_name, str)
                or not appliance_name.strip()
                or appliance_name != appliance_name.strip()
            ):
                raise ValueError("TorchAFHMM artifact has an invalid appliance name.")
            loaded[appliance_name] = _parameters_from_payload(
                model_payloads[appliance_name], num_states=num_states
            )

        self.num_states = num_states
        self.pseudocount = pseudocount
        self.kmeans_max_iterations = kmeans_max_iterations
        self.kmeans_tolerance = kmeans_tolerance
        self.noise_std = noise_std
        self.inference_max_iterations = inference_max_iterations
        self.fail_on_nonconvergence = fail_on_nonconvergence
        self.models = loaded
        self.last_inference_diagnostics = ()

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

        training_frames = {}
        aligned_mains = None
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
            if appliance_name in training_frames:
                raise ValueError(f"Duplicate appliance name {appliance_name!r}.")
            if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
                raise TypeError(
                    f"Training frames for {appliance_name!r} must be a sequence."
                )
            mains, targets = aligned_power_windows(
                main_frames, list(frames), appliance_name
            )
            if aligned_mains is None:
                aligned_mains = mains
            training_frames[appliance_name] = targets

        background_mean = _background_mean(aligned_mains, training_frames)
        fitted = OrderedDict()
        for appliance_name in sorted(training_frames):
            targets = tuple(
                values.to(self.device) for values in training_frames[appliance_name]
            )
            fit = fit_observed_gaussian_hmm(
                targets,
                n_states=self.num_states,
                max_iterations=self.kmeans_max_iterations,
                tolerance=self.kmeans_tolerance,
                pseudocount=self.pseudocount,
            )
            fitted[appliance_name] = _fitted_record(
                fit,
                background_mean=background_mean,
                num_samples=sum(values.numel() for values in targets),
                num_chunks=len(targets),
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
        appliance_names = tuple(sorted(models))
        background_mean = models[appliance_names[0]].background_mean
        if any(
            not math.isclose(
                models[name].background_mean,
                background_mean,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            for name in appliance_names[1:]
        ):
            raise ValueError(
                "TorchAFHMM appliance models must share one background_mean."
            )
        state_means = tuple(
            torch.tensor(
                models[name].state_means, dtype=torch.float64, device=self.device
            )
            for name in appliance_names
        )
        initial = tuple(
            torch.tensor(
                models[name].initial_probabilities,
                dtype=torch.float64,
                device=self.device,
            )
            for name in appliance_names
        )
        transition = tuple(
            torch.tensor(
                models[name].transition_probabilities,
                dtype=torch.float64,
                device=self.device,
            )
            for name in appliance_names
        )
        results = []
        diagnostics = []
        for chunk_index, frame in enumerate(test_main_list):
            values = power_vector(frame, f"mains chunk {chunk_index}", allow_empty=True)
            if bool((values < 0).any()):
                raise ValueError(f"mains chunk {chunk_index} must be non-negative.")
            output_index = frame_index(frame, len(values))
            if not len(values):
                results.append(
                    pd.DataFrame(
                        {
                            name: pd.Series(
                                np.empty(0, dtype=np.float32), index=output_index
                            )
                            for name in appliance_names
                        },
                        index=output_index,
                    )
                )
                diagnostics.append(
                    TorchAFHMMInferenceDiagnostics(
                        samples=0,
                        iterations=0,
                        converged=True,
                        score=None,
                        score_history=(),
                    )
                )
                continue
            observations = (
                torch.as_tensor(values, dtype=torch.float64, device=self.device)
                - background_mean
            )
            decoded = factorial_hmm_coordinate_viterbi(
                observations,
                state_means,
                initial,
                transition,
                noise_std=self.noise_std,
                max_iterations=self.inference_max_iterations,
            )
            if self.fail_on_nonconvergence and not decoded.converged:
                raise RuntimeError(
                    f"TorchAFHMM did not converge for mains chunk {chunk_index} "
                    f"within {self.inference_max_iterations} iterations."
                )
            predictions = decoded.appliance_power.to(
                dtype=torch.float32, device="cpu"
            ).numpy()
            results.append(
                pd.DataFrame(predictions, index=output_index, columns=appliance_names)
            )
            diagnostics.append(
                TorchAFHMMInferenceDiagnostics(
                    samples=len(values),
                    iterations=decoded.iterations,
                    converged=decoded.converged,
                    score=decoded.score,
                    score_history=decoded.score_history,
                )
            )
        self.last_inference_diagnostics = tuple(diagnostics)
        return results


__all__ = ["TorchAFHMM", "TorchAFHMMInferenceDiagnostics"]
