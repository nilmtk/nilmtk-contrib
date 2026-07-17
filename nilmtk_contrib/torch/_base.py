"""Small, shared building blocks for PyTorch disaggregators.

This module owns runtime concerns that should not be reimplemented by every
algorithm. Architectures, preprocessing, losses, and training loops remain in
the model modules where their scientific behavior is visible.
"""

from collections import OrderedDict
from collections.abc import Iterable, Mapping
import math
from numbers import Integral, Real

import numpy as np
import torch
from nilmtk.disaggregate import Disaggregator

from nilmtk_contrib.utils.model import initialize_runtime
from nilmtk_contrib.utils.params import normalize_common_params


SUPPORTED_DEVICE_TYPES = frozenset({"cpu", "cuda", "mps"})


def resolve_torch_device(requested=None):
    """Resolve and validate a CPU, CUDA, or MPS device.

    Automatic selection deliberately preserves contrib's established policy:
    use CUDA when available and otherwise use CPU. MPS must be requested
    explicitly so migrating a model cannot silently change its numerics.
    """
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(requested, bool) or not isinstance(requested, (str, torch.device)):
        raise ValueError("device must be a non-empty string, torch.device, or None.")
    if isinstance(requested, str):
        requested = requested.strip()
        if not requested:
            raise ValueError(
                "device must be a non-empty string, torch.device, or None."
            )
    try:
        device = torch.device(requested)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid torch device {requested!r}.") from exc

    if device.type not in SUPPORTED_DEVICE_TYPES:
        supported = ", ".join(sorted(SUPPORTED_DEVICE_TYPES))
        raise ValueError(
            f"Unsupported torch device type {device.type!r}; choose {supported}."
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} is unavailable; "
                f"visible device count is {torch.cuda.device_count()}."
            )
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
    return device


def compute_appliance_stats(
    train_appliances: Iterable,
    *,
    std_floor: float = 1.0,
    std_fallback: float = 100.0,
    include_extrema: bool = False,
):
    """Compute finite normalization statistics for appliance frames.

    ``std_fallback`` preserves the historical contrib convention for nearly
    constant targets while making that policy explicit and configurable.
    """
    for name, value in (("std_floor", std_floor), ("std_fallback", std_fallback)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive finite number.")
    if not isinstance(include_extrema, bool):
        raise ValueError("include_extrema must be a boolean.")

    if train_appliances is None:
        raise ValueError("At least one appliance target is required.")

    statistics = {}
    try:
        appliance_entries = iter(train_appliances)
    except TypeError as exc:
        raise TypeError(
            "train_appliances must be an iterable of (name, frames) pairs."
        ) from exc

    for entry in appliance_entries:
        try:
            appliance_name, frames = entry
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Each appliance target must be a (name, frames) pair."
            ) from exc
        if not isinstance(appliance_name, str) or not appliance_name.strip():
            raise ValueError("Appliance names must be non-empty strings.")
        if appliance_name in statistics:
            raise ValueError(f"Duplicate appliance name {appliance_name!r}.")
        try:
            frames = list(frames)
        except TypeError as exc:
            raise TypeError(
                f"Training frames for {appliance_name!r} must be iterable."
            ) from exc
        if not frames:
            raise ValueError(
                f"Training data for {appliance_name!r} contains no frames."
            )
        sample_count = 0
        mean = 0.0
        sum_squared_deviations = 0.0
        minimum = math.inf
        maximum = -math.inf
        for frame in frames:
            if hasattr(frame, "to_numpy"):
                values = frame.to_numpy(dtype=np.float64)
            else:
                values = np.asarray(frame, dtype=np.float64)
            values = values.reshape(-1)
            if values.size == 0:
                continue
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Training data for {appliance_name!r} must be finite."
                )

            frame_count = values.size
            with np.errstate(over="ignore", invalid="ignore"):
                frame_mean = float(np.mean(values))
                frame_variance = float(np.var(values))
                combined_count = sample_count + frame_count
                delta = frame_mean - mean
                mean += delta * frame_count / combined_count
                sum_squared_deviations += frame_variance * frame_count
                if sample_count:
                    sum_squared_deviations += (
                        delta * delta * sample_count * frame_count / combined_count
                    )
            sample_count = combined_count
            minimum = min(minimum, float(np.min(values)))
            maximum = max(maximum, float(np.max(values)))

        if sample_count == 0:
            raise ValueError(
                f"Training data for {appliance_name!r} contains no samples."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            std = math.sqrt(sum_squared_deviations / sample_count)
        if not math.isfinite(mean) or not math.isfinite(std):
            raise ValueError(
                f"Normalization statistics for {appliance_name!r} overflowed."
            )
        result = {
            "mean": mean,
            "std": float(std_fallback) if std < std_floor else std,
        }
        if include_extrema:
            result.update(max=maximum, min=minimum)
        statistics[appliance_name] = result
    if not statistics:
        raise ValueError("At least one appliance target is required.")
    return statistics


class TorchDisaggregator(Disaggregator):
    """Shared runtime state for one-model-per-appliance disaggregators.

    Subclasses still implement their architecture, preprocessing, training,
    inference, and persistence. The base only centralizes configuration that
    otherwise drifts between model implementations.
    """

    APPLIANCE_STD_FLOOR = 1.0
    APPLIANCE_STD_FALLBACK = 100.0
    INCLUDE_APPLIANCE_EXTREMA = False

    def __init__(self, params=None, *, defaults):
        if params is None:
            params = {}
        if not isinstance(params, Mapping):
            raise TypeError("params must be a mapping or None.")
        if not isinstance(defaults, Mapping):
            raise TypeError("defaults must be a mapping.")

        supplied_appliance_params = params.get(
            "appliance_params", defaults.get("appliance_params", {})
        )
        if not isinstance(supplied_appliance_params, Mapping):
            raise TypeError("appliance_params must be a mapping.")

        common = normalize_common_params(dict(params), dict(defaults))
        if (
            isinstance(common.mains_mean, bool)
            or not isinstance(common.mains_mean, Real)
            or not math.isfinite(common.mains_mean)
        ):
            raise ValueError("mains_mean must be a finite number.")
        if (
            isinstance(common.mains_std, bool)
            or not isinstance(common.mains_std, Real)
            or not math.isfinite(common.mains_std)
            or common.mains_std <= 0
        ):
            raise ValueError("mains_std must be a positive finite number.")
        if common.seed is not None and (
            isinstance(common.seed, bool) or not isinstance(common.seed, Integral)
        ):
            raise ValueError("seed must be an integer or None.")
        if not isinstance(common.verbose, bool):
            raise ValueError("verbose must be a boolean.")
        if not isinstance(common.chunk_wise_training, bool):
            raise ValueError("chunk_wise_training must be a boolean.")

        super().__init__()
        initialize_runtime(
            self,
            {"seed": common.seed, "verbose": common.verbose},
            backends=("python", "numpy", "torch"),
        )
        self.sequence_length = common.sequence_length
        self.n_epochs = common.n_epochs
        self.batch_size = common.batch_size
        self.mains_mean = common.mains_mean
        self.mains_std = common.mains_std
        self.appliance_params = dict(common.appliance_params)
        self.save_model_path = common.save_model_path
        self.load_model_path = common.pretrained_model_path
        self.chunk_wise_training = common.chunk_wise_training
        self.models = OrderedDict()
        self.device = resolve_torch_device(common.device)

    def set_appliance_params(self, train_appliances):
        """Update target normalization using the class's explicit policy."""
        statistics = compute_appliance_stats(
            train_appliances,
            std_floor=self.APPLIANCE_STD_FLOOR,
            std_fallback=self.APPLIANCE_STD_FALLBACK,
            include_extrema=self.INCLUDE_APPLIANCE_EXTREMA,
        )
        self.appliance_params.update(statistics)

    def require_models(self, models=None):
        """Optionally install and then return a validated model mapping."""
        if models is not None:
            if not isinstance(models, Mapping):
                raise TypeError(
                    "models must be a mapping of appliance names to modules."
                )
            validated = OrderedDict()
            for appliance_name, model in models.items():
                if not isinstance(appliance_name, str) or not appliance_name.strip():
                    raise ValueError("Model appliance names must be non-empty strings.")
                if not isinstance(model, torch.nn.Module):
                    raise TypeError(
                        f"Model for {appliance_name!r} must be a torch.nn.Module."
                    )
                validated[appliance_name] = model
            self.models = validated
        if not self.models:
            model_name = getattr(self, "MODEL_NAME", self.__class__.__name__)
            raise RuntimeError(f"{model_name} requires a trained or loaded model.")
        return self.models
