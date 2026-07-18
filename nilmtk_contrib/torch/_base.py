"""Small, shared building blocks for PyTorch disaggregators.

This module owns runtime concerns that should not be reimplemented by every
algorithm. Architectures, preprocessing, losses, and training loops remain in
the model modules where their scientific behavior is visible.
"""

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from copy import deepcopy
import math
from numbers import Integral, Real

import numpy as np
import torch
from nilmtk.disaggregate import Disaggregator

from nilmtk_contrib.utils.model import initialize_runtime
from nilmtk_contrib.utils.params import (
    DEFAULT_ALIASES,
    get_param,
    normalize_common_params,
)


SUPPORTED_DEVICE_TYPES = frozenset({"cpu", "cuda", "mps"})
_COMMON_DEFAULTS = {
    "sequence_length": 99,
    "n_epochs": 10,
    "batch_size": 512,
    "mains_mean": 1800.0,
    "mains_std": 600.0,
    "appliance_params": {},
    "save_model_path": None,
    "pretrained_model_path": None,
    "chunk_wise_training": False,
    "seed": None,
    "verbose": False,
    "device": None,
}


def torch_defaults(**overrides):
    """Return isolated common defaults with model-specific overrides."""
    unknown = set(overrides).difference(_COMMON_DEFAULTS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown common default parameters: {names}.")
    defaults = deepcopy(_COMMON_DEFAULTS)
    defaults.update(overrides)
    return defaults


def _validate_appliance_name(name, *, label="Appliance"):
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{label} names must be non-empty strings.")
    if name != name.strip():
        raise ValueError(f"{label} names must not have surrounding whitespace.")
    return name


def _copy_appliance_params(appliance_params):
    """Validate and detach caller-owned appliance normalization state."""
    copied = {}
    for appliance_name, stats in appliance_params.items():
        _validate_appliance_name(appliance_name)
        if not isinstance(stats, Mapping):
            raise TypeError(f"appliance_params[{appliance_name!r}] must be a mapping.")
        stats = deepcopy(dict(stats))
        missing = {"mean", "std"}.difference(stats)
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(
                f"appliance_params[{appliance_name!r}] is missing {fields}."
            )
        for field in ("mean", "std", "min", "max"):
            if field not in stats:
                continue
            value = stats[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"appliance_params[{appliance_name!r}][{field!r}] "
                    "must be a finite number."
                )
        if stats["std"] <= 0:
            raise ValueError(
                f"appliance_params[{appliance_name!r}]['std'] must be positive."
            )
        if "min" in stats and "max" in stats and stats["min"] > stats["max"]:
            raise ValueError(
                f"appliance_params[{appliance_name!r}]['min'] must not exceed max."
            )
        copied[appliance_name] = stats
    return copied


def resolve_torch_device(requested=None):
    """Resolve and validate a CPU, CUDA, or MPS device.

    Automatic selection deliberately preserves contrib's established policy:
    use CUDA when available and otherwise use CPU. MPS must be requested
    explicitly so migrating a model cannot silently change its numerics.
    This is runtime placement, not checkpoint metadata; loaders should map
    saved weights onto this resolved device.
    """
    cuda_available = None
    if requested is None:
        cuda_available = torch.cuda.is_available()
        requested = "cuda" if cuda_available else "cpu"
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
        if cuda_available is None:
            cuda_available = torch.cuda.is_available()
        if not cuda_available:
            raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU.")
        device_count = torch.cuda.device_count()
        if device_count < 1:
            raise RuntimeError(
                "CUDA was reported available but has no visible devices."
            )
        if device.index is None:
            current_index = torch.cuda.current_device()
            if (
                isinstance(current_index, bool)
                or not isinstance(current_index, Integral)
                or current_index < 0
                or current_index >= device_count
            ):
                raise RuntimeError(
                    f"CUDA current device index {current_index!r} is unavailable; "
                    f"visible device count is {device_count}."
                )
            device = torch.device("cuda", current_index)
        elif device.index >= device_count:
            raise RuntimeError(
                f"CUDA device index {device.index} is unavailable; "
                f"visible device count is {device_count}."
            )
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        if device.index not in (None, 0):
            raise RuntimeError("MPS exposes only device index 0.")
    return device


def compute_appliance_stats(
    train_appliances: Iterable,
    *,
    std_floor: float = 1.0,
    std_fallback: float = 100.0,
    include_extrema: bool = False,
):
    """Compute finite normalization statistics for one target power channel.

    ``std_fallback`` preserves the historical contrib convention for nearly
    constant targets while making that policy explicit and configurable.
    Frames may be one-dimensional or two-dimensional with exactly one column.
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
        _validate_appliance_name(appliance_name)
        if appliance_name in statistics:
            raise ValueError(f"Duplicate appliance name {appliance_name!r}.")
        if isinstance(frames, (str, bytes)):
            raise TypeError(
                f"Training frames for {appliance_name!r} must be iterable arrays."
            )
        try:
            frame_entries = iter(frames)
        except TypeError as exc:
            raise TypeError(
                f"Training frames for {appliance_name!r} must be iterable."
            ) from exc
        saw_frame = False
        sample_count = 0
        mean = 0.0
        sum_squared_deviations = 0.0
        minimum = math.inf
        maximum = -math.inf
        for frame in frame_entries:
            saw_frame = True
            try:
                if hasattr(frame, "to_numpy"):
                    raw_values = frame.to_numpy()
                else:
                    raw_values = np.asarray(frame)
                if np.iscomplexobj(raw_values):
                    raise ValueError("complex values are unsupported")
                values = np.asarray(raw_values, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Training data for {appliance_name!r} must be real numeric data."
                ) from exc
            if values.ndim == 1:
                pass
            elif values.ndim == 2 and values.shape[1] == 1:
                values = values[:, 0]
            else:
                raise ValueError(
                    f"Training data for {appliance_name!r} must contain exactly "
                    "one power column."
                )
            if values.size == 0:
                continue
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Training data for {appliance_name!r} must be finite."
                )

            frame_count = int(values.size)
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
            if include_extrema:
                minimum = min(minimum, float(np.min(values)))
                maximum = max(maximum, float(np.max(values)))

        if not saw_frame:
            raise ValueError(
                f"Training data for {appliance_name!r} contains no frames."
            )
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


class TorchStateSpaceDisaggregator(Disaggregator):
    """Shared runtime for PyTorch state-space models without fake neural options."""

    def __init__(self, params=None):
        if params is None:
            params = {}
        if not isinstance(params, Mapping):
            raise TypeError("params must be a mapping or None.")
        seed = get_param(params, "seed")
        verbose = get_param(params, "verbose", False)
        chunk_wise_training = get_param(params, "chunk_wise_training", False)
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, Integral)
        ):
            raise ValueError("seed must be an integer or None.")
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean.")
        if not isinstance(chunk_wise_training, bool):
            raise ValueError("chunk_wise_training must be a boolean.")

        super().__init__()
        initialize_runtime(
            self,
            {"seed": seed, "verbose": verbose},
            backends=("python", "numpy", "torch"),
        )
        self.device = resolve_torch_device(get_param(params, "device"))
        self.seed = seed
        self.verbose = verbose
        self.chunk_wise_training = chunk_wise_training
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

    def _validate_model_record(self, appliance_name, model):
        del appliance_name, model
        raise NotImplementedError("State-space subclasses must validate model records.")

    def require_models(self, models=None):
        """Validate and optionally install a detached state-space model mapping."""
        install_models = models is not None
        candidate = models if install_models else self.models
        if not isinstance(candidate, Mapping):
            raise TypeError("models must be a mapping of appliance names to records.")
        if not candidate:
            if install_models:
                raise ValueError("models must contain at least one appliance model.")
            model_name = getattr(self, "MODEL_NAME", self.__class__.__name__)
            raise RuntimeError(f"{model_name} requires a trained or loaded model.")

        validated = OrderedDict()
        for appliance_name, model in candidate.items():
            _validate_appliance_name(appliance_name, label="Model appliance")
            self._validate_model_record(appliance_name, model)
            validated[appliance_name] = model
        if install_models:
            self.models = validated
        return validated


class TorchDisaggregator(Disaggregator):
    """Shared runtime state for one-model-per-appliance disaggregators.

    Subclasses still implement their architecture, preprocessing, training,
    inference, and persistence. The base only centralizes configuration that
    otherwise drifts between model implementations.
    """

    APPLIANCE_STD_FLOOR = 1.0
    APPLIANCE_STD_FALLBACK = 100.0
    INCLUDE_APPLIANCE_EXTREMA = False
    REQUIRED_APPLIANCE_STATS = ("mean", "std")

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

        appliance_params = _copy_appliance_params(common.appliance_params)
        device = resolve_torch_device(common.device)

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
        self.appliance_params = appliance_params
        self.save_model_path = common.save_model_path
        self.load_model_path = common.pretrained_model_path
        self.chunk_wise_training = common.chunk_wise_training
        self.models = OrderedDict()
        self.device = device

    def set_appliance_params(self, train_appliances):
        """Update target normalization using the class's explicit policy."""
        statistics = compute_appliance_stats(
            train_appliances,
            std_floor=self.APPLIANCE_STD_FLOOR,
            std_fallback=self.APPLIANCE_STD_FALLBACK,
            include_extrema=self.INCLUDE_APPLIANCE_EXTREMA,
        )
        self.appliance_params.update(statistics)

    def _validated_appliance_stats(self, appliance_name, required_fields):
        """Return validated statistics required by one model appliance."""
        _validate_appliance_name(appliance_name, label="Model appliance")
        if appliance_name not in self.appliance_params:
            raise ValueError(
                f"Model appliance {appliance_name!r} has no normalization parameters."
            )
        statistics = _copy_appliance_params(
            {appliance_name: self.appliance_params[appliance_name]}
        )[appliance_name]
        missing = set(required_fields).difference(statistics)
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(
                f"Normalization parameters for model appliance "
                f"{appliance_name!r} are missing {fields}."
            )
        return statistics

    def appliance_minmax_params(self, appliance_name):
        """Return min, max, and a safe scale for min-max normalization.

        A constant target has a valid physical range of zero. Its normalized
        training target is therefore all zeros, using the established standard
        deviation fallback solely as a non-zero divisor. Denormalization should
        continue to use ``maximum - minimum`` so a constant target stays constant.
        """
        statistics = self._validated_appliance_stats(appliance_name, ("min", "max"))
        minimum = statistics["min"]
        maximum = statistics["max"]
        value_range = maximum - minimum
        scale = value_range if value_range > 0 else self.APPLIANCE_STD_FALLBACK
        return minimum, maximum, scale

    def _validate_model_device(self, appliance_name, model):
        devices = {parameter.device for parameter in model.parameters()}
        devices.update(buffer.device for buffer in model.buffers())
        incompatible = sorted(
            (
                device
                for device in devices
                if device.type != self.device.type
                or (self.device.index is not None and device.index != self.device.index)
            ),
            key=str,
        )
        if incompatible:
            actual = ", ".join(str(device) for device in incompatible)
            raise ValueError(
                f"Model for {appliance_name!r} uses incompatible device(s) "
                f"{actual}; expected {self.device}. Move the model before "
                "installing it."
            )

    def require_models(self, models=None):
        """Optionally install and then return a validated model mapping."""
        install_models = models is not None
        candidate = models if install_models else self.models
        if not isinstance(candidate, Mapping):
            raise TypeError("models must be a mapping of appliance names to modules.")
        if not candidate:
            if install_models:
                raise ValueError("models must contain at least one appliance model.")
            model_name = getattr(self, "MODEL_NAME", self.__class__.__name__)
            raise RuntimeError(f"{model_name} requires a trained or loaded model.")

        validated = OrderedDict()
        for appliance_name, model in candidate.items():
            _validate_appliance_name(appliance_name, label="Model appliance")
            if not isinstance(model, torch.nn.Module):
                raise TypeError(
                    f"Model for {appliance_name!r} must be a torch.nn.Module."
                )
            self._validate_model_device(appliance_name, model)
            validated[appliance_name] = model

        required_stats = self.REQUIRED_APPLIANCE_STATS
        if not isinstance(required_stats, tuple) or not all(
            isinstance(field, str) and field for field in required_stats
        ):
            raise TypeError("REQUIRED_APPLIANCE_STATS must be a tuple of field names.")
        for appliance_name in validated:
            if required_stats:
                self._validated_appliance_stats(appliance_name, required_stats)
        if install_models:
            self.models = validated
        return self.models
