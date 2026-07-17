"""Shared parameter parsing and validation helpers."""

from dataclasses import dataclass
import math
from numbers import Integral, Real
import warnings


@dataclass(frozen=True)
class CommonParams:
    sequence_length: int
    n_epochs: int
    batch_size: int
    mains_mean: float
    mains_std: float
    appliance_params: dict
    save_model_path: str | None
    pretrained_model_path: str | None
    chunk_wise_training: bool
    seed: int | None
    verbose: bool
    device: str | None


DEFAULT_ALIASES = {
    "save_model_path": ("save-model-path",),
    "pretrained_model_path": (
        "pretrained-model-path",
        "load_model_path",
        "load-model-path",
    ),
}


def get_param(params, canonical, default=None, aliases=(), required=False):
    """Return a parameter by canonical name, accepting deprecated aliases."""
    if params is None:
        params = {}

    if canonical in params:
        return params[canonical]

    for alias in aliases:
        if alias in params:
            warnings.warn(
                f"Parameter '{alias}' is deprecated; use '{canonical}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return params[alias]

    if required:
        raise ValueError(f"Missing required parameter '{canonical}'.")

    return default


def require_odd_sequence_length(sequence_length):
    """Validate models that require an odd sequence length."""
    if sequence_length % 2 == 0:
        raise ValueError("sequence_length must be odd.")


def validate_positive_int(name, value):
    """Validate a positive integer parameter."""
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def validate_non_negative_int(name, value):
    """Validate a non-negative integer parameter."""
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def validate_positive_number(name, value):
    """Validate a positive, finite real-valued parameter."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number.")
    return value


def _validate_non_zero_std(name, value):
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number.")
    if value == 0:
        raise ValueError(f"{name} must not be zero.")
    if value < 0 or not math.isfinite(value):
        raise ValueError(f"{name} must be a positive finite number.")
    return value


def _validate_finite_number(name, value):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number.")
    return value


def _validate_appliance_params(appliance_params):
    if not isinstance(appliance_params, dict):
        raise ValueError("appliance_params must be a dictionary.")
    for appliance, stats in appliance_params.items():
        if not isinstance(appliance, str) or not appliance.strip():
            raise ValueError("appliance_params keys must be non-empty strings.")
        if not isinstance(stats, dict):
            raise ValueError(
                f"appliance_params[{appliance!r}] must be a dictionary."
            )
        if "mean" in stats:
            _validate_finite_number(
                f"appliance_params[{appliance!r}]['mean']", stats["mean"]
            )
        if "std" in stats:
            _validate_non_zero_std(f"appliance_params[{appliance!r}]['std']", stats["std"])
    return appliance_params


def normalize_common_params(params, defaults):
    """Normalize common model parameters while preserving legacy aliases."""
    params = params or {}
    defaults = defaults or {}

    sequence_length = get_param(
        params,
        "sequence_length",
        default=defaults.get("sequence_length"),
    )
    n_epochs = get_param(params, "n_epochs", default=defaults.get("n_epochs"))
    batch_size = get_param(params, "batch_size", default=defaults.get("batch_size"))
    mains_mean = get_param(params, "mains_mean", default=defaults.get("mains_mean"))
    mains_std = get_param(params, "mains_std", default=defaults.get("mains_std"))
    appliance_params = get_param(
        params,
        "appliance_params",
        default=defaults.get("appliance_params", {}),
    )
    save_model_path = get_param(
        params,
        "save_model_path",
        default=defaults.get("save_model_path"),
        aliases=DEFAULT_ALIASES["save_model_path"],
    )
    pretrained_model_path = get_param(
        params,
        "pretrained_model_path",
        default=defaults.get("pretrained_model_path"),
        aliases=DEFAULT_ALIASES["pretrained_model_path"],
    )
    chunk_wise_training = get_param(
        params,
        "chunk_wise_training",
        default=defaults.get("chunk_wise_training", False),
    )
    seed = get_param(params, "seed", default=defaults.get("seed"))
    verbose = get_param(params, "verbose", default=defaults.get("verbose", False))
    device = get_param(params, "device", default=defaults.get("device"))

    validate_positive_int("sequence_length", sequence_length)
    validate_non_negative_int("n_epochs", n_epochs)
    validate_positive_int("batch_size", batch_size)
    _validate_finite_number("mains_mean", mains_mean)
    _validate_non_zero_std("mains_std", mains_std)
    _validate_appliance_params(appliance_params)
    if seed is not None and (
        not isinstance(seed, Integral) or isinstance(seed, bool)
    ):
        raise ValueError("seed must be an integer or None.")
    if not isinstance(verbose, bool):
        raise ValueError("verbose must be a boolean.")
    if not isinstance(chunk_wise_training, bool):
        raise ValueError("chunk_wise_training must be a boolean.")
    if device is not None and (not isinstance(device, str) or not device.strip()):
        raise ValueError("device must be a non-empty string or None.")

    return CommonParams(
        sequence_length=sequence_length,
        n_epochs=n_epochs,
        batch_size=batch_size,
        mains_mean=mains_mean,
        mains_std=mains_std,
        appliance_params=appliance_params,
        save_model_path=save_model_path,
        pretrained_model_path=pretrained_model_path,
        chunk_wise_training=chunk_wise_training,
        seed=seed,
        verbose=verbose,
        device=device,
    )
