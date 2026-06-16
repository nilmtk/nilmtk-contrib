"""Checkpoint and persistence helpers."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import atexit
import importlib.metadata
import inspect
import json
from pathlib import Path
import tempfile


METADATA_FILENAME = "metadata.json"
SCHEMA_VERSION = 1
_MANAGED_TEMP_DIRS = []


@dataclass(frozen=True)
class ModelMetadata:
    schema_version: int
    model_class: str
    backend: str
    sequence_length: int
    appliance_params: dict
    mains_mean: float
    mains_std: float
    created_at: str
    dependencies: dict


@contextmanager
def temporary_checkpoint(suffix):
    """Create a temporary checkpoint path that is removed on context exit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / f"checkpoint{suffix}"


def managed_checkpoint_path(suffix):
    """Return a process-managed temporary checkpoint path."""
    temp_dir = tempfile.TemporaryDirectory()
    _MANAGED_TEMP_DIRS.append(temp_dir)
    return Path(temp_dir.name) / f"checkpoint{suffix}"


def _cleanup_managed_temp_dirs():
    for temp_dir in _MANAGED_TEMP_DIRS:
        temp_dir.cleanup()


atexit.register(_cleanup_managed_temp_dirs)


def collect_dependencies(packages):
    """Return installed package versions for persistence metadata."""
    dependencies = {}
    for package in packages:
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    return dependencies


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def build_metadata(
    *,
    model_class,
    backend,
    sequence_length,
    appliance_params,
    mains_mean,
    mains_std,
    dependencies=None,
):
    """Build serializable model metadata."""
    return {
        "schema_version": SCHEMA_VERSION,
        "model_class": model_class,
        "backend": backend,
        "sequence_length": sequence_length,
        "appliance_params": _json_safe(appliance_params),
        "mains_mean": _json_safe(mains_mean),
        "mains_std": _json_safe(mains_std),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dependencies": dependencies or {},
    }


def save_metadata(path, metadata):
    """Write metadata JSON to a directory."""
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / METADATA_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def load_metadata(path, *, expected_model_class=None, expected_backend=None):
    """Load and validate persistence metadata."""
    metadata_path = Path(path) / METADATA_FILENAME
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    required_fields = {
        "schema_version",
        "model_class",
        "backend",
        "sequence_length",
        "appliance_params",
        "mains_mean",
        "mains_std",
        "created_at",
        "dependencies",
    }
    missing = required_fields.difference(metadata)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing metadata fields: {missing_list}.")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported metadata schema_version {metadata['schema_version']}."
        )
    if expected_model_class and metadata["model_class"] != expected_model_class:
        raise ValueError(
            f"Expected model_class {expected_model_class!r}, "
            f"got {metadata['model_class']!r}."
        )
    if expected_backend and metadata["backend"] != expected_backend:
        raise ValueError(
            f"Expected backend {expected_backend!r}, got {metadata['backend']!r}."
        )
    return metadata


def save_torch_state(model, path):
    """Save a PyTorch state dict."""
    import torch

    torch.save(model.state_dict(), path)


def load_torch_state(model, path, device, weights_only=True):
    """Load a PyTorch state dict, using weights_only where supported."""
    import torch

    load_kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = weights_only
    state = torch.load(path, **load_kwargs)
    model.load_state_dict(state)
    return model


def save_keras_weights(model, path):
    """Save Keras model weights."""
    model.save_weights(path)


def load_keras_weights(model, path):
    """Load Keras model weights."""
    model.load_weights(path)
    return model


def unsupported_persistence(model_name):
    """Raise a standard unsupported persistence error."""
    raise NotImplementedError(f"{model_name} does not implement model persistence.")
