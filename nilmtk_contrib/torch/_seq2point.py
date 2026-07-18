"""Reusable, fail-closed runtime for PyTorch sequence-to-point models."""

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import hmac
import math
from numbers import Real
import os
from pathlib import Path
import re
import tempfile
import uuid

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nilmtk_contrib.torch._base import (
    TorchDisaggregator,
    _validate_appliance_name,
)
from nilmtk_contrib.utils.checkpoints import (
    build_metadata,
    collect_dependencies,
    load_metadata,
    load_torch_payload,
    save_metadata_atomic,
)
from nilmtk_contrib.utils.logging import get_logger
from nilmtk_contrib.utils.params import get_param, require_odd_sequence_length
from nilmtk_contrib.utils.validation import train_validation_split


logger = get_logger(__name__)


def finite_number(name, value, *, minimum=None, strict=False):
    """Return a finite float subject to an optional lower bound."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None:
        invalid = value <= minimum if strict else value < minimum
        if invalid:
            qualifier = "greater than" if strict else "at least"
            raise ValueError(f"{name} must be {qualifier} {minimum}.")
    return float(value)


def _float32_array(frame, label):
    """Convert real numeric input without silently accepting lossy values."""
    raw = frame.to_numpy() if hasattr(frame, "to_numpy") else np.asarray(frame)
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"{label} must contain real numeric data.")
    if np.iscomplexobj(raw):
        raise TypeError(f"{label} must contain real numeric data.")
    if not np.isfinite(raw).all():
        raise ValueError(f"{label} must contain only finite values.")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} is not representable as float32.")
    return values


def power_vector(frame, label, *, allow_empty=False):
    """Validate one raw power channel and return a float32 vector."""
    values = _float32_array(frame, label)
    if values.ndim == 1:
        vector = values
    elif values.ndim == 2 and values.shape[1] == 1:
        vector = values[:, 0]
    else:
        raise ValueError(f"{label} must contain exactly one power column.")
    if not allow_empty and not len(vector):
        raise ValueError(f"{label} must contain at least one sample.")
    return vector


def window_matrix(frame, sequence_length, label, *, allow_empty=False):
    """Validate already-windowed sequence-to-point input."""
    values = _float32_array(frame, label)
    expected = ("samples", sequence_length)
    if values.ndim != 2 or values.shape[1] != sequence_length:
        raise ValueError(f"{label} must have shape {expected}; got {values.shape}.")
    if not allow_empty and not len(values):
        raise ValueError(f"{label} must contain at least one window.")
    return values


def centered_windows(values, sequence_length, mean, std):
    """Create one zero-padded normalized window for every raw input row."""
    values = power_vector(values, "mains", allow_empty=True)
    if not len(values):
        return np.empty((0, sequence_length), dtype=np.float32)
    padding = sequence_length // 2
    padded = np.pad(values, (padding, padding), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, sequence_length)
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = (windows.astype(np.float64) - mean) / std
        normalized = normalized.astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized mains windows must be finite float32 values.")
    return np.ascontiguousarray(normalized)


def _model_state_is_finite(model):
    state = model.state_dict()
    return all(
        not (torch.is_floating_point(value) or torch.is_complex(value))
        or torch.isfinite(value).all().item()
        for value in (state[name] for name in state)
    )


def _snapshot_models(models):
    snapshots = OrderedDict()
    for appliance_name, model in models.items():
        snapshots[appliance_name] = {
            "model": model,
            "state": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            "training": model.training,
            "gradients": {
                name: (
                    None if parameter.grad is None else parameter.grad.detach().clone()
                )
                for name, parameter in model.named_parameters()
            },
        }
    return snapshots


def _restore_models(snapshots):
    restored = OrderedDict()
    for appliance_name, snapshot in snapshots.items():
        model = snapshot["model"]
        model.load_state_dict(snapshot["state"])
        model.train(snapshot["training"])
        for name, parameter in model.named_parameters():
            gradient = snapshot["gradients"][name]
            parameter.grad = (
                None
                if gradient is None
                else gradient.to(parameter.device, parameter.dtype).clone()
            )
        restored[appliance_name] = model
    return restored


@contextmanager
def _isolated_torch_rng(device, seed):
    if seed is None:
        yield
        return
    cuda_devices = [device.index] if device.type == "cuda" else []
    mps = getattr(torch, "mps", None)
    mps_state = None
    if device.type == "mps" and mps is not None and hasattr(mps, "get_rng_state"):
        mps_state = mps.get_rng_state()
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        elif device.type == "mps" and mps is not None:
            mps.manual_seed(seed)
        try:
            yield
        finally:
            if mps_state is not None:
                mps.set_rng_state(mps_state)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SequenceToPointTorchDisaggregator(TorchDisaggregator):
    """Shared data, optimization, inference, and persistence contract.

    Subclasses only define their network and architecture parameters.  The
    engine keeps every raw row/index, validates preprocessed windows, trains
    one model per appliance, and publishes checksum-bound checkpoints.
    """

    CHECKPOINT_PREFIX = None
    MODEL_CONFIG_FIELDS = ()

    def __init__(self, params=None, *, defaults):
        super().__init__(params, defaults=defaults)
        params = {} if params is None else params
        require_odd_sequence_length(self.sequence_length)
        self.learning_rate = finite_number(
            "learning_rate",
            get_param(params, "learning_rate", 1e-3),
            minimum=0,
            strict=True,
        )
        self.weight_decay = finite_number(
            "weight_decay", get_param(params, "weight_decay", 0.0), minimum=0
        )
        self.validation_fraction = finite_number(
            "validation_fraction",
            get_param(params, "validation_fraction", 0.15),
            minimum=0,
            strict=True,
        )
        if self.validation_fraction >= 1:
            raise ValueError("validation_fraction must be less than 1.")
        self.validation_strategy = get_param(params, "validation_strategy", "tail")
        if not isinstance(
            self.validation_strategy, str
        ) or self.validation_strategy not in {
            "tail",
            "random",
        }:
            raise ValueError("validation_strategy must be 'tail' or 'random'.")
        self.gradient_clip_norm = finite_number(
            "gradient_clip_norm",
            get_param(params, "gradient_clip_norm", 1.0),
            minimum=0,
            strict=True,
        )
        self.last_split_metadata = OrderedDict()

    def return_network(self):
        raise NotImplementedError

    def model_config(self):
        fields = self.MODEL_CONFIG_FIELDS
        if (
            not isinstance(fields, tuple)
            or len(fields) != len(set(fields))
            or not all(isinstance(field, str) and field for field in fields)
        ):
            raise TypeError("MODEL_CONFIG_FIELDS must be a tuple of field names.")
        return {field: getattr(self, field) for field in fields}

    @staticmethod
    def _index(frame, length):
        index = getattr(frame, "index", None)
        return index if index is not None else pd.RangeIndex(length)

    def call_preprocessing(self, mains_lst, submeters_lst=None, method="test"):
        if not isinstance(method, str) or method not in {"train", "test"}:
            raise ValueError("method must be 'train' or 'test'.")
        mains_frames = list(mains_lst)
        processed_main = []
        for chunk_index, frame in enumerate(mains_frames):
            values = power_vector(
                frame, f"mains chunk {chunk_index}", allow_empty=method == "test"
            )
            windows = centered_windows(
                values, self.sequence_length, self.mains_mean, self.mains_std
            )
            processed_main.append(
                pd.DataFrame(windows, index=self._index(frame, len(values)))
            )
        if method == "test":
            return processed_main
        if submeters_lst is None:
            raise ValueError("Training requires appliance targets.")

        processed_appliances = []
        for appliance_name, frames in submeters_lst:
            statistics = self._validated_appliance_stats(
                appliance_name, self.REQUIRED_APPLIANCE_STATS
            )
            processed = []
            for chunk_index, frame in enumerate(frames):
                values = power_vector(
                    frame, f"{appliance_name!r} target chunk {chunk_index}"
                )
                with np.errstate(over="ignore", invalid="ignore"):
                    normalized = (
                        (values.astype(np.float64) - statistics["mean"])
                        / statistics["std"]
                    ).astype(np.float32)
                if not np.isfinite(normalized).all():
                    raise ValueError(
                        f"Normalized targets for {appliance_name!r} must be finite."
                    )
                processed.append(
                    pd.DataFrame(normalized, index=self._index(frame, len(values)))
                )
            processed_appliances.append((appliance_name, processed))
        return processed_main, processed_appliances

    def _training_arrays(self, train_main, train_appliances, do_preprocessing):
        mains_frames = list(train_main)
        if not mains_frames:
            raise ValueError("Training requires at least one mains chunk.")
        appliance_entries = []
        seen = set()
        for entry in train_appliances:
            try:
                appliance_name, frames = entry
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Each appliance target must be a (name, frames) pair."
                ) from exc
            _validate_appliance_name(appliance_name)
            if appliance_name in seen:
                raise ValueError(f"Duplicate appliance name {appliance_name!r}.")
            seen.add(appliance_name)
            frames = list(frames)
            if len(frames) != len(mains_frames):
                raise ValueError(
                    f"{appliance_name!r} has {len(frames)} chunks but mains has "
                    f"{len(mains_frames)}."
                )
            appliance_entries.append((appliance_name, frames))
        if not appliance_entries:
            raise ValueError("Training requires at least one appliance target.")

        missing = [
            entry
            for entry in appliance_entries
            if entry[0] not in self.appliance_params
        ]
        if not do_preprocessing and missing:
            raise ValueError(
                "Preprocessed training requires appliance_params for every target."
            )
        if do_preprocessing:
            mains_vectors = [
                power_vector(frame, f"mains chunk {index}")
                for index, frame in enumerate(mains_frames)
            ]
            for appliance_name, frames in appliance_entries:
                for index, (mains_frame, target_frame) in enumerate(
                    zip(mains_frames, frames)
                ):
                    target = power_vector(
                        target_frame, f"{appliance_name!r} target chunk {index}"
                    )
                    if len(target) != len(mains_vectors[index]):
                        raise ValueError(
                            f"Mains and {appliance_name!r} target chunk {index} "
                            "must contain the same number of samples."
                        )
                    if not self._index(mains_frame, len(target)).equals(
                        self._index(target_frame, len(target))
                    ):
                        raise ValueError(
                            f"Mains and {appliance_name!r} target chunk {index} "
                            "must have aligned indexes."
                        )
            if missing:
                self.set_appliance_params(missing)
            processed_main, processed_appliances = self.call_preprocessing(
                mains_frames, appliance_entries, method="train"
            )
        else:
            processed_main, processed_appliances = mains_frames, appliance_entries

        main_chunks = [
            window_matrix(frame, self.sequence_length, f"mains chunk {index}")
            for index, frame in enumerate(processed_main)
        ]
        targets_by_appliance = []
        for appliance_name, frames in processed_appliances:
            targets = []
            for index, (main_chunk, frame) in enumerate(zip(main_chunks, frames)):
                target = power_vector(
                    frame, f"{appliance_name!r} target chunk {index}"
                ).reshape(-1, 1)
                if len(target) != len(main_chunk):
                    raise ValueError(
                        f"Processed mains and {appliance_name!r} target chunk "
                        f"{index} must be aligned."
                    )
                if not self._index(processed_main[index], len(main_chunk)).equals(
                    self._index(frame, len(target))
                ):
                    raise ValueError(
                        f"Processed mains and {appliance_name!r} target chunk "
                        f"{index} must have aligned indexes."
                    )
                targets.append(target)
            targets_by_appliance.append((appliance_name, np.concatenate(targets)))
        return np.concatenate(main_chunks), targets_by_appliance

    def _checked_model_output(self, prediction, batch_size, appliance_name):
        if not isinstance(prediction, torch.Tensor):
            raise TypeError(f"Model for {appliance_name!r} must return a torch.Tensor.")
        if prediction.shape != (batch_size, 1):
            raise ValueError(
                f"Model for {appliance_name!r} returned shape "
                f"{tuple(prediction.shape)}; expected ({batch_size}, 1)."
            )
        if not torch.is_floating_point(prediction) or torch.is_complex(prediction):
            raise TypeError(
                f"Model for {appliance_name!r} must return real floating values."
            )
        if prediction.device != self.device:
            raise ValueError(
                f"Model for {appliance_name!r} returned values on "
                f"{prediction.device}; expected {self.device}."
            )
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(
                f"Model for {appliance_name!r} returned non-finite values."
            )
        return prediction

    def _checked_prediction(self, network, batch, appliance_name):
        prediction = network(batch.to(self.device).unsqueeze(1))
        return self._checked_model_output(prediction, len(batch), appliance_name)

    def training_loss(self, network, batch_inputs, batch_targets, appliance_name):
        """Return the scalar optimization objective for one training batch.

        Architecture subclasses may override this hook for auxiliary objectives
        while retaining the engine's optimization, rollback, and persistence
        contracts.  Custom hooks must validate any additional network outputs.
        """
        prediction = self._checked_prediction(network, batch_inputs, appliance_name)
        target = batch_targets.to(self.device)
        return nn.functional.mse_loss(prediction, target)

    def _checked_training_loss(
        self, network, batch_inputs, batch_targets, appliance_name
    ):
        loss = self.training_loss(network, batch_inputs, batch_targets, appliance_name)
        if not isinstance(loss, torch.Tensor):
            raise TypeError("training_loss must return a torch.Tensor.")
        if loss.shape != ():
            raise ValueError("training_loss must return a scalar tensor.")
        if not torch.is_floating_point(loss) or torch.is_complex(loss):
            raise TypeError("training_loss must return a real floating tensor.")
        if loss.device != self.device:
            raise ValueError(
                f"training_loss returned a value on {loss.device}; "
                f"expected {self.device}."
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss became non-finite.")
        if not loss.requires_grad:
            raise ValueError("training_loss must participate in autograd.")
        return loss

    def _score(self, network, inputs, targets, appliance_name):
        loader = DataLoader(
            TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets)),
            batch_size=self.batch_size,
        )
        total = torch.zeros((), device=self.device)
        count = 0
        network.eval()
        with torch.inference_mode():
            for batch_inputs, batch_targets in loader:
                prediction = self._checked_prediction(
                    network, batch_inputs, appliance_name
                )
                target = batch_targets.to(self.device)
                total += nn.functional.mse_loss(prediction, target, reduction="sum")
                count += len(batch_inputs)
        score = (total / count).item()
        if not math.isfinite(score):
            raise FloatingPointError("Validation loss became non-finite.")
        return score

    def _fit_appliance(self, appliance_name, inputs, targets):
        split = train_validation_split(
            inputs,
            targets,
            validation_fraction=self.validation_fraction,
            strategy=self.validation_strategy,
            seed=self.seed,
            min_train=1,
            min_val=1,
            allow_no_validation=True,
        )
        self.last_split_metadata[appliance_name] = split.metadata
        if not split.metadata.should_train:
            return
        network = self.models.get(appliance_name)
        if network is None:
            network = self.return_network()
            self.models[appliance_name] = network
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(split.X_train), torch.from_numpy(split.y_train)
            ),
            batch_size=self.batch_size,
            shuffle=True,
            generator=(
                None if self.seed is None else torch.Generator().manual_seed(self.seed)
            ),
        )
        optimizer = torch.optim.AdamW(
            network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        best_loss = math.inf
        best_state = None
        for epoch in range(self.n_epochs):
            network.train()
            total = torch.zeros((), device=self.device)
            count = 0
            for batch_inputs, batch_targets in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = self._checked_training_loss(
                    network, batch_inputs, batch_targets, appliance_name
                )
                loss.backward()
                try:
                    nn.utils.clip_grad_norm_(
                        network.parameters(),
                        self.gradient_clip_norm,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    raise FloatingPointError(
                        "Model gradients became non-finite."
                    ) from exc
                optimizer.step()
                total += loss.detach() * len(batch_inputs)
                count += len(batch_inputs)
            if not _model_state_is_finite(network):
                raise FloatingPointError("Optimizer produced non-finite model state.")
            score = (total / count).item()
            if split.metadata.validation_enabled:
                score = self._score(network, split.X_val, split.y_val, appliance_name)
            if not math.isfinite(score):
                raise FloatingPointError("Epoch score became non-finite.")
            if score < best_loss:
                best_loss = score
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                }
            if self.verbose:
                logger.info(
                    "%s %s epoch %d/%d score=%.6f",
                    self.MODEL_NAME,
                    appliance_name,
                    epoch + 1,
                    self.n_epochs,
                    score,
                )
        if best_state is None:
            raise RuntimeError(f"{self.MODEL_NAME} training produced no valid state.")
        network.load_state_dict(best_state)
        network.to(self.device)

    def partial_fit(
        self,
        train_main,
        train_appliances,
        do_preprocessing=True,
        current_epoch=0,
        **_,
    ):
        del current_epoch
        if not self.n_epochs:
            raise ValueError(f"{self.MODEL_NAME} partial_fit requires an epoch.")
        previous_params = deepcopy(self.appliance_params)
        previous_splits = deepcopy(self.last_split_metadata)
        previous_models = _snapshot_models(self.models)
        try:
            with _isolated_torch_rng(self.device, self.seed):
                inputs, targets_by_appliance = self._training_arrays(
                    train_main, train_appliances, do_preprocessing
                )
                if self.models:
                    self.require_models()
                for appliance_name, targets in targets_by_appliance:
                    self._fit_appliance(appliance_name, inputs, targets)
                if self.save_model_path:
                    self.save_model()
        except Exception:
            self.appliance_params = previous_params
            self.last_split_metadata = previous_splits
            self.models = _restore_models(previous_models)
            raise

    def _normalized_predictions(self, windows, network, appliance_name):
        loader = DataLoader(
            TensorDataset(torch.from_numpy(windows)),
            batch_size=self.batch_size,
        )
        batches = []
        network.eval()
        with torch.inference_mode():
            for (batch,) in loader:
                prediction = self._checked_prediction(network, batch, appliance_name)
                batches.append(prediction.detach().to(dtype=torch.float32).cpu())
        if not batches:
            return np.empty(0, dtype=np.float32)
        return torch.cat(batches).numpy().reshape(-1)

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        models = self.require_models(model)
        raw_frames = list(test_main_list)
        if not raw_frames:
            return []
        if do_preprocessing:
            processed = self.call_preprocessing(raw_frames, method="test")
            indexes = [
                self._index(frame, len(power_vector(frame, "mains", allow_empty=True)))
                for frame in raw_frames
            ]
        else:
            processed = raw_frames
            indexes = [self._index(frame, len(frame)) for frame in raw_frames]

        results = []
        for chunk_index, (frame, output_index) in enumerate(zip(processed, indexes)):
            windows = window_matrix(
                frame,
                self.sequence_length,
                f"mains chunk {chunk_index}",
                allow_empty=True,
            )
            if len(output_index) != len(windows):
                raise ValueError(
                    f"mains chunk {chunk_index} index length does not match windows."
                )
            columns = OrderedDict()
            for appliance_name, network in models.items():
                normalized = self._normalized_predictions(
                    windows, network, appliance_name
                )
                statistics = self._validated_appliance_stats(
                    appliance_name, self.REQUIRED_APPLIANCE_STATS
                )
                with np.errstate(over="ignore", invalid="ignore"):
                    values = np.clip(
                        statistics["mean"]
                        + normalized.astype(np.float64) * statistics["std"],
                        0,
                        None,
                    ).astype(np.float32)
                if not np.isfinite(values).all():
                    raise FloatingPointError(
                        f"Predictions for {appliance_name!r} became non-finite."
                    )
                columns[appliance_name] = pd.Series(values, index=output_index)
            results.append(pd.DataFrame(columns, index=output_index))
        return results

    def _checkpoint_pattern(self):
        prefix = self.CHECKPOINT_PREFIX
        if not isinstance(prefix, str) or not re.fullmatch(r"[a-z0-9-]+", prefix):
            raise TypeError("CHECKPOINT_PREFIX must contain lowercase letters/digits.")
        return re.compile(rf"^{re.escape(prefix)}-[0-9a-f]{{32}}\.pt$")

    def save_model(self, folder_name=None):
        models = self.require_models()
        destination = folder_name or self.save_model_path
        if not destination:
            raise ValueError("save_model_path is required to save models.")
        invalid = [
            name for name, model in models.items() if not _model_state_is_finite(model)
        ]
        if invalid:
            raise FloatingPointError(
                f"Refusing to save non-finite model state for {invalid!r}."
            )
        folder = Path(destination)
        folder.mkdir(parents=True, exist_ok=True)
        pattern = self._checkpoint_pattern()
        weights_name = f"{self.CHECKPOINT_PREFIX}-{uuid.uuid4().hex}.pt"
        weights_path = folder / weights_name
        if weights_path.exists():
            raise RuntimeError("Checkpoint generation collision.")
        payload = OrderedDict(
            (
                appliance_name,
                OrderedDict(
                    (name, value.detach().cpu().clone())
                    for name, value in model.state_dict().items()
                ),
            )
            for appliance_name, model in models.items()
        )
        temporary = None
        committed = False
        try:
            with tempfile.NamedTemporaryFile(
                dir=folder,
                prefix=f".{self.CHECKPOINT_PREFIX}-",
                suffix=".pt",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            torch.save(payload, temporary)
            os.replace(temporary, weights_path)
            temporary = None
            metadata = build_metadata(
                model_class=self.MODEL_NAME,
                backend="torch",
                sequence_length=self.sequence_length,
                appliance_params={name: self.appliance_params[name] for name in models},
                mains_mean=self.mains_mean,
                mains_std=self.mains_std,
                dependencies=collect_dependencies(
                    ["nilmtk-contrib", "torch", "numpy", "pandas"]
                ),
            )
            metadata.update(
                model_config=self.model_config(),
                model_order=list(models),
                weights_file=weights_name,
                weights_sha256=_sha256(weights_path),
            )
            save_metadata_atomic(folder, metadata)
            committed = True
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if not committed:
                weights_path.unlink(missing_ok=True)
        for path in folder.glob(f"{self.CHECKPOINT_PREFIX}-*.pt"):
            if path.name != weights_name and pattern.fullmatch(path.name):
                try:
                    path.unlink()
                except OSError:
                    logger.warning("Could not remove stale checkpoint %s.", path)

    def load_model(self, folder_name=None):
        source = folder_name or self.load_model_path
        if not source:
            raise ValueError("pretrained_model_path is required to load models.")
        folder = Path(source)
        metadata = load_metadata(
            folder,
            expected_model_class=self.MODEL_NAME,
            expected_backend="torch",
        )
        config = metadata.get("model_config")
        if not isinstance(config, Mapping) or set(config) != set(
            self.MODEL_CONFIG_FIELDS
        ):
            raise ValueError("Checkpoint has an invalid model_config.")
        model_order = metadata.get("model_order")
        checkpoint_params = metadata.get("appliance_params")
        if not isinstance(checkpoint_params, Mapping):
            raise ValueError("Checkpoint has invalid appliance_params.")
        if (
            not isinstance(model_order, list)
            or not model_order
            or not all(isinstance(name, str) for name in model_order)
            or len(model_order) != len(set(model_order))
            or set(model_order) != set(checkpoint_params)
        ):
            raise ValueError("Checkpoint has an invalid model_order.")
        weights_name = metadata.get("weights_file")
        if (
            not isinstance(weights_name, str)
            or Path(weights_name).name != weights_name
            or not self._checkpoint_pattern().fullmatch(weights_name)
        ):
            raise ValueError("Checkpoint has an unsafe weights_file.")
        expected_digest = metadata.get("weights_sha256")
        if not isinstance(expected_digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_digest
        ):
            raise ValueError("Checkpoint has an invalid weights_sha256.")
        weights_path = folder / weights_name
        if not weights_path.is_file():
            raise ValueError("Checkpoint weights file does not exist.")
        if not hmac.compare_digest(_sha256(weights_path), expected_digest):
            raise ValueError("Checkpoint weights checksum does not match metadata.")
        payload = load_torch_payload(weights_path, self.device)
        if not isinstance(payload, Mapping) or list(payload) != model_order:
            raise ValueError("Checkpoint weights do not match model_order.")

        candidate_params = {
            "sequence_length": metadata["sequence_length"],
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "mains_mean": metadata["mains_mean"],
            "mains_std": metadata["mains_std"],
            "appliance_params": metadata["appliance_params"],
            "chunk_wise_training": self.chunk_wise_training,
            "seed": self.seed,
            "verbose": self.verbose,
            "device": self.device,
            **dict(config),
        }
        candidate = type(self)(candidate_params)
        loaded = OrderedDict()
        for appliance_name in model_order:
            network = candidate.return_network()
            network.load_state_dict(payload[appliance_name])
            if not _model_state_is_finite(network):
                raise FloatingPointError(
                    f"Checkpoint contains non-finite state for {appliance_name!r}."
                )
            loaded[appliance_name] = network
        candidate.require_models(loaded)

        self.sequence_length = candidate.sequence_length
        self.mains_mean = candidate.mains_mean
        self.mains_std = candidate.mains_std
        self.appliance_params = deepcopy(candidate.appliance_params)
        for field in self.MODEL_CONFIG_FIELDS:
            setattr(self, field, getattr(candidate, field))
        self.require_models(candidate.models)


__all__ = [
    "SequenceToPointTorchDisaggregator",
    "centered_windows",
    "finite_number",
    "power_vector",
    "window_matrix",
]
