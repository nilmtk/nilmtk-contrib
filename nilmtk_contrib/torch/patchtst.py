"""PatchTST-inspired sequence-to-point disaggregator.

The network follows PatchTST's core idea of tokenizing a time series into
overlapping patches before applying a Transformer encoder.  NILM requires one
point per centered mains window, so the forecasting head is replaced by a
small sequence-to-point regression head.
"""

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import math
from numbers import Real
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nilmtk_contrib.torch._base import TorchDisaggregator, torch_defaults
from nilmtk_contrib.torch.preprocessing import preprocess
from nilmtk_contrib.utils.checkpoints import (
    build_metadata,
    collect_dependencies,
    load_metadata,
    load_torch_state,
    save_metadata,
    save_torch_state,
)
from nilmtk_contrib.utils.logging import get_logger
from nilmtk_contrib.utils.params import (
    get_param,
    require_odd_sequence_length,
    validate_positive_int,
)
from nilmtk_contrib.utils.validation import train_validation_split


logger = get_logger(__name__)

_CHECKPOINT_FILE_PATTERN = re.compile(r"^patchtst-[0-9a-f]{16}(?:-[0-9a-f]{32})?\.pt$")

_MODEL_CONFIG_FIELDS = (
    "patch_length",
    "patch_stride",
    "d_model",
    "n_heads",
    "n_layers",
    "d_ff",
    "dropout",
    "learning_rate",
    "weight_decay",
    "validation_fraction",
    "validation_strategy",
    "gradient_clip_norm",
)


def _finite_number(name, value, *, minimum=None, strict=False):
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


def _real_array(frame, label):
    try:
        raw = frame.to_numpy() if hasattr(frame, "to_numpy") else np.asarray(frame)
        if np.iscomplexobj(raw):
            raise ValueError("complex values are unsupported")
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray(raw, dtype=np.float32)
    except OverflowError as exc:
        raise ValueError(
            f"{label} contains values that are not representable as float32."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must contain real numeric data.") from exc
    if not np.isfinite(values).all():
        try:
            input_is_finite = np.isfinite(raw).all()
        except TypeError:
            input_is_finite = True
        if input_is_finite:
            raise ValueError(
                f"{label} contains values that are not representable as float32."
            )
        raise ValueError(f"{label} must contain only finite values.")
    return values


def _power_vector(frame, label, *, allow_empty=False):
    values = _real_array(frame, label)
    if values.ndim == 1:
        vector = values
    elif values.ndim == 2 and values.shape[1] == 1:
        vector = values[:, 0]
    else:
        raise ValueError(f"{label} must contain exactly one power column.")
    if vector.size == 0 and not allow_empty:
        raise ValueError(f"{label} must contain at least one sample.")
    return vector


def _window_matrix(frame, sequence_length, label, *, allow_empty=False):
    values = _real_array(frame, label)
    if values.ndim != 2 or values.shape[1] != sequence_length:
        raise ValueError(
            f"{label} must have shape (samples, {sequence_length}); got {values.shape}."
        )
    if values.shape[0] == 0 and not allow_empty:
        raise ValueError(f"{label} must contain at least one window.")
    return values


def _checkpoint_filename(appliance_name, generation=None):
    digest = hashlib.sha256(appliance_name.encode("utf-8")).hexdigest()[:16]
    if generation is None:
        return f"patchtst-{digest}.pt"
    return f"patchtst-{digest}-{generation}.pt"


def _checkpoint_generation(appliance_name, filename):
    """Return a generation, ``None`` for legacy names, or ``False`` if unsafe."""
    if not isinstance(filename, str) or Path(filename).name != filename:
        return False
    legacy = _checkpoint_filename(appliance_name)
    if filename == legacy:
        return None
    prefix = legacy[:-3] + "-"
    if not filename.startswith(prefix) or not filename.endswith(".pt"):
        return False
    generation = filename[len(prefix) : -3]
    if len(generation) != 32 or any(
        character not in "0123456789abcdef" for character in generation
    ):
        return False
    return generation


def _legacy_model_files(model_folder, appliance_names):
    """Recover fixed-name or unambiguous generation checkpoints without a map."""
    recovered = OrderedDict()
    for appliance_name in appliance_names:
        legacy = model_folder / _checkpoint_filename(appliance_name)
        candidates = [legacy] if legacy.is_file() else []
        digest_prefix = legacy.stem + "-"
        candidates.extend(
            path
            for path in model_folder.glob(f"{digest_prefix}*.pt")
            if _checkpoint_generation(appliance_name, path.name) is not False
        )
        if len(candidates) > 1:
            raise ValueError(
                f"PatchTST checkpoint has ambiguous model files for {appliance_name!r}."
            )
        if candidates:
            recovered[appliance_name] = candidates[0].name
    return recovered


def _model_state_is_finite(model):
    for _, value in model.state_dict().items():
        if (
            torch.is_floating_point(value) or torch.is_complex(value)
        ) and not torch.isfinite(value).all().item():
            return False
    return True


def _snapshot_model_runtime(models):
    snapshots = OrderedDict()
    for appliance_name, model in models.items():
        snapshots[appliance_name] = {
            "model": model,
            "state": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "training": model.training,
            "gradients": {
                name: None
                if parameter.grad is None
                else parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
            },
        }
    return snapshots


def _restore_model_runtime(snapshots):
    models = OrderedDict()
    for appliance_name, snapshot in snapshots.items():
        model = snapshot["model"]
        model.load_state_dict(snapshot["state"])
        for name, parameter in model.named_parameters():
            gradient = snapshot["gradients"][name]
            parameter.grad = (
                None
                if gradient is None
                else gradient.to(device=parameter.device, dtype=parameter.dtype).clone()
            )
        model.train(snapshot["training"])
        models[appliance_name] = model
    return models


@contextmanager
def _isolated_torch_rng(device, seed):
    """Seed one fit without consuming another instance's global Torch RNG."""
    if seed is None:
        yield
        return

    cuda_devices = []
    if device.type == "cuda":
        cuda_devices.append(device.index)
    mps_state = None
    mps = getattr(torch, "mps", None)
    if device.type == "mps" and mps is not None and hasattr(mps, "get_rng_state"):
        mps_state = mps.get_rng_state()

    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(seed)
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


class PatchTSTNetwork(nn.Module):
    """Overlapping-patch Transformer with a scalar regression head."""

    def __init__(
        self,
        sequence_length,
        patch_length=16,
        patch_stride=8,
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_ff=128,
        dropout=0.1,
    ):
        super().__init__()
        for name, value in (
            ("sequence_length", sequence_length),
            ("patch_length", patch_length),
            ("patch_stride", patch_stride),
            ("d_model", d_model),
            ("n_heads", n_heads),
            ("n_layers", n_layers),
            ("d_ff", d_ff),
        ):
            validate_positive_int(name, value)
        if patch_length > sequence_length:
            raise ValueError("patch_length must not exceed sequence_length.")
        if patch_stride > patch_length:
            raise ValueError("patch_stride must not exceed patch_length.")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        dropout = _finite_number("dropout", dropout, minimum=0)
        if dropout >= 1:
            raise ValueError("dropout must be less than 1.")

        self.sequence_length = sequence_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.num_patches = (sequence_length - patch_length) // patch_stride + 2

        self.patch_projection = nn.Linear(patch_length, d_model)
        self.position = nn.Parameter(torch.empty(1, self.num_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(self.num_patches * d_model),
            nn.Linear(self.num_patches * d_model, 1),
        )
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("PatchTSTNetwork inputs must be a torch.Tensor.")
        if inputs.ndim != 3 or inputs.shape[1:] != (1, self.sequence_length):
            raise ValueError(
                "PatchTSTNetwork expects input shape "
                f"(batch, 1, {self.sequence_length}); got {tuple(inputs.shape)}."
            )
        if not torch.is_floating_point(inputs) or torch.is_complex(inputs):
            raise TypeError("PatchTSTNetwork inputs must be real floating tensors.")
        expected_device = self.patch_projection.weight.device
        if inputs.device != expected_device:
            raise ValueError(
                f"PatchTSTNetwork input is on {inputs.device}; "
                f"model is on {expected_device}."
            )
        if not torch.isfinite(inputs).all():
            raise ValueError("PatchTSTNetwork inputs must be finite.")
        inputs = inputs.to(dtype=self.patch_projection.weight.dtype)
        padded = nn.functional.pad(inputs, (0, self.patch_stride), mode="replicate")
        patches = padded.unfold(-1, self.patch_length, self.patch_stride)
        tokens = self.patch_projection(patches.squeeze(1)) + self.position
        encoded = self.encoder(tokens)
        output = self.head(encoded.flatten(start_dim=1))
        if not torch.isfinite(output).all():
            raise RuntimeError("PatchTSTNetwork produced non-finite output.")
        return output


class PatchTST(TorchDisaggregator):
    """PatchTST-inspired sequence-to-point NILM disaggregator.

    Prediction columns follow the installed model mapping's insertion order.
    New checkpoints preserve that order explicitly; legacy checkpoints use
    sorted appliance names for deterministic output.
    """

    MODEL_NAME = "PatchTST"

    def __init__(self, params=None):
        super().__init__(params, defaults=torch_defaults(batch_size=128))
        params = {} if params is None else params
        require_odd_sequence_length(self.sequence_length)

        self.patch_length = validate_positive_int(
            "patch_length", get_param(params, "patch_length", 16)
        )
        self.patch_stride = validate_positive_int(
            "patch_stride", get_param(params, "patch_stride", 8)
        )
        self.d_model = validate_positive_int(
            "d_model", get_param(params, "d_model", 64)
        )
        self.n_heads = validate_positive_int("n_heads", get_param(params, "n_heads", 4))
        self.n_layers = validate_positive_int(
            "n_layers", get_param(params, "n_layers", 3)
        )
        self.d_ff = validate_positive_int("d_ff", get_param(params, "d_ff", 128))
        self.dropout = _finite_number(
            "dropout", get_param(params, "dropout", 0.1), minimum=0
        )
        self.learning_rate = _finite_number(
            "learning_rate",
            get_param(params, "learning_rate", 1e-3),
            minimum=0,
            strict=True,
        )
        self.weight_decay = _finite_number(
            "weight_decay", get_param(params, "weight_decay", 0.0), minimum=0
        )
        self.validation_fraction = _finite_number(
            "validation_fraction",
            get_param(params, "validation_fraction", 0.15),
            minimum=0,
            strict=True,
        )
        self.validation_strategy = get_param(params, "validation_strategy", "tail")
        self.gradient_clip_norm = _finite_number(
            "gradient_clip_norm",
            get_param(params, "gradient_clip_norm", 1.0),
            minimum=0,
            strict=True,
        )
        self.last_split_metadata = OrderedDict()

        self._validate_configuration()
        if self.load_model_path:
            self.load_model()

    def _validate_configuration(self):
        if self.patch_length > self.sequence_length:
            raise ValueError("patch_length must not exceed sequence_length.")
        if self.patch_stride > self.patch_length:
            raise ValueError("patch_stride must not exceed patch_length.")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.dropout >= 1:
            raise ValueError("dropout must be less than 1.")
        if self.validation_fraction >= 1:
            raise ValueError("validation_fraction must be less than 1.")
        if not isinstance(
            self.validation_strategy, str
        ) or self.validation_strategy not in {"tail", "random"}:
            raise ValueError("validation_strategy must be 'tail' or 'random'.")

    def _model_config(self):
        return {field: getattr(self, field) for field in _MODEL_CONFIG_FIELDS}

    def return_network(self):
        return PatchTSTNetwork(
            sequence_length=self.sequence_length,
            patch_length=self.patch_length,
            patch_stride=self.patch_stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
        ).to(self.device)

    def call_preprocessing(self, mains_lst, submeters_lst=None, method="test"):
        if method not in {"train", "test"}:
            raise ValueError("method must be 'train' or 'test'.")
        mains_frames = []
        for index, frame in enumerate(mains_lst):
            values = _power_vector(frame, f"mains chunk {index}")
            frame_index = getattr(frame, "index", None)
            mains_frames.append(pd.DataFrame(values, index=frame_index))

        appliance_frames = None
        if method == "train":
            if submeters_lst is None:
                raise ValueError("Training requires appliance targets.")
            appliance_frames = []
            for appliance_name, frames in submeters_lst:
                converted = []
                for index, frame in enumerate(frames):
                    values = _power_vector(
                        frame, f"{appliance_name!r} target chunk {index}"
                    )
                    frame_index = getattr(frame, "index", None)
                    converted.append(pd.DataFrame(values, index=frame_index))
                appliance_frames.append((appliance_name, converted))

        return preprocess(
            sequence_length=self.sequence_length,
            mains_mean=self.mains_mean,
            mains_std=self.mains_std,
            mains_lst=mains_frames,
            submeters_lst=appliance_frames,
            method=method,
            appliance_params=self.appliance_params,
            windowing=False,
        )

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
            if not isinstance(appliance_name, str) or not appliance_name.strip():
                raise ValueError("Appliance names must be non-empty strings.")
            if appliance_name in seen:
                raise ValueError(f"Duplicate appliance name {appliance_name!r}.")
            seen.add(appliance_name)
            frame_list = list(frames)
            if len(frame_list) != len(mains_frames):
                raise ValueError(
                    f"{appliance_name!r} has {len(frame_list)} chunks but mains has "
                    f"{len(mains_frames)}."
                )
            appliance_entries.append((appliance_name, frame_list))
        if not appliance_entries:
            raise ValueError("Training requires at least one appliance target.")

        missing_statistics = [
            entry
            for entry in appliance_entries
            if entry[0] not in self.appliance_params
        ]
        if not do_preprocessing and missing_statistics:
            raise ValueError(
                "Preprocessed training requires appliance_params for every target."
            )

        if do_preprocessing:
            mains_vectors = [
                _power_vector(frame, f"mains chunk {index}")
                for index, frame in enumerate(mains_frames)
            ]
            for appliance_name, frames in appliance_entries:
                for index, (mains_frame, target_frame) in enumerate(
                    zip(mains_frames, frames)
                ):
                    target = _power_vector(
                        target_frame, f"{appliance_name!r} target chunk {index}"
                    )
                    if len(target) != len(mains_vectors[index]):
                        raise ValueError(
                            f"Mains and {appliance_name!r} target chunk {index} "
                            "must contain the same number of samples."
                        )
                    mains_index = getattr(mains_frame, "index", None)
                    target_index = getattr(target_frame, "index", None)
                    if (
                        mains_index is not None
                        and target_index is not None
                        and not mains_index.equals(target_index)
                    ):
                        raise ValueError(
                            f"Mains and {appliance_name!r} target chunk {index} "
                            "must have aligned indexes."
                        )

            if missing_statistics:
                self.set_appliance_params(missing_statistics)
            processed_main, processed_appliances = self.call_preprocessing(
                mains_frames, appliance_entries, method="train"
            )
        else:
            processed_main, processed_appliances = mains_frames, appliance_entries

        main_chunks = [
            _window_matrix(frame, self.sequence_length, f"mains chunk {index}")
            for index, frame in enumerate(processed_main)
        ]
        target_arrays = []
        for appliance_name, frames in processed_appliances:
            if len(frames) != len(main_chunks):
                raise ValueError(
                    f"{appliance_name!r} processed chunk count does not match mains."
                )
            targets = []
            for index, (main_chunk, frame) in enumerate(zip(main_chunks, frames)):
                target = _power_vector(
                    frame, f"{appliance_name!r} target chunk {index}"
                ).reshape(-1, 1)
                if len(target) != len(main_chunk):
                    raise ValueError(
                        f"Processed mains and {appliance_name!r} target chunk "
                        f"{index} must be aligned."
                    )
                main_index = getattr(processed_main[index], "index", None)
                target_index = getattr(frame, "index", None)
                if (
                    main_index is not None
                    and target_index is not None
                    and not main_index.equals(target_index)
                ):
                    raise ValueError(
                        f"Processed mains and {appliance_name!r} target chunk "
                        f"{index} must have aligned indexes."
                    )
                targets.append(target)
            target_arrays.append((appliance_name, np.concatenate(targets)))
        return np.concatenate(main_chunks), target_arrays

    def _validation_loss(self, model, inputs, targets):
        loader = DataLoader(
            TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        total_loss = torch.zeros((), device=self.device)
        total_samples = 0
        model.eval()
        with torch.inference_mode():
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device).unsqueeze(1)
                batch_targets = batch_targets.to(self.device)
                loss = nn.functional.mse_loss(
                    model(batch_inputs), batch_targets, reduction="sum"
                )
                total_loss += loss.detach()
                total_samples += len(batch_inputs)
        score = (total_loss / total_samples).item()
        if not math.isfinite(score):
            raise FloatingPointError("Validation loss became non-finite.")
        return score

    def partial_fit(
        self,
        train_main,
        train_appliances,
        do_preprocessing=True,
        current_epoch=0,
        **_,
    ):
        del current_epoch
        if self.n_epochs == 0:
            raise ValueError("PatchTST partial_fit requires at least one epoch.")
        appliance_params = deepcopy(self.appliance_params)
        split_metadata = deepcopy(self.last_split_metadata)
        model_runtime = _snapshot_model_runtime(self.models)
        try:
            with _isolated_torch_rng(self.device, self.seed):
                inputs, appliance_targets = self._training_arrays(
                    train_main, train_appliances, do_preprocessing
                )
                if self.models:
                    self.require_models()
                self._fit_appliance_targets(inputs, appliance_targets)
                if self.save_model_path:
                    self.save_model()
        except Exception:
            self.appliance_params = appliance_params
            self.last_split_metadata = split_metadata
            self.models = _restore_model_runtime(model_runtime)
            raise

    def _fit_appliance_targets(self, inputs, appliance_targets):
        for appliance_name, targets in appliance_targets:
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
                continue

            model = self.models.get(appliance_name)
            if model is None:
                model = self.return_network()
                self.models[appliance_name] = model

            train_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(split.X_train),
                    torch.from_numpy(split.y_train),
                ),
                batch_size=self.batch_size,
                shuffle=True,
                generator=(
                    None
                    if self.seed is None
                    else torch.Generator().manual_seed(self.seed)
                ),
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            best_loss = math.inf
            best_state = None

            for epoch in range(self.n_epochs):
                model.train()
                train_loss = torch.zeros((), device=self.device)
                train_samples = 0
                for batch_inputs, batch_targets in train_loader:
                    batch_inputs = batch_inputs.to(self.device).unsqueeze(1)
                    batch_targets = batch_targets.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.mse_loss(model(batch_inputs), batch_targets)
                    loss.backward()
                    try:
                        nn.utils.clip_grad_norm_(
                            model.parameters(),
                            self.gradient_clip_norm,
                            error_if_nonfinite=True,
                        )
                    except RuntimeError as exc:
                        raise FloatingPointError(
                            "Model gradients became non-finite."
                        ) from exc
                    optimizer.step()
                    train_loss += loss.detach() * len(batch_inputs)
                    train_samples += len(batch_inputs)

                if not _model_state_is_finite(model):
                    raise FloatingPointError(
                        "Optimizer step produced non-finite model state."
                    )
                score = (train_loss / train_samples).item()
                if split.metadata.validation_enabled:
                    score = self._validation_loss(model, split.X_val, split.y_val)
                if not math.isfinite(score):
                    raise FloatingPointError("Epoch score became non-finite.")
                if score < best_loss:
                    best_loss = score
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
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
                raise RuntimeError("PatchTST training produced no valid state.")
            model.load_state_dict(best_state)
            model.to(self.device)

    def save_model(self):
        models = self.require_models()
        if not self.save_model_path:
            raise ValueError("save_model_path is required to save PatchTST models.")
        nonfinite = [
            appliance_name
            for appliance_name, model in models.items()
            if not _model_state_is_finite(model)
        ]
        if nonfinite:
            names = ", ".join(repr(name) for name in nonfinite)
            raise FloatingPointError(
                f"Refusing to save non-finite PatchTST model state for {names}."
            )

        model_folder = Path(self.save_model_path)
        model_folder.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        model_files = OrderedDict(
            (
                appliance_name,
                _checkpoint_filename(appliance_name, generation),
            )
            for appliance_name in models
        )
        filenames = [filename for _, filename in model_files.items()]
        if len(filenames) != len(set(filenames)):
            raise RuntimeError("Appliance checkpoint filename collision.")
        if any((model_folder / filename).exists() for filename in filenames):
            raise RuntimeError("PatchTST checkpoint generation collision.")

        metadata = build_metadata(
            model_class=self.MODEL_NAME,
            backend="torch",
            sequence_length=self.sequence_length,
            appliance_params=self.appliance_params,
            mains_mean=self.mains_mean,
            mains_std=self.mains_std,
            dependencies=collect_dependencies(
                ["nilmtk-contrib", "torch", "numpy", "pandas"]
            ),
        )
        metadata["model_config"] = self._model_config()
        metadata["model_files"] = model_files
        metadata["model_order"] = list(models)

        staging_folder = Path(
            tempfile.mkdtemp(prefix=".patchtst-stage-", dir=model_folder)
        )
        published_files = []
        committed = False
        try:
            for appliance_name, filename in model_files.items():
                save_torch_state(models[appliance_name], staging_folder / filename)
            save_metadata(staging_folder, metadata)
            for filename in filenames:
                destination = model_folder / filename
                os.replace(staging_folder / filename, destination)
                published_files.append(destination)
            committed = True
            try:
                os.replace(
                    staging_folder / "metadata.json", model_folder / "metadata.json"
                )
            except Exception:
                committed = False
                raise
        finally:
            if not committed:
                for path in published_files:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        logger.warning(
                            "Could not remove unpublished PatchTST weight %s.", path
                        )
            shutil.rmtree(staging_folder, ignore_errors=True)

        current_files = set(filenames)
        for path in model_folder.glob("patchtst-*.pt"):
            if path.name not in current_files and _CHECKPOINT_FILE_PATTERN.fullmatch(
                path.name
            ):
                try:
                    path.unlink()
                except OSError:
                    logger.warning("Could not remove stale PatchTST weight %s.", path)

    def load_model(self):
        if not self.load_model_path:
            raise ValueError(
                "pretrained_model_path is required to load PatchTST models."
            )
        model_folder = Path(self.load_model_path)
        metadata = load_metadata(
            model_folder,
            expected_model_class=self.MODEL_NAME,
            expected_backend="torch",
        )
        config = metadata.get("model_config")
        if not isinstance(config, Mapping) or set(config) != set(_MODEL_CONFIG_FIELDS):
            raise ValueError("PatchTST checkpoint has an invalid model_config.")
        checkpoint_params = metadata.get("appliance_params")
        if not isinstance(checkpoint_params, Mapping) or not all(
            isinstance(name, str) for name in checkpoint_params
        ):
            raise ValueError("PatchTST checkpoint has invalid appliance_params.")
        model_files = metadata.get("model_files")
        if model_files is None:
            saved_order = metadata.get("model_order")
            if saved_order is None:
                appliance_names = sorted(checkpoint_params)
            elif (
                not isinstance(saved_order, list)
                or not all(isinstance(name, str) for name in saved_order)
                or len(saved_order) != len(set(saved_order))
                or not set(saved_order).issubset(checkpoint_params)
            ):
                raise ValueError("PatchTST checkpoint has an invalid model_order.")
            else:
                appliance_names = saved_order
            model_files = _legacy_model_files(model_folder, appliance_names)
        if not isinstance(model_files, Mapping) or not model_files:
            raise ValueError("PatchTST checkpoint has no valid model_files mapping.")
        if not set(model_files).issubset(checkpoint_params):
            raise ValueError(
                "PatchTST checkpoint model_files lack matching appliance_params."
            )
        generations = set()
        legacy_names = False
        for appliance_name, filename in model_files.items():
            if not isinstance(appliance_name, str):
                raise ValueError(
                    f"PatchTST checkpoint has an unsafe model file for "
                    f"{appliance_name!r}."
                )
            generation = _checkpoint_generation(appliance_name, filename)
            if generation is False:
                raise ValueError(
                    f"PatchTST checkpoint has an unsafe model file for "
                    f"{appliance_name!r}."
                )
            if generation is None:
                legacy_names = True
            else:
                generations.add(generation)
        if generations and (legacy_names or len(generations) != 1):
            raise ValueError(
                "PatchTST checkpoint model files mix incompatible generations."
            )

        model_order = metadata.get("model_order")
        if model_order is None:
            model_order = sorted(model_files)
        if (
            not isinstance(model_order, list)
            or not all(isinstance(name, str) for name in model_order)
            or len(model_order) != len(set(model_order))
            or set(model_order) != set(model_files)
        ):
            raise ValueError("PatchTST checkpoint has an invalid model_order.")

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
            filename = model_files[appliance_name]
            network = candidate.return_network()
            load_torch_state(
                network,
                model_folder / filename,
                candidate.device,
            )
            if not _model_state_is_finite(network):
                raise FloatingPointError(
                    f"PatchTST checkpoint contains non-finite model state for "
                    f"{appliance_name!r}."
                )
            loaded[appliance_name] = network
        candidate.require_models(loaded)

        self.sequence_length = candidate.sequence_length
        self.mains_mean = candidate.mains_mean
        self.mains_std = candidate.mains_std
        self.appliance_params = deepcopy(candidate.appliance_params)
        for field in _MODEL_CONFIG_FIELDS:
            setattr(self, field, getattr(candidate, field))
        self.require_models(candidate.models)

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        models = self.require_models(model)
        test_frames = list(test_main_list)
        if not test_frames:
            return []

        indexes = []
        if do_preprocessing:
            processed = []
            for index, frame in enumerate(test_frames):
                values = _power_vector(frame, f"mains chunk {index}", allow_empty=True)
                frame_index = getattr(frame, "index", None)
                indexes.append(
                    frame_index
                    if frame_index is not None
                    else pd.RangeIndex(len(values))
                )
                if len(values):
                    processed.append(self.call_preprocessing([frame], method="test")[0])
                else:
                    processed.append(
                        pd.DataFrame(
                            np.empty((0, self.sequence_length), dtype=np.float32)
                        )
                    )
        else:
            processed = test_frames

        results = []
        for index, frame in enumerate(processed):
            windows = _window_matrix(
                frame,
                self.sequence_length,
                f"mains chunk {index}",
                allow_empty=True,
            )
            output_index = (
                indexes[index]
                if do_preprocessing
                else getattr(frame, "index", pd.RangeIndex(len(windows)))
            )
            if len(output_index) != len(windows):
                raise ValueError(
                    f"mains chunk {index} index length does not match its windows."
                )
            loader = DataLoader(
                TensorDataset(torch.from_numpy(windows)),
                batch_size=self.batch_size,
                shuffle=False,
            )
            predictions = {}
            for appliance_name, network in models.items():
                network.eval()
                batches = []
                with torch.inference_mode():
                    for (batch,) in loader:
                        prediction = network(batch.to(self.device).unsqueeze(1))
                        if not isinstance(prediction, torch.Tensor):
                            raise TypeError(
                                f"Model for {appliance_name!r} must return "
                                "a torch.Tensor."
                            )
                        if prediction.shape != (len(batch), 1):
                            raise ValueError(
                                f"Model for {appliance_name!r} returned shape "
                                f"{tuple(prediction.shape)}; expected ({len(batch)}, 1)."
                            )
                        if not torch.is_floating_point(prediction) or torch.is_complex(
                            prediction
                        ):
                            raise TypeError(
                                f"Model for {appliance_name!r} must return real "
                                "floating values."
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
                        batches.append(prediction.detach().cpu())
                if batches:
                    normalized = torch.cat(batches).numpy().reshape(-1)
                else:
                    normalized = np.empty(0, dtype=np.float32)
                statistics = self.appliance_params[appliance_name]
                with np.errstate(over="ignore", invalid="ignore"):
                    denormalized = (
                        statistics["mean"]
                        + normalized.astype(np.float64) * statistics["std"]
                    )
                    denormalized = np.clip(denormalized, 0, None)
                if not np.isfinite(denormalized).all():
                    raise FloatingPointError(
                        f"Predictions for {appliance_name!r} became non-finite."
                    )
                with np.errstate(over="ignore", invalid="ignore"):
                    public_values = denormalized.astype(np.float32)
                if not np.isfinite(public_values).all():
                    raise ValueError(
                        f"Predictions for {appliance_name!r} overflow float32."
                    )
                predictions[appliance_name] = pd.Series(
                    public_values, index=output_index
                )
            results.append(pd.DataFrame(predictions, index=output_index))
        return results


__all__ = ["PatchTST", "PatchTSTNetwork"]
