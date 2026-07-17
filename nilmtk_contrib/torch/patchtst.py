"""PatchTST-inspired sequence-to-point disaggregator.

The network follows PatchTST's core idea of tokenizing a time series into
overlapping patches before applying a Transformer encoder.  NILM requires one
point per centered mains window, so the forecasting head is replaced by a
small sequence-to-point regression head.
"""

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import math
from numbers import Real
from pathlib import Path

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
        values = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must contain real numeric data.") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must contain only finite values.")
    return values


def _power_vector(frame, label):
    values = _real_array(frame, label)
    if values.ndim == 1:
        vector = values
    elif values.ndim == 2 and values.shape[1] == 1:
        vector = values[:, 0]
    else:
        raise ValueError(f"{label} must contain exactly one power column.")
    if vector.size == 0:
        raise ValueError(f"{label} must contain at least one sample.")
    return vector


def _window_matrix(frame, sequence_length, label):
    values = _real_array(frame, label)
    if values.ndim != 2 or values.shape[1] != sequence_length:
        raise ValueError(
            f"{label} must have shape (samples, {sequence_length}); got {values.shape}."
        )
    if values.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one window.")
    return values


def _checkpoint_filename(appliance_name):
    digest = hashlib.sha256(appliance_name.encode("utf-8")).hexdigest()[:16]
    return f"patchtst-{digest}.pt"


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
        if inputs.ndim != 3 or inputs.shape[1:] != (1, self.sequence_length):
            raise ValueError(
                "PatchTSTNetwork expects input shape "
                f"(batch, 1, {self.sequence_length}); got {tuple(inputs.shape)}."
            )
        padded = nn.functional.pad(inputs, (0, self.patch_stride), mode="replicate")
        patches = padded.unfold(-1, self.patch_length, self.patch_stride)
        tokens = self.patch_projection(patches.squeeze(1)) + self.position
        encoded = self.encoder(tokens)
        return self.head(encoded.flatten(start_dim=1))


class PatchTST(TorchDisaggregator):
    """PatchTST-inspired sequence-to-point NILM disaggregator."""

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

            missing_statistics = [
                entry
                for entry in appliance_entries
                if entry[0] not in self.appliance_params
            ]
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
        inputs, appliance_targets = self._training_arrays(
            train_main, train_appliances, do_preprocessing
        )
        if self.models:
            self.require_models()

        for appliance_name, targets in appliance_targets:
            model = self.models.get(appliance_name)
            if model is None:
                model = self.return_network()
                self.models[appliance_name] = model
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

            if best_state is not None:
                model.load_state_dict(best_state)
            model.to(self.device)

        if self.save_model_path:
            self.save_model()

    def save_model(self):
        models = self.require_models()
        if not self.save_model_path:
            raise ValueError("save_model_path is required to save PatchTST models.")
        model_folder = Path(self.save_model_path)
        model_folder.mkdir(parents=True, exist_ok=True)

        filenames = [_checkpoint_filename(name) for name in models]
        if len(filenames) != len(set(filenames)):
            raise RuntimeError("Appliance checkpoint filename collision.")
        for appliance_name, model in models.items():
            save_torch_state(model, model_folder / _checkpoint_filename(appliance_name))

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
        save_metadata(model_folder, metadata)

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
        for appliance_name in candidate.appliance_params:
            network = candidate.return_network()
            load_torch_state(
                network,
                model_folder / _checkpoint_filename(appliance_name),
                candidate.device,
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
        self.require_models(model)
        test_frames = list(test_main_list)
        if not test_frames:
            return []

        indexes = []
        if do_preprocessing:
            for index, frame in enumerate(test_frames):
                values = _power_vector(frame, f"mains chunk {index}")
                frame_index = getattr(frame, "index", None)
                indexes.append(
                    frame_index
                    if frame_index is not None
                    else pd.RangeIndex(len(values))
                )
            processed = self.call_preprocessing(test_frames, method="test")
        else:
            processed = test_frames

        results = []
        for index, frame in enumerate(processed):
            windows = _window_matrix(
                frame, self.sequence_length, f"mains chunk {index}"
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
            for appliance_name, network in self.models.items():
                network.eval()
                batches = []
                with torch.inference_mode():
                    for (batch,) in loader:
                        prediction = network(batch.to(self.device).unsqueeze(1))
                        if prediction.shape != (len(batch), 1):
                            raise ValueError(
                                f"Model for {appliance_name!r} returned shape "
                                f"{tuple(prediction.shape)}; expected ({len(batch)}, 1)."
                            )
                        if not torch.isfinite(prediction).all():
                            raise FloatingPointError(
                                f"Model for {appliance_name!r} returned non-finite values."
                            )
                        batches.append(prediction.detach().cpu())
                normalized = torch.cat(batches).numpy().reshape(-1)
                statistics = self.appliance_params[appliance_name]
                denormalized = statistics["mean"] + normalized * statistics["std"]
                denormalized = np.clip(denormalized, 0, None)
                if not np.isfinite(denormalized).all():
                    raise FloatingPointError(
                        f"Predictions for {appliance_name!r} became non-finite."
                    )
                predictions[appliance_name] = pd.Series(
                    denormalized.astype(np.float32, copy=False), index=output_index
                )
            results.append(pd.DataFrame(predictions, index=output_index))
        return results


__all__ = ["PatchTST", "PatchTSTNetwork"]
