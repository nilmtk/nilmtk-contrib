"""PatchTST-inspired sequence-to-point disaggregator for NILM.

This is a clean NILMTK adaptation of PatchTST's defining ideas: split a time
series into overlapping patches, embed each patch as a token, process the
tokens with a Transformer encoder, and flatten the encoded patches for the
prediction head.

PatchTST paper: https://arxiv.org/abs/2211.14730
Reference implementation: https://github.com/yuqinie98/PatchTST

The reference forecasting model predicts future values of the same input
series. NILM instead maps aggregate mains power to a different appliance
series, so this implementation intentionally uses separate mains and target
normalization rather than reversing input-series normalization at the output.
"""

from collections import OrderedDict
from copy import deepcopy
import hashlib
import math
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from nilmtk.disaggregate import Disaggregator
from torch.utils.data import DataLoader, TensorDataset

from nilmtk_contrib.torch.preprocessing import preprocess
from nilmtk_contrib.utils.checkpoints import (
    build_metadata,
    collect_dependencies,
    load_metadata,
    load_torch_state,
    save_metadata,
    save_torch_state,
)
from nilmtk_contrib.utils.model import initialize_runtime, legacy_print, module_logger
from nilmtk_contrib.utils.params import (
    get_param,
    normalize_common_params,
    require_odd_sequence_length,
    validate_positive_int,
    validate_positive_number,
)
from nilmtk_contrib.utils.validation import train_validation_split


logger = module_logger(__name__)
_log_print = legacy_print(logger)


class SequenceLengthError(ValueError):
    """Raised when centered sequence-to-point windowing cannot be used."""


class ApplianceNotFoundError(ValueError):
    """Raised when target normalization statistics are unavailable."""


class PatchTSTNetwork(nn.Module):
    """Univariate patch-based Transformer with a scalar regression head."""

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
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, Real)
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be a finite number in [0, 1).")
        self.sequence_length = sequence_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride

        # PatchTST's ``padding_patch=end`` appends one stride using endpoint
        # replication, ensuring that the final part of the window is covered.
        self.num_patches = (sequence_length - patch_length) // patch_stride + 2
        self.patch_projection = nn.Linear(patch_length, d_model)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.num_patches, d_model)
        )

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
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(self.num_patches * d_model, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the learned positional encoding and regression head."""
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, inputs):
        """Predict normalized appliance power from ``[batch, 1, time]``."""
        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("PatchTSTNetwork expects input shape [batch, 1, time].")
        if inputs.shape[-1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, "
                f"received {inputs.shape[-1]}."
            )

        padded = F.pad(inputs, (0, self.patch_stride), mode="replicate")
        patches = padded.unfold(
            dimension=-1,
            size=self.patch_length,
            step=self.patch_stride,
        ).squeeze(1)
        tokens = self.patch_projection(patches) + self.position_embedding
        encoded = self.final_norm(self.encoder(tokens))
        return self.head(encoded)


class PatchTST(Disaggregator):
    """PatchTST-inspired sequence-to-point NILMTK disaggregator.

    Common parameters follow the other PyTorch disaggregators. PatchTST-specific
    parameters are ``patch_length`` (16), ``patch_stride`` (8), ``d_model``
    (64), ``n_heads`` (4), ``n_layers`` (3), ``d_ff`` (128), ``dropout``
    (0.1), ``learning_rate`` (1e-3), ``weight_decay`` (0),
    ``validation_fraction`` (0.15), and ``validation_strategy`` (``tail``).
    """

    def __init__(self, params):
        params = params or {}
        initialize_runtime(self, params, backends=("python", "numpy", "torch"))
        super().__init__()

        common = normalize_common_params(
            params,
            defaults={
                "sequence_length": 99,
                "n_epochs": 10,
                "batch_size": 128,
                "mains_mean": 1800.0,
                "mains_std": 600.0,
                "appliance_params": {},
                "save_model_path": None,
                "pretrained_model_path": None,
                "chunk_wise_training": False,
                "seed": None,
                "verbose": False,
                "device": None,
            },
        )
        try:
            require_odd_sequence_length(common.sequence_length)
        except ValueError as exc:
            raise SequenceLengthError(str(exc)) from exc

        self.MODEL_NAME = "PatchTST"
        self.models = OrderedDict()
        self.sequence_length = common.sequence_length
        self.n_epochs = common.n_epochs
        self.batch_size = common.batch_size
        self.mains_mean = common.mains_mean
        self.mains_std = common.mains_std
        self.appliance_params = dict(common.appliance_params)
        self.save_model_path = common.save_model_path
        self.load_model_path = common.pretrained_model_path
        self.chunk_wise_training = common.chunk_wise_training

        self.patch_length = get_param(params, "patch_length", 16)
        self.patch_stride = get_param(params, "patch_stride", 8)
        self.d_model = get_param(params, "d_model", 64)
        self.n_heads = get_param(params, "n_heads", 4)
        self.n_layers = get_param(params, "n_layers", 3)
        self.d_ff = get_param(params, "d_ff", 128)
        self.dropout = get_param(params, "dropout", 0.1)
        self.learning_rate = get_param(params, "learning_rate", 1e-3)
        self.weight_decay = get_param(params, "weight_decay", 0.0)
        self.validation_fraction = get_param(params, "validation_fraction", 0.15)
        self.validation_strategy = get_param(params, "validation_strategy", "tail")
        self.gradient_clip = get_param(params, "gradient_clip", 1.0)

        validate_positive_int("n_epochs", self.n_epochs)
        if (
            isinstance(self.mains_mean, bool)
            or not isinstance(self.mains_mean, Real)
            or not math.isfinite(self.mains_mean)
        ):
            raise ValueError("mains_mean must be a finite number.")
        if (
            isinstance(self.mains_std, bool)
            or not isinstance(self.mains_std, Real)
            or not math.isfinite(self.mains_std)
            or self.mains_std <= 0
        ):
            raise ValueError("mains_std must be a positive finite number.")

        for name in (
            "patch_length",
            "patch_stride",
            "d_model",
            "n_heads",
            "n_layers",
            "d_ff",
        ):
            validate_positive_int(name, getattr(self, name))
        validate_positive_number("learning_rate", self.learning_rate)
        validate_positive_number("gradient_clip", self.gradient_clip)
        if self.patch_length > self.sequence_length:
            raise ValueError("patch_length must not exceed sequence_length.")
        if self.patch_stride > self.patch_length:
            raise ValueError("patch_stride must not exceed patch_length.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, Real)
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must be a finite number in [0, 1).")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, Real)
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be a non-negative finite number.")
        if (
            isinstance(self.validation_fraction, bool)
            or not isinstance(self.validation_fraction, Real)
            or not math.isfinite(self.validation_fraction)
            or not 0 < self.validation_fraction < 1
        ):
            raise ValueError("validation_fraction must be between 0 and 1.")
        if self.validation_strategy not in {"tail", "random"}:
            raise ValueError("validation_strategy must be 'tail' or 'random'.")

        device_name = common.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.device = torch.device(device_name)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid torch device {device_name!r}.") from exc
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "A CUDA device was requested but PyTorch cannot see one."
                )
            if (
                self.device.index is not None
                and self.device.index >= torch.cuda.device_count()
            ):
                raise RuntimeError(
                    f"CUDA device index {self.device.index} is unavailable; "
                    f"visible device count is {torch.cuda.device_count()}."
                )
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("An MPS device was requested but is unavailable.")
        self.last_split_metadata = None
        if self.load_model_path:
            self.load_model()

    def return_network(self):
        """Build a PatchTST network on the configured device."""
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

    def _model_config(self):
        return {
            "sequence_length": self.sequence_length,
            "patch_length": self.patch_length,
            "patch_stride": self.patch_stride,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
        }

    @staticmethod
    def _model_filename(appliance_name):
        digest = hashlib.sha256(appliance_name.encode("utf-8")).hexdigest()[:16]
        return f"model-{digest}.pt"

    def save_model(self, folder_name=None):
        """Persist model weights, architecture, and normalization metadata."""
        destination = folder_name or self.save_model_path
        if not destination:
            raise ValueError(
                "A checkpoint directory is required via folder_name or "
                "save_model_path."
            )
        if not self.models:
            raise RuntimeError("PatchTST has no trained models to save.")

        model_folder = Path(destination)
        model_folder.mkdir(parents=True, exist_ok=True)
        model_files = {
            name: self._model_filename(name) for name in sorted(self.models)
        }
        metadata = build_metadata(
            model_class=self.MODEL_NAME,
            backend="torch",
            sequence_length=self.sequence_length,
            appliance_params=self.appliance_params,
            mains_mean=self.mains_mean,
            mains_std=self.mains_std,
            dependencies=collect_dependencies(
                ["nilmtk-contrib", "nilmtk", "torch", "numpy", "pandas"]
            ),
        )
        metadata["model_config"] = self._model_config()
        metadata["model_files"] = model_files

        for appliance_name, filename in model_files.items():
            save_torch_state(
                self.models[appliance_name],
                model_folder / filename,
            )
        # Metadata is written last so an interrupted save is never advertised as
        # a complete checkpoint before all weight files exist.
        save_metadata(model_folder, metadata)

    def load_model(self, folder_name=None):
        """Load a PatchTST checkpoint using weights-only Torch deserialization."""
        source = folder_name or self.load_model_path
        if not source:
            raise ValueError(
                "A checkpoint directory is required via folder_name or "
                "pretrained_model_path."
            )
        model_folder = Path(source)
        metadata = load_metadata(
            model_folder,
            expected_model_class=self.MODEL_NAME,
            expected_backend="torch",
        )
        model_config = metadata.get("model_config")
        model_files = metadata.get("model_files")
        if not isinstance(model_config, dict):
            raise ValueError("PatchTST metadata has no valid model_config.")
        if not isinstance(model_files, dict) or not model_files:
            raise ValueError("PatchTST metadata has no valid model_files mapping.")

        expected_config_fields = set(self._model_config())
        if set(model_config) != expected_config_fields:
            missing = sorted(expected_config_fields - set(model_config))
            extra = sorted(set(model_config) - expected_config_fields)
            raise ValueError(
                "PatchTST model_config fields do not match the checkpoint schema; "
                f"missing={missing}, extra={extra}."
            )
        common = normalize_common_params(
            {
                "sequence_length": model_config["sequence_length"],
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
                "mains_mean": metadata["mains_mean"],
                "mains_std": metadata["mains_std"],
                "appliance_params": metadata["appliance_params"],
                "chunk_wise_training": self.chunk_wise_training,
                "seed": self.seed,
                "verbose": self.verbose,
                "device": str(self.device),
            },
            defaults={},
        )
        try:
            require_odd_sequence_length(common.sequence_length)
        except ValueError as exc:
            raise SequenceLengthError(str(exc)) from exc
        if metadata["sequence_length"] != common.sequence_length:
            raise ValueError(
                "PatchTST metadata sequence_length disagrees with model_config."
            )

        self.sequence_length = common.sequence_length
        self.mains_mean = common.mains_mean
        self.mains_std = common.mains_std
        self.appliance_params = dict(common.appliance_params)
        for field in expected_config_fields - {"sequence_length", "dropout"}:
            validate_positive_int(field, model_config[field])
        self.patch_length = model_config["patch_length"]
        self.patch_stride = model_config["patch_stride"]
        self.d_model = model_config["d_model"]
        self.n_heads = model_config["n_heads"]
        self.n_layers = model_config["n_layers"]
        self.d_ff = model_config["d_ff"]
        self.dropout = model_config["dropout"]

        # return_network performs the final geometry/dropout validation before
        # any state is deserialized.
        restored = OrderedDict()
        for appliance_name, filename in sorted(model_files.items()):
            if appliance_name not in self.appliance_params:
                raise ValueError(
                    f"Checkpoint model {appliance_name!r} has no normalization data."
                )
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not filename.endswith(".pt")
            ):
                raise ValueError(
                    f"Unsafe checkpoint filename for {appliance_name!r}: {filename!r}."
                )
            network = self.return_network()
            load_torch_state(
                network,
                model_folder / filename,
                self.device,
            )
            network.eval()
            restored[appliance_name] = network
        self.models = restored

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        """Create centered mains windows and normalize mains/targets."""
        if method not in {"train", "test"}:
            raise ValueError("method must be 'train' or 'test'.")
        return preprocess(
            sequence_length=self.sequence_length,
            mains_mean=self.mains_mean,
            mains_std=self.mains_std,
            mains_lst=mains_lst,
            submeters_lst=submeters_lst,
            method=method,
            appliance_params=self.appliance_params,
            windowing=False,
        )

    def set_appliance_params(self, train_appliances):
        """Compute separate normalization statistics for each target series."""
        for appliance_name, frames in train_appliances:
            if not frames:
                raise ValueError(
                    f"Training data for {appliance_name!r} contains no frames."
                )
            values = np.concatenate(
                [frame.to_numpy(dtype=np.float32).reshape(-1) for frame in frames]
            )
            if values.size == 0:
                raise ValueError(
                    f"Training data for {appliance_name!r} contains no samples."
                )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Training data for {appliance_name!r} must be finite."
                )
            appliance_std = float(np.std(values))
            if appliance_std < 1.0:
                appliance_std = 1.0
            self.appliance_params[appliance_name] = {
                "mean": float(np.mean(values)),
                "std": appliance_std,
            }

    def _train_appliance(self, appliance_name, inputs, targets):
        if inputs.ndim != 2 or inputs.shape[1] != self.sequence_length:
            raise ValueError(
                f"PatchTST training mains must have shape [samples, "
                f"{self.sequence_length}]."
            )
        if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
            raise ValueError("PatchTST training inputs and targets must be finite.")
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
        self.last_split_metadata = split.metadata
        if not split.metadata.should_train:
            _log_print(f"Skipping {appliance_name}: {split.metadata.reason}")
            return

        if appliance_name not in self.models:
            self.models[appliance_name] = self.return_network()
        model = self.models[appliance_name]

        train_x = torch.from_numpy(split.X_train).unsqueeze(1)
        train_y = torch.from_numpy(split.y_train).reshape(-1, 1)
        generator = None
        if self.seed is not None:
            generator = torch.Generator().manual_seed(self.seed)
        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        validation_tensors = None
        if split.metadata.validation_enabled:
            validation_tensors = (
                torch.from_numpy(split.X_val).unsqueeze(1).to(self.device),
                torch.from_numpy(split.y_val).reshape(-1, 1).to(self.device),
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()
        best_loss = float("inf")
        best_state = None

        for epoch in range(self.n_epochs):
            model.train()
            batch_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_x), batch_y)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"PatchTST produced a non-finite loss for {appliance_name!r}."
                    )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))

            score = float(np.mean(batch_losses))
            if validation_tensors is not None:
                model.eval()
                with torch.no_grad():
                    score = float(
                        criterion(model(validation_tensors[0]), validation_tensors[1])
                        .detach()
                        .cpu()
                    )
            if not math.isfinite(score):
                raise RuntimeError(
                    f"PatchTST produced a non-finite selection loss for "
                    f"{appliance_name!r}."
                )
            if score < best_loss:
                best_loss = score
                best_state = deepcopy(model.state_dict())
            _log_print(
                f"{appliance_name} epoch {epoch + 1}/{self.n_epochs}: "
                f"selection_loss={score:.6f}"
            )

        if best_state is not None:
            model.load_state_dict(best_state)

    def partial_fit(
        self,
        train_main,
        train_appliances,
        do_preprocessing=True,
        current_epoch=0,
        **load_kwargs,
    ):
        """Train or continue training one PatchTST per appliance."""
        del current_epoch, load_kwargs
        train_main = list(train_main)
        train_appliances = [
            (name, list(frames)) for name, frames in train_appliances
        ]
        if not train_main:
            raise ValueError("PatchTST requires at least one mains training frame.")
        if not train_appliances:
            raise ValueError("PatchTST requires at least one appliance target.")
        appliance_names = [name for name, _ in train_appliances]
        if len(appliance_names) != len(set(appliance_names)):
            raise ValueError("PatchTST appliance targets must have unique names.")
        missing_stats = any(
            name not in self.appliance_params for name, _ in train_appliances
        )
        if missing_stats:
            self.set_appliance_params(train_appliances)

        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main,
                train_appliances,
                "train",
            )

        main_arrays = [
            frame.to_numpy(dtype=np.float32) for frame in train_main
        ]
        for values in main_arrays:
            if values.ndim != 2 or values.shape[1] != self.sequence_length:
                raise ValueError(
                    f"PatchTST training mains must have shape [samples, "
                    f"{self.sequence_length}]."
                )
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError(
                    "PatchTST training mains must be non-empty and finite."
                )
        inputs = np.concatenate(main_arrays, axis=0)
        for appliance_name, frames in train_appliances:
            if not frames:
                raise ValueError(
                    f"Training data for {appliance_name!r} contains no frames."
                )
            targets = np.concatenate(
                [frame.to_numpy(dtype=np.float32).reshape(-1) for frame in frames]
            )
            if targets.size == 0 or not np.isfinite(targets).all():
                raise ValueError(
                    f"Training data for {appliance_name!r} must be non-empty "
                    "and finite."
                )
            if len(inputs) != len(targets):
                raise ValueError(
                    f"Mains and {appliance_name} contain different sample counts: "
                    f"{len(inputs)} and {len(targets)}."
                )
            self._train_appliance(appliance_name, inputs, targets)
        if self.save_model_path:
            self.save_model()

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        """Disaggregate mains chunks, batching inference on CPU or CUDA."""
        if model is not None:
            self.models = OrderedDict(model)
        if not self.models:
            raise RuntimeError("PatchTST requires a trained or loaded model.")

        test_main_list = list(test_main_list)
        output_indexes = [frame.index for frame in test_main_list]
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, None, "test")

        outputs = []
        for chunk_index, test_mains in zip(
            output_indexes, test_main_list, strict=True
        ):
            input_values = test_mains.to_numpy(dtype=np.float32)
            if (
                input_values.ndim != 2
                or input_values.shape[1] != self.sequence_length
            ):
                raise ValueError(
                    f"PatchTST inference mains must have shape [samples, "
                    f"{self.sequence_length}]."
                )
            if input_values.size == 0 or not np.isfinite(input_values).all():
                raise ValueError(
                    "PatchTST inference mains must be non-empty and finite."
                )
            if len(chunk_index) != len(input_values):
                raise ValueError(
                    "PatchTST preprocessing changed the sample count unexpectedly."
                )
            inputs = torch.from_numpy(input_values).unsqueeze(1)
            loader = DataLoader(inputs, batch_size=self.batch_size, shuffle=False)
            predictions = {}

            for appliance_name, network in self.models.items():
                network = network.to(self.device)
                network.eval()
                normalized_batches = []
                with torch.no_grad():
                    for batch in loader:
                        normalized_batches.append(network(batch.to(self.device)).cpu())
                normalized = torch.cat(normalized_batches).numpy().reshape(-1)
                try:
                    stats = self.appliance_params[appliance_name]
                except KeyError as exc:
                    raise ApplianceNotFoundError(
                        f"Parameters for appliance {appliance_name!r} are unavailable."
                    ) from exc
                power = stats["mean"] + normalized * stats["std"]
                if not np.isfinite(power).all():
                    raise RuntimeError(
                        f"PatchTST produced non-finite predictions for "
                        f"{appliance_name!r}."
                    )
                predictions[appliance_name] = np.maximum(power, 0.0)

            outputs.append(
                pd.DataFrame(predictions, index=chunk_index, dtype="float32")
            )
        return outputs


__all__ = ["PatchTST", "PatchTSTNetwork"]
