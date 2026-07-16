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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from nilmtk.disaggregate import Disaggregator
from torch.utils.data import DataLoader, TensorDataset

from nilmtk_contrib.torch.preprocessing import preprocess
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
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the interval [0, 1).")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if self.validation_strategy not in {"tail", "random"}:
            raise ValueError("validation_strategy must be 'tail' or 'random'.")

        device_name = common.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if str(device_name).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested but PyTorch cannot see one.")
        self.device = torch.device(device_name)
        self.last_split_metadata = None

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
            values = np.concatenate(
                [frame.to_numpy(dtype=np.float32).reshape(-1) for frame in frames]
            )
            appliance_std = float(np.std(values))
            if appliance_std < 1.0:
                appliance_std = 1.0
            self.appliance_params[appliance_name] = {
                "mean": float(np.mean(values)),
                "std": appliance_std,
            }

    def _train_appliance(self, appliance_name, inputs, targets):
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

        inputs = np.concatenate(
            [frame.to_numpy(dtype=np.float32) for frame in train_main], axis=0
        )
        for appliance_name, frames in train_appliances:
            targets = np.concatenate(
                [frame.to_numpy(dtype=np.float32).reshape(-1) for frame in frames]
            )
            if len(inputs) != len(targets):
                raise ValueError(
                    f"Mains and {appliance_name} contain different sample counts: "
                    f"{len(inputs)} and {len(targets)}."
                )
            self._train_appliance(appliance_name, inputs, targets)

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        """Disaggregate mains chunks, batching inference on CPU or CUDA."""
        if model is not None:
            self.models = model

        output_indexes = [frame.index for frame in test_main_list]
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, None, "test")

        outputs = []
        for chunk_index, test_mains in zip(output_indexes, test_main_list):
            inputs = torch.from_numpy(
                test_mains.to_numpy(dtype=np.float32)
            ).unsqueeze(1)
            loader = DataLoader(inputs, batch_size=self.batch_size, shuffle=False)
            predictions = {}

            for appliance_name, network in self.models.items():
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
                predictions[appliance_name] = np.maximum(power, 0.0)

            outputs.append(
                pd.DataFrame(predictions, index=chunk_index, dtype="float32")
            )
        return outputs


__all__ = ["PatchTST", "PatchTSTNetwork"]
