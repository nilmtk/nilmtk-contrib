"""Subtask-gated sequence-to-point energy disaggregator."""

from collections.abc import Mapping

import torch
from torch import nn

from nilmtk_contrib.torch._base import torch_defaults
from nilmtk_contrib.torch._seq2point import (
    SequenceToPointTorchDisaggregator,
    finite_number,
)
from nilmtk_contrib.utils.params import get_param, validate_positive_int


PAPER_KERNEL_SIZES = (10, 8, 6, 5, 5, 5)
PAPER_FILTERS = (30, 30, 40, 50, 50, 50)
PAPER_ON_POWER_THRESHOLD = 15.0


class SGNTower(nn.Module):
    """One paper-shaped CNN subnetwork with a scalar adaptation head."""

    def __init__(self, sequence_length, hidden_dim=1024, dropout=0.0):
        super().__init__()
        sequence_length = validate_positive_int("sequence_length", sequence_length)
        hidden_dim = validate_positive_int("hidden_dim", hidden_dim)
        dropout = finite_number("dropout", dropout, minimum=0)
        if dropout >= 1:
            raise ValueError("dropout must be less than 1.")
        reduced_length = sequence_length - sum(
            kernel_size - 1 for kernel_size in PAPER_KERNEL_SIZES
        )
        if reduced_length < 1:
            raise ValueError(
                f"sequence_length must be at least {1 + sequence_length - reduced_length}."
            )

        channels = (1, *PAPER_FILTERS[:-1])
        self.convolutions = nn.ModuleList(
            nn.Conv1d(in_channels, out_channels, kernel_size)
            for in_channels, out_channels, kernel_size in zip(
                channels, PAPER_FILTERS, PAPER_KERNEL_SIZES, strict=True
            )
        )
        self.feature_dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(PAPER_FILTERS[-1] * reduced_length, hidden_dim)
        self.hidden_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in (*self.convolutions, self.hidden):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.output.weight, nonlinearity="linear")
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs):
        encoded = inputs
        for convolution in self.convolutions:
            encoded = torch.relu(convolution(encoded))
        encoded = self.feature_dropout(encoded).flatten(1)
        encoded = self.hidden_dropout(torch.relu(self.hidden(encoded)))
        return self.output(encoded)


class SGNNetwork(nn.Module):
    """Independent regression and on/off towers joined by a soft gate."""

    def __init__(
        self,
        sequence_length,
        hidden_dim=1024,
        dropout=0.0,
    ):
        super().__init__()
        sequence_length = validate_positive_int("sequence_length", sequence_length)
        if sequence_length % 2 == 0:
            raise ValueError("sequence_length must be odd.")
        minimum_length = 1 + sum(size - 1 for size in PAPER_KERNEL_SIZES)
        if sequence_length < minimum_length:
            raise ValueError(f"sequence_length must be at least {minimum_length}.")
        self.sequence_length = sequence_length
        self.regression_tower = SGNTower(sequence_length, hidden_dim, dropout)
        self.classification_tower = SGNTower(sequence_length, hidden_dim, dropout)
        self.register_buffer("target_mean", torch.tensor(float("nan")))
        self.register_buffer("target_std", torch.tensor(float("nan")))

    def configure_target(self, mean, std):
        """Store the affine target transform used by the raw-power gate."""
        mean = finite_number("target mean", mean)
        std = finite_number("target std", std, minimum=0, strict=True)
        self.target_mean.copy_(self.target_mean.new_tensor(mean))
        self.target_std.copy_(self.target_std.new_tensor(std))

    def _validate_inputs(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("SGNNetwork inputs must be a torch.Tensor.")
        expected_shape = (1, self.sequence_length)
        if inputs.ndim != 3 or inputs.shape[1:] != expected_shape:
            raise ValueError(
                "SGNNetwork expects input shape "
                f"(batch, 1, {self.sequence_length}); got {tuple(inputs.shape)}."
            )
        if not torch.is_floating_point(inputs) or torch.is_complex(inputs):
            raise TypeError("SGNNetwork inputs must be real floating tensors.")
        device = self.target_mean.device
        if inputs.device != device:
            raise ValueError(
                f"SGNNetwork input is on {inputs.device}; model is on {device}."
            )
        if not torch.isfinite(inputs).all():
            raise ValueError("SGNNetwork inputs must be finite.")
        if not torch.isfinite(self.target_mean) or not torch.isfinite(self.target_std):
            raise RuntimeError("SGNNetwork target normalization is not configured.")

    def training_outputs(self, inputs):
        """Return gated prediction, ungated regression, and on/off logits."""
        self._validate_inputs(inputs)
        inputs = inputs.to(dtype=self.target_mean.dtype)
        regression = self.regression_tower(inputs)
        logits = self.classification_tower(inputs)
        probability = torch.sigmoid(logits)
        off_normalized = -self.target_mean / self.target_std
        gated = off_normalized + probability * (regression - off_normalized)
        if not all(
            torch.isfinite(output).all() for output in (gated, regression, logits)
        ):
            raise RuntimeError("SGNNetwork produced non-finite output.")
        return gated, regression, logits

    def forward(self, inputs):
        gated, _, _ = self.training_outputs(inputs)
        return gated


class SGN(SequenceToPointTorchDisaggregator):
    """Subtask Gated Network adapted to centered scalar NILM estimates."""

    MODEL_NAME = "SGN"
    CHECKPOINT_PREFIX = "sgn"
    MODEL_CONFIG_FIELDS = (
        "hidden_dim",
        "dropout",
        "classification_weight",
        "on_power_threshold",
        "learning_rate",
        "weight_decay",
        "validation_fraction",
        "validation_strategy",
        "gradient_clip_norm",
    )

    def __init__(self, params=None):
        if params is None:
            params = {}
        if isinstance(params, Mapping):
            params = dict(params)
            params.setdefault("learning_rate", 1e-4)
        super().__init__(
            params, defaults=torch_defaults(sequence_length=299, batch_size=16)
        )
        self.hidden_dim = validate_positive_int(
            "hidden_dim", get_param(params, "hidden_dim", 1024)
        )
        self.dropout = finite_number(
            "dropout", get_param(params, "dropout", 0.0), minimum=0
        )
        if self.dropout >= 1:
            raise ValueError("dropout must be less than 1.")
        self.classification_weight = finite_number(
            "classification_weight",
            get_param(params, "classification_weight", 1.0),
            minimum=0,
            strict=True,
        )
        threshold = get_param(params, "on_power_threshold", None)
        self.on_power_threshold = (
            None
            if threshold is None
            else finite_number("on_power_threshold", threshold, minimum=0)
        )
        minimum_length = 1 + sum(size - 1 for size in PAPER_KERNEL_SIZES)
        if self.sequence_length < minimum_length:
            raise ValueError(f"sequence_length must be at least {minimum_length}.")
        if self.load_model_path:
            self.load_model()

    def _threshold_for(self, appliance_name):
        if self.on_power_threshold is not None:
            return self.on_power_threshold
        statistics = self._validated_appliance_stats(
            appliance_name, self.REQUIRED_APPLIANCE_STATS
        )
        for key in ("on_power_threshold", "threshold"):
            if key in statistics:
                return finite_number(
                    f"{appliance_name!r} {key}", statistics[key], minimum=0
                )
        return PAPER_ON_POWER_THRESHOLD

    def configure_network(self, network, appliance_name):
        if not isinstance(network, SGNNetwork):
            raise TypeError("SGN requires an SGNNetwork for every appliance.")
        statistics = self._validated_appliance_stats(
            appliance_name, self.REQUIRED_APPLIANCE_STATS
        )
        network.configure_target(statistics["mean"], statistics["std"])

    def training_loss(self, network, batch_inputs, batch_targets, appliance_name):
        inputs = batch_inputs.to(self.device).unsqueeze(1)
        gated, regression, logits = network.training_outputs(inputs)
        for output in (gated, regression, logits):
            self._checked_model_output(output, len(batch_inputs), appliance_name)
        target = batch_targets.to(self.device, dtype=gated.dtype)
        statistics = self._validated_appliance_stats(
            appliance_name, self.REQUIRED_APPLIANCE_STATS
        )
        raw_target = target * statistics["std"] + statistics["mean"]
        on_target = (raw_target > self._threshold_for(appliance_name)).to(gated.dtype)
        output_loss = nn.functional.mse_loss(gated, target)
        on_loss = nn.functional.binary_cross_entropy_with_logits(logits, on_target)
        return output_loss + self.classification_weight * on_loss

    def return_network(self):
        return SGNNetwork(
            sequence_length=self.sequence_length,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)


__all__ = ["SGN", "SGNNetwork", "SGNTower"]
