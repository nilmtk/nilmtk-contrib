"""Input-conditioned mixture of complementary NILM experts."""

from collections.abc import Mapping

import torch
from torch import nn

from nilmtk_contrib.torch._base import torch_defaults
from nilmtk_contrib.torch._seq2point import (
    SequenceToPointTorchDisaggregator,
    finite_number,
)
from nilmtk_contrib.torch.dlinear import DLinearNetwork
from nilmtk_contrib.torch.moderntcn import ModernTCNNetwork
from nilmtk_contrib.torch.timesnet import TimesNetNetwork
from nilmtk_contrib.utils.params import get_param, validate_positive_int


EXPERT_NAMES = ("DLinear", "ModernTCN", "TimesNet")
GATE_FEATURES = (
    "center",
    "mean",
    "standard_deviation",
    "minimum",
    "maximum",
    "mean_absolute_difference",
    "end_to_start_change",
)


class NILMMoEGate(nn.Module):
    """Route each normalized mains window using inspectable summary features."""

    def __init__(
        self,
        sequence_length,
        hidden_dim=32,
        dropout=0.1,
        temperature=1.0,
    ):
        super().__init__()
        self.sequence_length = validate_positive_int(
            "sequence_length", sequence_length
        )
        hidden_dim = validate_positive_int("hidden_dim", hidden_dim)
        dropout = finite_number("dropout", dropout, minimum=0)
        if dropout >= 1:
            raise ValueError("dropout must be less than 1.")
        self.temperature = finite_number(
            "temperature", temperature, minimum=0, strict=True
        )
        self.input_layer = nn.Linear(len(GATE_FEATURES), hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, len(EXPERT_NAMES))
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def summary_features(self, inputs):
        """Return the seven normalized features used by the routing gate."""
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("NILMMoEGate inputs must be a torch.Tensor.")
        expected_shape = (1, self.sequence_length)
        if inputs.ndim != 3 or inputs.shape[1:] != expected_shape:
            raise ValueError(
                "NILMMoEGate expects input shape "
                f"(batch, 1, {self.sequence_length}); got {tuple(inputs.shape)}."
            )
        if not torch.is_floating_point(inputs) or torch.is_complex(inputs):
            raise TypeError("NILMMoEGate inputs must be real floating tensors.")
        device = self.input_layer.weight.device
        if inputs.device != device:
            raise ValueError(
                f"NILMMoEGate input is on {inputs.device}; model is on {device}."
            )
        if not torch.isfinite(inputs).all():
            raise ValueError("NILMMoEGate inputs must be finite.")
        inputs = inputs.to(dtype=self.input_layer.weight.dtype)
        center = inputs[:, :, self.sequence_length // 2]
        mean = inputs.mean(dim=-1)
        standard_deviation = inputs.std(dim=-1, unbiased=False)
        minimum = inputs.amin(dim=-1)
        maximum = inputs.amax(dim=-1)
        differences = inputs.diff(dim=-1)
        mean_absolute_difference = differences.abs().mean(dim=-1)
        end_to_start_change = inputs[:, :, -1] - inputs[:, :, 0]
        return torch.cat(
            (
                center,
                mean,
                standard_deviation,
                minimum,
                maximum,
                mean_absolute_difference,
                end_to_start_change,
            ),
            dim=1,
        )

    def forward(self, inputs):
        features = self.summary_features(inputs)
        logits = self.output_layer(
            self.dropout(self.activation(self.input_layer(features)))
        )
        weights = torch.softmax(logits / self.temperature, dim=1)
        if not torch.isfinite(weights).all():
            raise RuntimeError("NILMMoE gate produced non-finite weights.")
        return weights


class NILMMoENetwork(nn.Module):
    """Blend linear, convolutional, and periodic scalar experts."""

    def __init__(
        self,
        sequence_length,
        moving_average=25,
        gate_hidden_dim=32,
        gate_dropout=0.1,
        gate_temperature=1.0,
        expert_dropout=0.1,
    ):
        super().__init__()
        self.sequence_length = validate_positive_int(
            "sequence_length", sequence_length
        )
        moving_average = validate_positive_int("moving_average", moving_average)
        if sequence_length % 2 == 0:
            raise ValueError("sequence_length must be odd.")
        if sequence_length < 25:
            raise ValueError("sequence_length must be at least 25.")
        if moving_average % 2 == 0:
            raise ValueError("moving_average must be odd.")
        if moving_average > sequence_length:
            raise ValueError("moving_average must not exceed sequence_length.")
        expert_dropout = finite_number(
            "expert_dropout", expert_dropout, minimum=0
        )
        if expert_dropout >= 1:
            raise ValueError("expert_dropout must be less than 1.")

        self.experts = nn.ModuleDict(
            {
                "DLinear": DLinearNetwork(sequence_length, moving_average),
                "ModernTCN": ModernTCNNetwork(
                    sequence_length, dropout=expert_dropout
                ),
                "TimesNet": TimesNetNetwork(
                    sequence_length, dropout=expert_dropout
                ),
            }
        )
        self.gate = NILMMoEGate(
            sequence_length,
            hidden_dim=gate_hidden_dim,
            dropout=gate_dropout,
            temperature=gate_temperature,
        )

    def _validate_inputs(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("NILMMoENetwork inputs must be a torch.Tensor.")
        expected_shape = (1, self.sequence_length)
        if inputs.ndim != 3 or inputs.shape[1:] != expected_shape:
            raise ValueError(
                "NILMMoENetwork expects input shape "
                f"(batch, 1, {self.sequence_length}); got {tuple(inputs.shape)}."
            )
        if not torch.is_floating_point(inputs) or torch.is_complex(inputs):
            raise TypeError("NILMMoENetwork inputs must be real floating tensors.")
        device = self.gate.input_layer.weight.device
        if inputs.device != device:
            raise ValueError(
                f"NILMMoENetwork input is on {inputs.device}; model is on {device}."
            )
        if not torch.isfinite(inputs).all():
            raise ValueError("NILMMoENetwork inputs must be finite.")

    def training_outputs(self, inputs):
        """Return the mixture, individual predictions, and routing weights."""
        self._validate_inputs(inputs)
        inputs = inputs.to(dtype=self.gate.input_layer.weight.dtype)
        predictions = torch.stack(
            tuple(expert(inputs) for expert in self.experts.values()), dim=-1
        )
        weights = self.gate(inputs)
        mixture = torch.sum(predictions * weights.unsqueeze(1), dim=-1)
        if not torch.isfinite(mixture).all():
            raise RuntimeError("NILMMoENetwork produced non-finite output.")
        return mixture, predictions, weights

    def forward(self, inputs):
        mixture, _, _ = self.training_outputs(inputs)
        return mixture


class NILMMoE(SequenceToPointTorchDisaggregator):
    """End-to-end trainable mixture of complementary NILM experts."""

    MODEL_NAME = "NILMMoE"
    CHECKPOINT_PREFIX = "nilmmoe"
    MODEL_CONFIG_FIELDS = (
        "moving_average",
        "gate_hidden_dim",
        "gate_dropout",
        "gate_temperature",
        "expert_dropout",
        "load_balance_weight",
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
            params.setdefault("weight_decay", 1e-4)
        super().__init__(
            params, defaults=torch_defaults(sequence_length=299, batch_size=128)
        )
        self.moving_average = validate_positive_int(
            "moving_average", get_param(params, "moving_average", 25)
        )
        self.gate_hidden_dim = validate_positive_int(
            "gate_hidden_dim", get_param(params, "gate_hidden_dim", 32)
        )
        self.gate_dropout = finite_number(
            "gate_dropout", get_param(params, "gate_dropout", 0.1), minimum=0
        )
        self.gate_temperature = finite_number(
            "gate_temperature",
            get_param(params, "gate_temperature", 1.0),
            minimum=0,
            strict=True,
        )
        self.expert_dropout = finite_number(
            "expert_dropout", get_param(params, "expert_dropout", 0.1), minimum=0
        )
        self.load_balance_weight = finite_number(
            "load_balance_weight",
            get_param(params, "load_balance_weight", 0.01),
            minimum=0,
        )
        self._validate_architecture()
        if self.load_model_path:
            self.load_model()

    def _validate_architecture(self):
        if self.sequence_length < 25:
            raise ValueError("sequence_length must be at least 25.")
        if self.moving_average % 2 == 0:
            raise ValueError("moving_average must be odd.")
        if self.moving_average > self.sequence_length:
            raise ValueError("moving_average must not exceed sequence_length.")
        for name in ("gate_dropout", "expert_dropout"):
            if getattr(self, name) >= 1:
                raise ValueError(f"{name} must be less than 1.")

    def training_loss(self, network, batch_inputs, batch_targets, appliance_name):
        if not isinstance(network, NILMMoENetwork):
            raise TypeError("NILMMoE requires an NILMMoENetwork for every appliance.")
        inputs = batch_inputs.to(self.device).unsqueeze(1)
        prediction, expert_predictions, weights = network.training_outputs(inputs)
        self._checked_model_output(prediction, len(batch_inputs), appliance_name)
        for expert_prediction in expert_predictions.unbind(dim=-1):
            self._checked_model_output(
                expert_prediction, len(batch_inputs), appliance_name
            )
        target = batch_targets.to(self.device, dtype=prediction.dtype)
        prediction_loss = nn.functional.mse_loss(prediction, target)
        mean_routing = weights.mean(dim=0)
        balance_loss = len(EXPERT_NAMES) * mean_routing.square().sum() - 1
        return prediction_loss + self.load_balance_weight * balance_loss

    def return_network(self):
        return NILMMoENetwork(
            sequence_length=self.sequence_length,
            moving_average=self.moving_average,
            gate_hidden_dim=self.gate_hidden_dim,
            gate_dropout=self.gate_dropout,
            gate_temperature=self.gate_temperature,
            expert_dropout=self.expert_dropout,
        ).to(self.device)


__all__ = [
    "EXPERT_NAMES",
    "GATE_FEATURES",
    "NILMMoE",
    "NILMMoEGate",
    "NILMMoENetwork",
]
