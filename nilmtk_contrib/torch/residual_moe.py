"""Conservative residual mixture of strong NILM sequence-to-point experts."""

from collections.abc import Mapping
from typing import NamedTuple

import torch
from torch import nn

from nilmtk_contrib.torch._base import torch_defaults
from nilmtk_contrib.torch._seq2point import (
    SequenceToPointTorchDisaggregator,
    finite_number,
)
from nilmtk_contrib.torch._window_features import (
    WINDOW_FEATURE_NAMES,
    window_summary_features,
)
from nilmtk_contrib.torch.moderntcn import ModernTCNNetwork
from nilmtk_contrib.torch.patchtst import PatchTSTNetwork
from nilmtk_contrib.torch.timesnet import TimesNetNetwork
from nilmtk_contrib.utils.params import get_param, validate_positive_int


ANCHOR_NAME = "TimesNet"
SPECIALIST_NAMES = ("PatchTST", "ModernTCN")


class ResidualMoEOutputs(NamedTuple):
    """Inspectable predictions and routing decisions from ``ResidualMoENetwork``."""

    prediction: torch.Tensor
    anchor_prediction: torch.Tensor
    specialist_predictions: torch.Tensor
    specialist_weights: torch.Tensor
    correction_amplitude: torch.Tensor


class ResidualRouter(nn.Module):
    """Choose a specialist residual and its bounded signed amplitude."""

    def __init__(
        self,
        sequence_length,
        hidden_dim=32,
        dropout=0.1,
        residual_limit=0.25,
    ):
        super().__init__()
        self.sequence_length = validate_positive_int(
            "sequence_length", sequence_length
        )
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2.")
        hidden_dim = validate_positive_int("hidden_dim", hidden_dim)
        dropout = finite_number("dropout", dropout, minimum=0)
        if dropout >= 1:
            raise ValueError("dropout must be less than 1.")
        self.residual_limit = finite_number(
            "residual_limit", residual_limit, minimum=0, strict=True
        )
        if self.residual_limit > 1:
            raise ValueError("residual_limit must not exceed 1.")

        self.input_layer = nn.Linear(len(WINDOW_FEATURE_NAMES), hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.weight_layer = nn.Linear(hidden_dim, len(SPECIALIST_NAMES))
        self.amplitude_layer = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_uniform_(self.weight_layer.weight)
        nn.init.zeros_(self.weight_layer.bias)
        nn.init.zeros_(self.amplitude_layer.weight)
        nn.init.zeros_(self.amplitude_layer.bias)

    def summary_features(self, inputs):
        """Return the normalized window features used by the router."""
        return window_summary_features(
            inputs,
            sequence_length=self.sequence_length,
            reference=self.input_layer.weight,
            owner="ResidualRouter",
        )

    def forward(self, inputs):
        hidden = self.dropout(self.activation(self.input_layer(self.summary_features(inputs))))
        weights = torch.softmax(self.weight_layer(hidden), dim=1)
        amplitude = self.residual_limit * torch.tanh(self.amplitude_layer(hidden))
        if not torch.isfinite(weights).all() or not torch.isfinite(amplitude).all():
            raise RuntimeError("ResidualMoE router produced non-finite outputs.")
        return weights, amplitude


class ResidualMoENetwork(nn.Module):
    """Start from TimesNet and apply a bounded, input-conditioned correction."""

    def __init__(
        self,
        sequence_length,
        gate_hidden_dim=32,
        gate_dropout=0.1,
        residual_limit=0.25,
        expert_dropout=0.1,
    ):
        super().__init__()
        self.sequence_length = validate_positive_int(
            "sequence_length", sequence_length
        )
        if self.sequence_length % 2 == 0:
            raise ValueError("sequence_length must be odd.")
        if self.sequence_length < 25:
            raise ValueError("sequence_length must be at least 25.")
        expert_dropout = finite_number("expert_dropout", expert_dropout, minimum=0)
        if expert_dropout >= 1:
            raise ValueError("expert_dropout must be less than 1.")

        self.anchor = TimesNetNetwork(
            self.sequence_length,
            dropout=expert_dropout,
        )
        self.specialists = nn.ModuleDict(
            {
                "PatchTST": PatchTSTNetwork(
                    self.sequence_length,
                    dropout=expert_dropout,
                ),
                "ModernTCN": ModernTCNNetwork(
                    self.sequence_length,
                    dropout=expert_dropout,
                ),
            }
        )
        self.router = ResidualRouter(
            self.sequence_length,
            hidden_dim=gate_hidden_dim,
            dropout=gate_dropout,
            residual_limit=residual_limit,
        )

    def training_outputs(self, inputs):
        """Return the prediction, experts, and complete routing decision."""
        weights, amplitude = self.router(inputs)
        anchor_prediction = self.anchor(inputs)
        specialist_predictions = torch.stack(
            tuple(specialist(inputs) for specialist in self.specialists.values()),
            dim=-1,
        )
        specialist_residuals = specialist_predictions - anchor_prediction.unsqueeze(-1)
        selected_residual = torch.sum(
            specialist_residuals * weights.unsqueeze(1), dim=-1
        )
        prediction = anchor_prediction + amplitude * selected_residual
        outputs = ResidualMoEOutputs(
            prediction,
            anchor_prediction,
            specialist_predictions,
            weights,
            amplitude,
        )
        if not all(torch.isfinite(output).all() for output in outputs):
            raise RuntimeError("ResidualMoENetwork produced non-finite outputs.")
        return outputs

    def forward(self, inputs):
        return self.training_outputs(inputs).prediction


class ResidualMoE(SequenceToPointTorchDisaggregator):
    """TimesNet anchored residual mixture with independently trained specialists."""

    MODEL_NAME = "ResidualMoE"
    CHECKPOINT_PREFIX = "residual-moe"
    MODEL_CONFIG_FIELDS = (
        "gate_hidden_dim",
        "gate_dropout",
        "residual_limit",
        "expert_dropout",
        "auxiliary_weight",
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
        self.gate_hidden_dim = validate_positive_int(
            "gate_hidden_dim", get_param(params, "gate_hidden_dim", 32)
        )
        self.gate_dropout = finite_number(
            "gate_dropout", get_param(params, "gate_dropout", 0.1), minimum=0
        )
        self.residual_limit = finite_number(
            "residual_limit",
            get_param(params, "residual_limit", 0.25),
            minimum=0,
            strict=True,
        )
        self.expert_dropout = finite_number(
            "expert_dropout", get_param(params, "expert_dropout", 0.1), minimum=0
        )
        self.auxiliary_weight = finite_number(
            "auxiliary_weight",
            get_param(params, "auxiliary_weight", 0.1),
            minimum=0,
        )
        self._validate_architecture()
        if self.load_model_path:
            self.load_model()

    def _validate_architecture(self):
        if self.sequence_length < 25:
            raise ValueError("sequence_length must be at least 25.")
        for name in ("gate_dropout", "expert_dropout"):
            if getattr(self, name) >= 1:
                raise ValueError(f"{name} must be less than 1.")
        if self.residual_limit > 1:
            raise ValueError("residual_limit must not exceed 1.")

    def training_loss(self, network, batch_inputs, batch_targets, appliance_name):
        if not isinstance(network, ResidualMoENetwork):
            raise TypeError(
                "ResidualMoE requires a ResidualMoENetwork for every appliance."
            )
        inputs = batch_inputs.to(self.device).unsqueeze(1)
        outputs = network.training_outputs(inputs)
        for output in (
            outputs.prediction,
            outputs.anchor_prediction,
            *outputs.specialist_predictions.unbind(dim=-1),
        ):
            self._checked_model_output(output, len(batch_inputs), appliance_name)
        target = batch_targets.to(self.device, dtype=outputs.prediction.dtype)
        prediction_loss = nn.functional.mse_loss(outputs.prediction, target)
        specialist_loss = torch.stack(
            tuple(
                nn.functional.mse_loss(prediction, target)
                for prediction in outputs.specialist_predictions.unbind(dim=-1)
            )
        ).mean()
        return prediction_loss + self.auxiliary_weight * specialist_loss

    def return_network(self):
        return ResidualMoENetwork(
            sequence_length=self.sequence_length,
            gate_hidden_dim=self.gate_hidden_dim,
            gate_dropout=self.gate_dropout,
            residual_limit=self.residual_limit,
            expert_dropout=self.expert_dropout,
        ).to(self.device)


__all__ = [
    "ANCHOR_NAME",
    "SPECIALIST_NAMES",
    "ResidualMoE",
    "ResidualMoENetwork",
    "ResidualMoEOutputs",
    "ResidualRouter",
]
