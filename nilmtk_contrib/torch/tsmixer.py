"""TSMixer-inspired sequence-to-point energy disaggregator."""

from collections.abc import Mapping

import torch
from torch import nn

from nilmtk_contrib.torch._base import torch_defaults
from nilmtk_contrib.torch._seq2point import (
    SequenceToPointTorchDisaggregator,
    finite_number,
)
from nilmtk_contrib.utils.params import get_param, validate_positive_int


def _activation(name):
    if not isinstance(name, str):
        raise ValueError("activation must be 'relu' or 'gelu'.")
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError("activation must be 'relu' or 'gelu'.")


class TSMixerBlock(nn.Module):
    """Mix information along the temporal and feature dimensions."""

    def __init__(
        self,
        sequence_length,
        channels=1,
        ff_dim=64,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        for name, value in (
            ("sequence_length", sequence_length),
            ("channels", channels),
            ("ff_dim", ff_dim),
        ):
            validate_positive_int(name, value)
        dropout = finite_number("dropout", dropout, minimum=0)
        if dropout >= 1:
            raise ValueError("dropout must be less than 1.")

        normalized_shape = (sequence_length, channels)
        self.temporal_norm = nn.LayerNorm(normalized_shape)
        self.temporal_linear = nn.Linear(sequence_length, sequence_length)
        self.temporal_activation = _activation(activation)
        self.temporal_dropout = nn.Dropout(dropout)

        self.feature_norm = nn.LayerNorm(normalized_shape)
        self.feature_in = nn.Linear(channels, ff_dim)
        self.feature_activation = _activation(activation)
        self.feature_dropout = nn.Dropout(dropout)
        self.feature_out = nn.Linear(ff_dim, channels)
        self.output_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.temporal_linear, self.feature_in, self.feature_out):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        temporal = self.temporal_norm(inputs).transpose(1, 2)
        temporal = self.temporal_activation(self.temporal_linear(temporal))
        mixed = inputs + self.temporal_dropout(temporal).transpose(1, 2)

        features = self.feature_norm(mixed)
        features = self.feature_dropout(
            self.feature_activation(self.feature_in(features))
        )
        features = self.output_dropout(self.feature_out(features))
        return mixed + features


class TSMixerNetwork(nn.Module):
    """All-MLP mixer backbone with a centered scalar NILM head."""

    def __init__(
        self,
        sequence_length,
        ff_dim=64,
        n_blocks=2,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        validate_positive_int("sequence_length", sequence_length)
        validate_positive_int("ff_dim", ff_dim)
        validate_positive_int("n_blocks", n_blocks)
        if sequence_length % 2 == 0:
            raise ValueError("sequence_length must be odd.")
        dropout = finite_number("dropout", dropout, minimum=0)
        if dropout >= 1:
            raise ValueError("dropout must be less than 1.")
        _activation(activation)

        self.sequence_length = sequence_length
        self.blocks = nn.ModuleList(
            TSMixerBlock(
                sequence_length=sequence_length,
                channels=1,
                ff_dim=ff_dim,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(n_blocks)
        )
        self.head = nn.Linear(sequence_length, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("TSMixerNetwork inputs must be a torch.Tensor.")
        expected_shape = (1, self.sequence_length)
        if inputs.ndim != 3 or inputs.shape[1:] != expected_shape:
            raise ValueError(
                "TSMixerNetwork expects input shape "
                f"(batch, 1, {self.sequence_length}); got {tuple(inputs.shape)}."
            )
        if not torch.is_floating_point(inputs) or torch.is_complex(inputs):
            raise TypeError("TSMixerNetwork inputs must be real floating tensors.")
        device = self.head.weight.device
        if inputs.device != device:
            raise ValueError(
                f"TSMixerNetwork input is on {inputs.device}; model is on {device}."
            )
        if not torch.isfinite(inputs).all():
            raise ValueError("TSMixerNetwork inputs must be finite.")

        encoded = inputs.to(dtype=self.head.weight.dtype).transpose(1, 2)
        for block in self.blocks:
            encoded = block(encoded)
        output = self.head(encoded.transpose(1, 2)).squeeze(-1)
        if not torch.isfinite(output).all():
            raise RuntimeError("TSMixerNetwork produced non-finite output.")
        return output


class TSMixer(SequenceToPointTorchDisaggregator):
    """TSMixer adapted to one NILM estimate per centered mains row."""

    MODEL_NAME = "TSMixer"
    CHECKPOINT_PREFIX = "tsmixer"
    MODEL_CONFIG_FIELDS = (
        "ff_dim",
        "n_blocks",
        "dropout",
        "activation",
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
        self.ff_dim = validate_positive_int(
            "ff_dim", get_param(params, "ff_dim", 64)
        )
        self.n_blocks = validate_positive_int(
            "n_blocks", get_param(params, "n_blocks", 2)
        )
        self.dropout = finite_number(
            "dropout", get_param(params, "dropout", 0.1), minimum=0
        )
        self.activation = get_param(params, "activation", "relu")
        self._validate_architecture()
        if self.load_model_path:
            self.load_model()

    def _validate_architecture(self):
        if self.dropout >= 1:
            raise ValueError("dropout must be less than 1.")
        _activation(self.activation)

    def return_network(self):
        return TSMixerNetwork(
            sequence_length=self.sequence_length,
            ff_dim=self.ff_dim,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
            activation=self.activation,
        ).to(self.device)


__all__ = ["TSMixer", "TSMixerBlock", "TSMixerNetwork"]
