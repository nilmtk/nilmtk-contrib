"""DLinear-inspired sequence-to-point energy disaggregator."""

import torch
from torch import nn

from nilmtk_contrib.torch._base import torch_defaults
from nilmtk_contrib.torch._seq2point import SequenceToPointTorchDisaggregator
from nilmtk_contrib.utils.params import get_param, validate_positive_int


class SeriesDecomposition(nn.Module):
    """Split a series into moving-average trend and seasonal residual."""

    def __init__(self, moving_average):
        super().__init__()
        validate_positive_int("moving_average", moving_average)
        if moving_average % 2 == 0:
            raise ValueError("moving_average must be odd to preserve alignment.")
        self.moving_average = moving_average
        self.pool = nn.AvgPool1d(kernel_size=moving_average, stride=1)

    def forward(self, inputs):
        padding = self.moving_average // 2
        padded = nn.functional.pad(inputs, (padding, padding), mode="replicate")
        trend = self.pool(padded)
        return inputs - trend, trend


class DLinearNetwork(nn.Module):
    """Decompose one input window and regress its centered appliance value."""

    def __init__(self, sequence_length, moving_average=25):
        super().__init__()
        validate_positive_int("sequence_length", sequence_length)
        validate_positive_int("moving_average", moving_average)
        if sequence_length % 2 == 0:
            raise ValueError("sequence_length must be odd.")
        if moving_average % 2 == 0:
            raise ValueError("moving_average must be odd to preserve alignment.")
        if moving_average > sequence_length:
            raise ValueError("moving_average must not exceed sequence_length.")

        self.sequence_length = sequence_length
        self.moving_average = moving_average
        self.decomposition = SeriesDecomposition(moving_average)
        self.seasonal_linear = nn.Linear(sequence_length, 1)
        self.trend_linear = nn.Linear(sequence_length, 1)
        self.reset_parameters()

    def reset_parameters(self):
        initial_weight = 1.0 / self.sequence_length
        nn.init.constant_(self.seasonal_linear.weight, initial_weight)
        nn.init.constant_(self.trend_linear.weight, initial_weight)
        nn.init.zeros_(self.seasonal_linear.bias)
        nn.init.zeros_(self.trend_linear.bias)

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("DLinearNetwork inputs must be a torch.Tensor.")
        expected_shape = (1, self.sequence_length)
        if inputs.ndim != 3 or inputs.shape[1:] != expected_shape:
            raise ValueError(
                "DLinearNetwork expects input shape "
                f"(batch, 1, {self.sequence_length}); got {tuple(inputs.shape)}."
            )
        if not torch.is_floating_point(inputs) or torch.is_complex(inputs):
            raise TypeError("DLinearNetwork inputs must be real floating tensors.")
        device = self.seasonal_linear.weight.device
        if inputs.device != device:
            raise ValueError(
                f"DLinearNetwork input is on {inputs.device}; model is on {device}."
            )
        if not torch.isfinite(inputs).all():
            raise ValueError("DLinearNetwork inputs must be finite.")

        inputs = inputs.to(dtype=self.seasonal_linear.weight.dtype)
        seasonal, trend = self.decomposition(inputs)
        output = self.seasonal_linear(seasonal.squeeze(1)) + self.trend_linear(
            trend.squeeze(1)
        )
        if not torch.isfinite(output).all():
            raise RuntimeError("DLinearNetwork produced non-finite output.")
        return output


class DLinear(SequenceToPointTorchDisaggregator):
    """DLinear adapted to one NILM estimate per centered mains row."""

    MODEL_NAME = "DLinear"
    CHECKPOINT_PREFIX = "dlinear"
    MODEL_CONFIG_FIELDS = (
        "moving_average",
        "learning_rate",
        "weight_decay",
        "validation_fraction",
        "validation_strategy",
        "gradient_clip_norm",
    )

    def __init__(self, params=None):
        super().__init__(
            params, defaults=torch_defaults(sequence_length=299, batch_size=128)
        )
        params = {} if params is None else params
        self.moving_average = validate_positive_int(
            "moving_average", get_param(params, "moving_average", 25)
        )
        self._validate_architecture()
        if self.load_model_path:
            self.load_model()

    def _validate_architecture(self):
        if self.moving_average % 2 == 0:
            raise ValueError("moving_average must be odd to preserve alignment.")
        if self.moving_average > self.sequence_length:
            raise ValueError("moving_average must not exceed sequence_length.")

    def return_network(self):
        return DLinearNetwork(
            sequence_length=self.sequence_length,
            moving_average=self.moving_average,
        ).to(self.device)


__all__ = ["DLinear", "DLinearNetwork", "SeriesDecomposition"]
