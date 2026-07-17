"""TimesNet-inspired sequence-to-point energy disaggregator."""

from collections.abc import Mapping

import torch
from torch import nn

from nilmtk_contrib.torch._base import torch_defaults
from nilmtk_contrib.torch._seq2point import (
    SequenceToPointTorchDisaggregator,
    finite_number,
)
from nilmtk_contrib.utils.params import get_param, validate_positive_int


def dominant_periods(inputs, top_k):
    """Return FFT-selected periods and per-sample frequency amplitudes."""
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("dominant_periods inputs must be a torch.Tensor.")
    if inputs.ndim != 3 or inputs.shape[1] < 3:
        raise ValueError("dominant_periods expects shape (batch, time>=3, channels).")
    if not torch.is_floating_point(inputs) or torch.is_complex(inputs):
        raise TypeError("dominant_periods inputs must be real floating tensors.")
    if not torch.isfinite(inputs).all():
        raise ValueError("dominant_periods inputs must be finite.")
    top_k = validate_positive_int("top_k", top_k)
    available = inputs.shape[1] // 2
    if top_k > available:
        raise ValueError(f"top_k must not exceed {available} for this sequence.")

    spectrum = torch.fft.rfft(inputs, dim=1)
    mean_amplitude = spectrum.abs().mean(dim=(0, 2))
    frequency_indices = torch.topk(mean_amplitude[1:], top_k).indices.add(1).detach()
    periods = torch.div(
        inputs.shape[1], frequency_indices, rounding_mode="floor"
    ).clamp_min(1)
    sample_amplitudes = spectrum.abs().mean(dim=-1)[:, frequency_indices]
    return periods, sample_amplitudes


class InceptionBlock2D(nn.Module):
    """Average parallel odd 2D kernels without changing the temporal grid."""

    def __init__(self, in_channels, out_channels, num_kernels):
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("num_kernels", num_kernels),
        ):
            validate_positive_int(name, value)
        self.kernels = nn.ModuleList(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=2 * index + 1,
                padding=index,
            )
            for index in range(num_kernels)
        )
        for kernel in self.kernels:
            nn.init.kaiming_normal_(kernel.weight, nonlinearity="relu")
            nn.init.zeros_(kernel.bias)

    def forward(self, inputs):
        return torch.stack([kernel(inputs) for kernel in self.kernels], dim=-1).mean(
            dim=-1
        )


class TimesBlock(nn.Module):
    """Model adaptive intra-period and inter-period variation in 2D."""

    def __init__(self, sequence_length, d_model, d_ff, top_k, num_kernels):
        super().__init__()
        self.sequence_length = validate_positive_int("sequence_length", sequence_length)
        self.top_k = validate_positive_int("top_k", top_k)
        if self.top_k > self.sequence_length // 2:
            raise ValueError("top_k exceeds the non-DC frequency count.")
        self.convolution = nn.Sequential(
            InceptionBlock2D(d_model, d_ff, num_kernels),
            nn.GELU(),
            InceptionBlock2D(d_ff, d_model, num_kernels),
        )

    def forward(self, inputs):
        periods, amplitudes = dominant_periods(inputs, self.top_k)
        batch, length, channels = inputs.shape
        variations = []
        for period_value in periods:
            period = int(period_value.item())
            padded_length = ((length + period - 1) // period) * period
            if padded_length != length:
                padding = inputs.new_zeros(batch, padded_length - length, channels)
                values = torch.cat((inputs, padding), dim=1)
            else:
                values = inputs
            grid = values.reshape(batch, padded_length // period, period, channels)
            grid = grid.permute(0, 3, 1, 2).contiguous()
            encoded = self.convolution(grid)
            encoded = encoded.permute(0, 2, 3, 1).reshape(
                batch, padded_length, channels
            )
            variations.append(encoded[:, :length])
        stacked = torch.stack(variations, dim=-1)
        weights = torch.softmax(amplitudes, dim=1)[:, None, None, :]
        return inputs + torch.sum(stacked * weights, dim=-1)


class TimesNetNetwork(nn.Module):
    """TimesNet backbone with a centered scalar NILM regression head."""

    def __init__(
        self,
        sequence_length,
        d_model=32,
        d_ff=64,
        n_blocks=2,
        top_k=3,
        num_kernels=3,
        dropout=0.1,
    ):
        super().__init__()
        for name, value in (
            ("sequence_length", sequence_length),
            ("d_model", d_model),
            ("d_ff", d_ff),
            ("n_blocks", n_blocks),
            ("top_k", top_k),
            ("num_kernels", num_kernels),
        ):
            validate_positive_int(name, value)
        if sequence_length % 2 == 0:
            raise ValueError("sequence_length must be odd.")
        if sequence_length < 3:
            raise ValueError("sequence_length must be at least 3.")
        if top_k > sequence_length // 2:
            raise ValueError("top_k exceeds the non-DC frequency count.")
        dropout = finite_number("dropout", dropout, minimum=0)
        if dropout >= 1:
            raise ValueError("dropout must be less than 1.")

        self.sequence_length = sequence_length
        self.embedding = nn.Conv1d(
            1, d_model, kernel_size=3, padding=1, padding_mode="circular"
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            TimesBlock(sequence_length, d_model, d_ff, top_k, num_kernels)
            for _ in range(n_blocks)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_blocks))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * d_model, 1))
        nn.init.kaiming_normal_(self.embedding.weight, nonlinearity="linear")
        nn.init.zeros_(self.embedding.bias)
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("TimesNetNetwork inputs must be a torch.Tensor.")
        expected_shape = (1, self.sequence_length)
        if inputs.ndim != 3 or inputs.shape[1:] != expected_shape:
            raise ValueError(
                "TimesNetNetwork expects input shape "
                f"(batch, 1, {self.sequence_length}); got {tuple(inputs.shape)}."
            )
        if not torch.is_floating_point(inputs) or torch.is_complex(inputs):
            raise TypeError("TimesNetNetwork inputs must be real floating tensors.")
        device = self.embedding.weight.device
        if inputs.device != device:
            raise ValueError(
                f"TimesNetNetwork input is on {inputs.device}; model is on {device}."
            )
        if not torch.isfinite(inputs).all():
            raise ValueError("TimesNetNetwork inputs must be finite.")

        inputs = inputs.to(dtype=self.embedding.weight.dtype)
        encoded = self.embedding_dropout(self.embedding(inputs).transpose(1, 2))
        for block, norm in zip(self.blocks, self.norms, strict=True):
            encoded = norm(block(encoded))
        center = encoded[:, self.sequence_length // 2]
        pooled = encoded.mean(dim=1)
        output = self.head(torch.cat((center, pooled), dim=1))
        if not torch.isfinite(output).all():
            raise RuntimeError("TimesNetNetwork produced non-finite output.")
        return output


class TimesNet(SequenceToPointTorchDisaggregator):
    """TimesNet adapted to one NILM estimate per centered mains row."""

    MODEL_NAME = "TimesNet"
    CHECKPOINT_PREFIX = "timesnet"
    MODEL_CONFIG_FIELDS = (
        "d_model",
        "d_ff",
        "n_blocks",
        "top_k",
        "num_kernels",
        "dropout",
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
        self.d_model = validate_positive_int(
            "d_model", get_param(params, "d_model", 32)
        )
        self.d_ff = validate_positive_int("d_ff", get_param(params, "d_ff", 64))
        self.n_blocks = validate_positive_int(
            "n_blocks", get_param(params, "n_blocks", 2)
        )
        self.top_k = validate_positive_int("top_k", get_param(params, "top_k", 3))
        self.num_kernels = validate_positive_int(
            "num_kernels", get_param(params, "num_kernels", 3)
        )
        self.dropout = finite_number(
            "dropout", get_param(params, "dropout", 0.1), minimum=0
        )
        self._validate_architecture()
        if self.load_model_path:
            self.load_model()

    def _validate_architecture(self):
        if self.sequence_length < 3:
            raise ValueError("sequence_length must be at least 3.")
        if self.top_k > self.sequence_length // 2:
            raise ValueError("top_k exceeds the non-DC frequency count.")
        if self.dropout >= 1:
            raise ValueError("dropout must be less than 1.")

    def return_network(self):
        return TimesNetNetwork(
            sequence_length=self.sequence_length,
            d_model=self.d_model,
            d_ff=self.d_ff,
            n_blocks=self.n_blocks,
            top_k=self.top_k,
            num_kernels=self.num_kernels,
            dropout=self.dropout,
        ).to(self.device)


__all__ = [
    "InceptionBlock2D",
    "TimesBlock",
    "TimesNet",
    "TimesNetNetwork",
    "dominant_periods",
]
