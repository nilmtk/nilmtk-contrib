"""Compact ModernTCN-inspired sequence-to-point energy disaggregator.

This original NILM adaptation follows ModernTCN's patching, parallel
large/small depthwise kernels, channel mixing, and residual design. It predicts
the appliance value at the center of each aggregate-power window rather than a
future value of the input series.
"""

from collections.abc import Mapping

import torch
from torch import nn

from nilmtk_contrib.torch._base import torch_defaults
from nilmtk_contrib.torch._seq2point import (
    SequenceToPointTorchDisaggregator,
    finite_number,
)
from nilmtk_contrib.utils.params import get_param, validate_positive_int


class ModernTCNBlock(nn.Module):
    """Residual multi-scale depthwise temporal convolution and channel mixer."""

    def __init__(
        self,
        d_model,
        d_ff,
        large_kernel_size,
        small_kernel_size,
        dropout,
    ):
        super().__init__()
        self.large_kernel = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=large_kernel_size,
            padding=large_kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        self.small_kernel = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=small_kernel_size,
            padding=small_kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(d_model, d_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, inputs):
        mixed = self.large_kernel(inputs) + self.small_kernel(inputs)
        mixed = self.norm(mixed.transpose(1, 2)).transpose(1, 2)
        return inputs + self.channel_mixer(mixed)


class ModernTCNNetwork(nn.Module):
    """One-stage ModernTCN backbone with a centered scalar regression head."""

    def __init__(
        self,
        sequence_length,
        patch_length=16,
        patch_stride=8,
        d_model=64,
        d_ff=128,
        n_blocks=2,
        large_kernel_size=51,
        small_kernel_size=5,
        dropout=0.1,
    ):
        super().__init__()
        for name, value in (
            ("sequence_length", sequence_length),
            ("patch_length", patch_length),
            ("patch_stride", patch_stride),
            ("d_model", d_model),
            ("d_ff", d_ff),
            ("n_blocks", n_blocks),
            ("large_kernel_size", large_kernel_size),
            ("small_kernel_size", small_kernel_size),
        ):
            validate_positive_int(name, value)
        if sequence_length % 2 == 0:
            raise ValueError("sequence_length must be odd.")
        if patch_length > sequence_length:
            raise ValueError("patch_length must not exceed sequence_length.")
        if patch_stride > patch_length:
            raise ValueError("patch_stride must not exceed patch_length.")
        for name, value in (
            ("large_kernel_size", large_kernel_size),
            ("small_kernel_size", small_kernel_size),
        ):
            if value % 2 == 0:
                raise ValueError(f"{name} must be odd to preserve token alignment.")
        if small_kernel_size > large_kernel_size:
            raise ValueError("small_kernel_size must not exceed large_kernel_size.")
        dropout = finite_number("dropout", dropout, minimum=0)
        if dropout >= 1:
            raise ValueError("dropout must be less than 1.")

        self.sequence_length = sequence_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        remainder = (sequence_length - patch_length) % patch_stride
        self.end_padding = (patch_stride - remainder) % patch_stride
        padded_length = sequence_length + self.end_padding
        self.num_patches = (padded_length - patch_length) // patch_stride + 1
        patch_centers = (
            torch.arange(self.num_patches, dtype=torch.float64) * patch_stride
            + (patch_length - 1) / 2
        )
        sequence_center = (sequence_length - 1) / 2
        self.center_patch_index = int(
            torch.argmin(torch.abs(patch_centers - sequence_center)).item()
        )

        self.patch_embedding = nn.Conv1d(
            1, d_model, kernel_size=patch_length, stride=patch_stride
        )
        self.blocks = nn.Sequential(
            *[
                ModernTCNBlock(
                    d_model,
                    d_ff,
                    large_kernel_size,
                    small_kernel_size,
                    dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * d_model, 1))
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("ModernTCNNetwork inputs must be a torch.Tensor.")
        expected_shape = (1, self.sequence_length)
        if inputs.ndim != 3 or inputs.shape[1:] != expected_shape:
            raise ValueError(
                "ModernTCNNetwork expects input shape "
                f"(batch, 1, {self.sequence_length}); got {tuple(inputs.shape)}."
            )
        if not torch.is_floating_point(inputs) or torch.is_complex(inputs):
            raise TypeError("ModernTCNNetwork inputs must be real floating tensors.")
        device = self.patch_embedding.weight.device
        if inputs.device != device:
            raise ValueError(
                f"ModernTCNNetwork input is on {inputs.device}; model is on {device}."
            )
        if not torch.isfinite(inputs).all():
            raise ValueError("ModernTCNNetwork inputs must be finite.")

        inputs = inputs.to(dtype=self.patch_embedding.weight.dtype)
        if self.end_padding:
            inputs = nn.functional.pad(inputs, (0, self.end_padding), mode="replicate")
        encoded = self.blocks(self.patch_embedding(inputs))
        encoded = self.final_norm(encoded.transpose(1, 2)).transpose(1, 2)
        center = encoded[:, :, self.center_patch_index]
        pooled = encoded.mean(dim=-1)
        output = self.head(torch.cat((center, pooled), dim=1))
        if not torch.isfinite(output).all():
            raise RuntimeError("ModernTCNNetwork produced non-finite output.")
        return output


class ModernTCN(SequenceToPointTorchDisaggregator):
    """ModernTCN adapted to one NILM estimate per centered mains row."""

    MODEL_NAME = "ModernTCN"
    CHECKPOINT_PREFIX = "moderntcn"
    MODEL_CONFIG_FIELDS = (
        "patch_length",
        "patch_stride",
        "d_model",
        "d_ff",
        "n_blocks",
        "large_kernel_size",
        "small_kernel_size",
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
        self.patch_length = validate_positive_int(
            "patch_length", get_param(params, "patch_length", 16)
        )
        self.patch_stride = validate_positive_int(
            "patch_stride", get_param(params, "patch_stride", 8)
        )
        self.d_model = validate_positive_int(
            "d_model", get_param(params, "d_model", 64)
        )
        self.d_ff = validate_positive_int("d_ff", get_param(params, "d_ff", 128))
        self.n_blocks = validate_positive_int(
            "n_blocks", get_param(params, "n_blocks", 2)
        )
        self.large_kernel_size = validate_positive_int(
            "large_kernel_size", get_param(params, "large_kernel_size", 51)
        )
        self.small_kernel_size = validate_positive_int(
            "small_kernel_size", get_param(params, "small_kernel_size", 5)
        )
        self.dropout = finite_number(
            "dropout", get_param(params, "dropout", 0.1), minimum=0
        )
        self._validate_architecture()
        if self.load_model_path:
            self.load_model()

    def _validate_architecture(self):
        if self.patch_length > self.sequence_length:
            raise ValueError("patch_length must not exceed sequence_length.")
        if self.patch_stride > self.patch_length:
            raise ValueError("patch_stride must not exceed patch_length.")
        for name in ("large_kernel_size", "small_kernel_size"):
            if getattr(self, name) % 2 == 0:
                raise ValueError(f"{name} must be odd to preserve token alignment.")
        if self.small_kernel_size > self.large_kernel_size:
            raise ValueError("small_kernel_size must not exceed large_kernel_size.")
        if self.dropout >= 1:
            raise ValueError("dropout must be less than 1.")

    def return_network(self):
        return ModernTCNNetwork(
            sequence_length=self.sequence_length,
            patch_length=self.patch_length,
            patch_stride=self.patch_stride,
            d_model=self.d_model,
            d_ff=self.d_ff,
            n_blocks=self.n_blocks,
            large_kernel_size=self.large_kernel_size,
            small_kernel_size=self.small_kernel_size,
            dropout=self.dropout,
        ).to(self.device)


__all__ = ["ModernTCN", "ModernTCNBlock", "ModernTCNNetwork"]
