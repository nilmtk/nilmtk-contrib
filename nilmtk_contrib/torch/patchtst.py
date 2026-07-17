"""PatchTST-inspired sequence-to-point energy disaggregator."""

import torch
from torch import nn

from nilmtk_contrib.torch._base import torch_defaults
from nilmtk_contrib.torch._seq2point import (
    SequenceToPointTorchDisaggregator,
    finite_number,
)
from nilmtk_contrib.utils.params import get_param, validate_positive_int


class PatchTSTNetwork(nn.Module):
    """Tokenize overlapping temporal patches, then regress the center point."""

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
        dropout = finite_number("dropout", dropout, minimum=0)
        if dropout >= 1:
            raise ValueError("dropout must be less than 1.")

        self.sequence_length = sequence_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.num_patches = (sequence_length - patch_length) // patch_stride + 2
        self.patch_projection = nn.Linear(patch_length, d_model)
        self.position = nn.Parameter(torch.empty(1, self.num_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(self.num_patches * d_model),
            nn.Linear(self.num_patches * d_model, 1),
        )
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("PatchTSTNetwork inputs must be a torch.Tensor.")
        expected_shape = (1, self.sequence_length)
        if inputs.ndim != 3 or inputs.shape[1:] != expected_shape:
            raise ValueError(
                "PatchTSTNetwork expects input shape "
                f"(batch, 1, {self.sequence_length}); got {tuple(inputs.shape)}."
            )
        if not torch.is_floating_point(inputs) or torch.is_complex(inputs):
            raise TypeError("PatchTSTNetwork inputs must be real floating tensors.")
        device = self.patch_projection.weight.device
        if inputs.device != device:
            raise ValueError(
                f"PatchTSTNetwork input is on {inputs.device}; model is on {device}."
            )
        if not torch.isfinite(inputs).all():
            raise ValueError("PatchTSTNetwork inputs must be finite.")

        inputs = inputs.to(dtype=self.patch_projection.weight.dtype)
        padded = nn.functional.pad(inputs, (0, self.patch_stride), mode="replicate")
        patches = padded.unfold(-1, self.patch_length, self.patch_stride)
        tokens = self.patch_projection(patches.squeeze(1)) + self.position
        encoded = self.encoder(tokens)
        output = self.head(encoded.flatten(start_dim=1))
        if not torch.isfinite(output).all():
            raise RuntimeError("PatchTSTNetwork produced non-finite output.")
        return output


class PatchTST(SequenceToPointTorchDisaggregator):
    """PatchTST patch encoder adapted to one NILM estimate per mains row."""

    MODEL_NAME = "PatchTST"
    CHECKPOINT_PREFIX = "patchtst"
    MODEL_CONFIG_FIELDS = (
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

    def __init__(self, params=None):
        super().__init__(params, defaults=torch_defaults(batch_size=128))
        params = {} if params is None else params
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
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.dropout >= 1:
            raise ValueError("dropout must be less than 1.")

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


__all__ = ["PatchTST", "PatchTSTNetwork"]
