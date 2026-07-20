"""Small sequence-to-point baseline over inspectable window features."""

from collections.abc import Mapping

import torch
from torch import nn

from nilmtk_contrib.torch._base import torch_defaults
from nilmtk_contrib.torch._seq2point import (
    SequenceToPointTorchDisaggregator,
    finite_number,
)
from nilmtk_contrib.torch._window_features import WindowFeatureExtractor
from nilmtk_contrib.utils.params import get_param, validate_positive_int


class FeatureMLPNetwork(nn.Module):
    """Regress appliance power from fixed statistical and spectral features."""

    def __init__(self, sequence_length, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.extractor = WindowFeatureExtractor(sequence_length)
        self.sequence_length = self.extractor.sequence_length
        self.hidden_dim = validate_positive_int("hidden_dim", hidden_dim)
        self.dropout_rate = finite_number("dropout", dropout, minimum=0)
        if self.dropout_rate >= 1:
            raise ValueError("dropout must be less than 1.")

        feature_count = len(self.extractor.feature_names)
        self.feature_normalization = nn.LayerNorm(feature_count)
        self.regressor = nn.Sequential(
            nn.Linear(feature_count, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 1),
        )
        for layer in self.regressor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    @property
    def feature_names(self):
        return self.extractor.feature_names

    def extract_features(self, inputs):
        """Return the named feature matrix used by the predictor."""
        features = self.extractor(inputs)
        reference = self.feature_normalization.weight
        if features.device != reference.device:
            raise ValueError(
                f"FeatureMLPNetwork input is on {features.device}; model is on "
                f"{reference.device}."
            )
        return features.to(dtype=reference.dtype)

    def forward(self, inputs):
        features = self.extract_features(inputs)
        output = self.regressor(self.feature_normalization(features))
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("FeatureMLPNetwork produced non-finite output.")
        return output


class FeatureMLP(SequenceToPointTorchDisaggregator):
    """NILM baseline using fixed window descriptors and a small PyTorch MLP."""

    MODEL_NAME = "FeatureMLP"
    CHECKPOINT_PREFIX = "feature-mlp"
    MODEL_CONFIG_FIELDS = (
        "hidden_dim",
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
        self.hidden_dim = validate_positive_int(
            "hidden_dim", get_param(params, "hidden_dim", 32)
        )
        self.dropout = finite_number(
            "dropout", get_param(params, "dropout", 0.1), minimum=0
        )
        if self.sequence_length < 9:
            raise ValueError("sequence_length must be at least 9.")
        if self.dropout >= 1:
            raise ValueError("dropout must be less than 1.")
        if self.load_model_path:
            self.load_model()

    def return_network(self):
        return FeatureMLPNetwork(
            sequence_length=self.sequence_length,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)


__all__ = ["FeatureMLP", "FeatureMLPNetwork"]
