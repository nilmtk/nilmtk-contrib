"""Compact neural emissions with exact semi-Markov likelihood and decoding."""

from __future__ import annotations

from numbers import Real
import math

import torch
from torch import nn

from nilmtk_contrib.torch._semi_markov import (
    semi_markov_nll,
    semi_markov_viterbi,
)
from nilmtk_contrib.utils.params import validate_positive_int


def _finite_probability(name, value, *, allow_zero=False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number.")
    minimum_invalid = value < 0 if allow_zero else value <= 0
    if minimum_invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}.")
    return float(value)


class TemporalResidualBlock(nn.Module):
    """Length-preserving dilated temporal residual block."""

    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        validate_positive_int("channels", channels)
        validate_positive_int("kernel_size", kernel_size)
        validate_positive_int("dilation", dilation)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve alignment.")
        dropout = _finite_probability("dropout", dropout, allow_zero=True)
        if dropout >= 1:
            raise ValueError("dropout must be less than one.")
        padding = dilation * (kernel_size - 1) // 2
        self.temporal = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.normalization = nn.GroupNorm(1, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, inputs):
        residual = inputs
        values = self.temporal(inputs)
        values = self.normalization(values)
        values = self.activation(values)
        values = self.dropout(values)
        return residual + self.projection(values)


class NeuralSemiMarkovNetwork(nn.Module):
    """TCN emission scorer with trainable exact semi-Markov structure."""

    def __init__(
        self,
        *,
        num_states=2,
        max_duration=99,
        hidden_channels=32,
        num_blocks=3,
        kernel_size=5,
        dropout=0.1,
    ):
        super().__init__()
        self.num_states = validate_positive_int("num_states", num_states)
        if self.num_states < 2:
            raise ValueError("num_states must be at least two.")
        self.max_duration = validate_positive_int("max_duration", max_duration)
        self.hidden_channels = validate_positive_int("hidden_channels", hidden_channels)
        self.num_blocks = validate_positive_int("num_blocks", num_blocks)
        self.kernel_size = validate_positive_int("kernel_size", kernel_size)
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve alignment.")
        self.dropout_probability = _finite_probability(
            "dropout", dropout, allow_zero=True
        )
        if self.dropout_probability >= 1:
            raise ValueError("dropout must be less than one.")

        padding = self.kernel_size // 2
        self.stem = nn.Conv1d(
            1,
            self.hidden_channels,
            self.kernel_size,
            padding=padding,
        )
        self.blocks = nn.ModuleList(
            TemporalResidualBlock(
                self.hidden_channels,
                self.kernel_size,
                dilation=2**index,
                dropout=self.dropout_probability,
            )
            for index in range(self.num_blocks)
        )
        self.emission_head = nn.Conv1d(
            self.hidden_channels, self.num_states, kernel_size=1
        )
        self.initial_logits = nn.Parameter(torch.zeros(self.num_states))
        self.transition_logits = nn.Parameter(
            torch.zeros((self.num_states, self.num_states))
        )
        self.duration_logits = nn.Parameter(
            torch.zeros((self.num_states, self.max_duration))
        )

    def _validate_inputs(self, inputs) -> None:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor.")
        if (
            inputs.ndim != 3
            or inputs.shape[0] < 1
            or inputs.shape[1] != 1
            or inputs.shape[2] < 1
        ):
            raise ValueError(
                "inputs must have shape (batch, 1, time) with batch/time >= 1."
            )
        if not torch.is_floating_point(inputs) or inputs.is_complex():
            raise TypeError("inputs must be real floating tensors.")
        if inputs.device != self.initial_logits.device:
            raise ValueError(
                f"inputs must be on model device {self.initial_logits.device}."
            )
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("inputs must contain only finite values.")

    def emission_scores(self, inputs) -> torch.Tensor:
        """Return contextual per-time state scores with shape ``(B, T, K)``."""
        self._validate_inputs(inputs)
        values = inputs.to(dtype=self.initial_logits.dtype)
        values = self.stem(values)
        for block in self.blocks:
            values = block(values)
        scores = self.emission_head(values).transpose(1, 2)
        if not bool(torch.isfinite(scores).all()):
            raise RuntimeError("Neural semi-Markov emissions became non-finite.")
        return scores

    def structure_scores(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized initial, off-diagonal transition, and duration logs."""
        initial = torch.log_softmax(self.initial_logits, dim=0)
        diagonal = torch.eye(
            self.num_states,
            dtype=torch.bool,
            device=self.transition_logits.device,
        )
        transitions = torch.log_softmax(
            self.transition_logits.masked_fill(diagonal, -torch.inf),
            dim=1,
        )
        durations = torch.log_softmax(self.duration_logits, dim=1)
        return initial, transitions, durations

    @torch.no_grad()
    def initialize_structure(
        self,
        initial_probabilities,
        transition_probabilities,
        duration_probabilities,
    ) -> None:
        """Initialize logits from a fitted classical semi-Markov model."""
        device = self.initial_logits.device
        dtype = self.initial_logits.dtype
        values = []
        for name, probabilities, shape in (
            ("initial_probabilities", initial_probabilities, (self.num_states,)),
            (
                "transition_probabilities",
                transition_probabilities,
                (self.num_states, self.num_states),
            ),
            (
                "duration_probabilities",
                duration_probabilities,
                (self.num_states, self.max_duration),
            ),
        ):
            try:
                tensor = torch.as_tensor(probabilities, dtype=dtype, device=device)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise TypeError(f"{name} must contain real numeric values.") from exc
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}.")
            if not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any()):
                raise ValueError(f"{name} must contain finite non-negative values.")
            values.append(tensor)
        initial, transitions, durations = values
        if bool((initial <= 0).any()) or not torch.isclose(
            initial.sum(), initial.new_tensor(1.0), atol=1e-6, rtol=1e-6
        ):
            raise ValueError("initial_probabilities must be positive and sum to one.")
        diagonal = torch.eye(self.num_states, dtype=torch.bool, device=device)
        if bool((transitions[diagonal] != 0).any()) or bool(
            (transitions[~diagonal] <= 0).any()
        ):
            raise ValueError(
                "transition_probabilities must have zero diagonal and positive "
                "off-diagonal values."
            )
        if not torch.allclose(
            transitions.sum(dim=1),
            torch.ones(self.num_states, dtype=dtype, device=device),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError("transition_probabilities rows must sum to one.")
        if bool((durations <= 0).any()) or not torch.allclose(
            durations.sum(dim=1),
            torch.ones(self.num_states, dtype=dtype, device=device),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError(
                "duration_probabilities must be positive and rows must sum to one."
            )
        self.initial_logits.copy_(torch.log(initial))
        self.transition_logits.copy_(
            torch.where(diagonal, torch.zeros_like(transitions), torch.log(transitions))
        )
        self.duration_logits.copy_(torch.log(durations))

    def negative_log_likelihood(self, inputs, states) -> torch.Tensor:
        """Return mean exact semi-Markov NLL for a labeled mini-batch."""
        emissions = self.emission_scores(inputs)
        if not isinstance(states, torch.Tensor):
            raise TypeError("states must be a torch.Tensor.")
        if states.dtype != torch.int64 or states.ndim != 2:
            raise TypeError("states must be a two-dimensional torch.int64 tensor.")
        if states.device != emissions.device:
            raise ValueError(f"states must be on {emissions.device}.")
        if tuple(states.shape) != emissions.shape[:2]:
            raise ValueError(f"states must have shape {tuple(emissions.shape[:2])}.")
        initial, transitions, durations = self.structure_scores()
        losses = [
            semi_markov_nll(
                labels,
                scores,
                initial,
                transitions,
                durations,
            )
            for scores, labels in zip(emissions, states)
        ]
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Neural semi-Markov loss became non-finite.")
        return loss

    @torch.no_grad()
    def decode(self, inputs):
        """Decode each sequence with the shared exact semi-Markov Viterbi core."""
        was_training = self.training
        self.eval()
        try:
            emissions = self.emission_scores(inputs)
            initial, transitions, durations = self.structure_scores()
            return tuple(
                semi_markov_viterbi(scores, initial, transitions, durations)
                for scores in emissions
            )
        finally:
            self.train(was_training)


__all__ = ["NeuralSemiMarkovNetwork", "TemporalResidualBlock"]
