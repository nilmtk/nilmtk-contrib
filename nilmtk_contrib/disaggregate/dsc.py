"""Compatibility path for the maintained PyTorch DSC implementation."""

from __future__ import annotations

from collections.abc import Mapping
import warnings

from nilmtk_contrib.torch.dsc import DSC as TorchDSC


class DSC(TorchDSC):
    """Deprecated import path retaining the historical DSC defaults."""

    def __init__(self, params=None):
        warnings.warn(
            "nilmtk_contrib.disaggregate.dsc.DSC now uses the maintained "
            "PyTorch implementation; import DSC from nilmtk_contrib.torch.",
            FutureWarning,
            stacklevel=2,
        )
        if params is None:
            normalized = {}
        elif isinstance(params, Mapping):
            normalized = dict(params)
        else:
            raise TypeError("params must be a mapping or None.")
        if (
            "discriminative_iterations" not in normalized
            and "iterations" not in normalized
        ):
            normalized["discriminative_iterations"] = 3_000
        super().__init__(normalized)
        self.n_epochs = self.discriminative_iterations

    @property
    def iterations(self):
        return super().iterations

    @iterations.setter
    def iterations(self, value):
        TorchDSC.iterations.fset(self, value)
        self.n_epochs = self.discriminative_iterations

    def load_model(self, path=None):
        super().load_model(path)
        self.n_epochs = self.discriminative_iterations


__all__ = ["DSC"]
