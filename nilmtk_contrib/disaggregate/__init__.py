"""Compatibility exports for historical and classical NILMTK disaggregators.

These classes require optional backend dependencies. Importing this package does
not import PyTorch, cvxpy, hmmlearn, or NILMTK until a class is requested.

Historical package-level neural exports resolve to the maintained PyTorch
implementations. The former TensorFlow implementation modules have been
retired.
"""

import warnings
from importlib import import_module

from nilmtk_contrib.utils.optional_imports import OptionalDependencyError

_EXPORTS = {
    "AFHMM": ("nilmtk_contrib.disaggregate.afhmm", "AFHMM", "classical"),
    "AFHMM_SAC": (
        "nilmtk_contrib.disaggregate.afhmm_sac",
        "AFHMM_SAC",
        "classical",
    ),
    "BERT": ("nilmtk_contrib.torch.bert", "BERT", "torch"),
    "DAE": ("nilmtk_contrib.torch.dae", "DAE", "torch"),
    "DSC": ("nilmtk_contrib.disaggregate.dsc", "DSC", "torch"),
    "RNN": ("nilmtk_contrib.torch.rnn", "RNN", "torch"),
    "RNN_attention": (
        "nilmtk_contrib.torch.rnn_attention",
        "RNN_attention",
        "torch",
    ),
    "RNN_attention_classification": (
        "nilmtk_contrib.torch.rnn_attention_classification",
        "RNN_attention_classification",
        "torch",
    ),
    "ResNet": ("nilmtk_contrib.torch.resnet", "ResNet", "torch"),
    "ResNet_classification": (
        "nilmtk_contrib.torch.resnet_classification",
        "ResNet_classification",
        "torch",
    ),
    "Seq2Point": ("nilmtk_contrib.torch.seq2point", "Seq2PointTorch", "torch"),
    "Seq2Seq": ("nilmtk_contrib.torch.seq2seq", "Seq2Seq", "torch"),
    "WindowGRU": ("nilmtk_contrib.torch.WindowGRU", "WindowGRU", "torch"),
}

_TORCH_REDIRECTS = {
    "BERT",
    "DAE",
    "RNN",
    "RNN_attention",
    "RNN_attention_classification",
    "ResNet",
    "ResNet_classification",
    "Seq2Point",
    "Seq2Seq",
    "WindowGRU",
}

_DEPENDENCY_EXTRAS = {
    "cvxpy": "classical",
    "hmmlearn": "classical",
    "nilmtk": "nilm",
    "sklearn": "classical",
    "torch": "torch",
    "tqdm": "torch",
}

__all__ = sorted([*_EXPORTS, "Disaggregator"])


def __getattr__(name):
    if name == "Disaggregator":
        try:
            module = import_module("nilmtk.disaggregate")
        except ModuleNotFoundError as exc:
            message = (
                "Disaggregator requires 'nilmtk'. "
                "Install nilmtk-contrib[nilm]."
            )
            raise OptionalDependencyError(message) from exc
        value = module.Disaggregator
        globals()[name] = value
        return value

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, class_name, extra_name = _EXPORTS[name]
    if name in _TORCH_REDIRECTS:
        warnings.warn(
            f"nilmtk_contrib.disaggregate.{name} now resolves to its maintained "
            f"PyTorch implementation; import {class_name} from "
            "nilmtk_contrib.torch.",
            FutureWarning,
            stacklevel=2,
        )
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_package = exc.name or "required dependency"
        install_extra = _DEPENDENCY_EXTRAS.get(missing_package, extra_name)
        message = (
            f"{name} requires '{missing_package}'. "
            f"Install nilmtk-contrib[{install_extra}]."
        )
        raise OptionalDependencyError(message) from exc

    value = getattr(module, class_name)
    globals()[name] = value
    return value
