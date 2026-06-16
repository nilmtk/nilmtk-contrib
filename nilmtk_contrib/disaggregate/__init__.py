"""Lazy exports for TensorFlow and classical NILMTK disaggregators.

These classes require optional backend dependencies. Importing this package does
not import TensorFlow, cvxpy, hmmlearn, or NILMTK until a class is requested.
"""

from importlib import import_module

from nilmtk_contrib.utils.optional_imports import OptionalDependencyError

_EXPORTS = {
    "AFHMM": ("nilmtk_contrib.disaggregate.afhmm", "classical", "AFHMM"),
    "AFHMM_SAC": ("nilmtk_contrib.disaggregate.afhmm_sac", "classical", "AFHMM_SAC"),
    "BERT": ("nilmtk_contrib.disaggregate.bert", "tensorflow", "BERT"),
    "DAE": ("nilmtk_contrib.disaggregate.dae", "tensorflow", "DAE"),
    "DSC": ("nilmtk_contrib.disaggregate.dsc", "classical", "DSC"),
    "RNN": ("nilmtk_contrib.disaggregate.rnn", "tensorflow", "RNN"),
    "RNN_attention": (
        "nilmtk_contrib.disaggregate.rnn_attention",
        "tensorflow",
        "RNN_attention",
    ),
    "RNN_attention_classification": (
        "nilmtk_contrib.disaggregate.rnn_attention_classification",
        "tensorflow",
        "RNN_attention_classification",
    ),
    "ResNet": ("nilmtk_contrib.disaggregate.resnet", "tensorflow", "ResNet"),
    "ResNet_classification": (
        "nilmtk_contrib.disaggregate.resnet_classification",
        "tensorflow",
        "ResNet_classification",
    ),
    "Seq2Point": ("nilmtk_contrib.disaggregate.seq2point", "tensorflow", "Seq2Point"),
    "Seq2Seq": ("nilmtk_contrib.disaggregate.seq2seq", "tensorflow", "Seq2Seq"),
    "WindowGRU": ("nilmtk_contrib.disaggregate.WindowGRU", "tensorflow", "WindowGRU"),
}

_DEPENDENCY_EXTRAS = {
    "cvxpy": "classical",
    "hmmlearn": "classical",
    "nilmtk": "nilm",
    "sklearn": "classical",
    "tensorflow": "tensorflow",
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

    module_name, extra_name, purpose = _EXPORTS[name]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_package = exc.name or "required dependency"
        install_extra = _DEPENDENCY_EXTRAS.get(missing_package, extra_name)
        message = (
            f"{purpose} requires '{missing_package}'. "
            f"Install nilmtk-contrib[{install_extra}]."
        )
        raise OptionalDependencyError(message) from exc

    value = getattr(module, name)
    globals()[name] = value
    return value
