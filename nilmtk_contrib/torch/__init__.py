"""Lazy exports for PyTorch NILMTK disaggregators."""

from importlib import import_module

from nilmtk_contrib.utils.optional_imports import OptionalDependencyError

_EXPORTS = {
    "BERT": ("nilmtk_contrib.torch.bert", "BERT"),
    "ConvLSTM": ("nilmtk_contrib.torch.conv_lstm", "ConvLSTM"),
    "DAE": ("nilmtk_contrib.torch.dae", "DAE"),
    "MSDC": ("nilmtk_contrib.torch.msdc", "MSDC"),
    "NILMFormer": ("nilmtk_contrib.torch.nilmformer", "NILMFormer"),
    "Reformer": ("nilmtk_contrib.torch.reformer", "Reformer"),
    "ResNet": ("nilmtk_contrib.torch.resnet", "ResNet"),
    "ResNet_classification": (
        "nilmtk_contrib.torch.resnet_classification",
        "ResNet_classification",
    ),
    "RNN": ("nilmtk_contrib.torch.rnn", "RNN"),
    "RNN_attention": ("nilmtk_contrib.torch.rnn_attention", "RNN_attention"),
    "RNN_attention_classification": (
        "nilmtk_contrib.torch.rnn_attention_classification",
        "RNN_attention_classification",
    ),
    "Seq2PointTorch": ("nilmtk_contrib.torch.seq2point", "Seq2PointTorch"),
    "Seq2Seq": ("nilmtk_contrib.torch.seq2seq", "Seq2Seq"),
    "TCN": ("nilmtk_contrib.torch.TCN", "TCN"),
    "WindowGRU": ("nilmtk_contrib.torch.WindowGRU", "WindowGRU"),
}

_DEPENDENCY_EXTRAS = {
    "nilmtk": "nilm",
    "sklearn": "classical",
    "torch": "torch",
    "tqdm": "torch",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, class_name = _EXPORTS[name]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_package = exc.name or "required dependency"
        install_extra = _DEPENDENCY_EXTRAS.get(missing_package, "torch")
        message = (
            f"{name} requires '{missing_package}'. "
            f"Install nilmtk-contrib[{install_extra}]."
        )
        raise OptionalDependencyError(message) from exc

    value = getattr(module, class_name)
    globals()[name] = value
    return value
