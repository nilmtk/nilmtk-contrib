"""Shared model-level migration helpers."""

from types import MethodType

from nilmtk_contrib.utils.checkpoints import managed_checkpoint_path, unsupported_persistence
from nilmtk_contrib.utils.logging import configure_logging, get_logger, log_print
from nilmtk_contrib.utils.random import set_random_seed


def _unsupported_save_model(self, *args, **kwargs):
    model_name = getattr(self, "MODEL_NAME", self.__class__.__name__)
    unsupported_persistence(model_name)


def _unsupported_load_model(self, *args, **kwargs):
    model_name = getattr(self, "MODEL_NAME", self.__class__.__name__)
    unsupported_persistence(model_name)


def initialize_runtime(model, params, *, backends):
    """Attach common runtime controls to a model instance."""
    model.seed = params.get("seed", getattr(model, "seed", None))
    model.verbose = params.get("verbose", getattr(model, "verbose", False))
    configure_logging(model.verbose)
    set_random_seed(model.seed, backends=backends)
    if not callable(getattr(model, "save_model", None)):
        model.save_model = MethodType(_unsupported_save_model, model)
    if not callable(getattr(model, "load_model", None)):
        model.load_model = MethodType(_unsupported_load_model, model)


def module_logger(name):
    """Return a logger for model modules."""
    return get_logger(name)


def legacy_print(logger):
    """Return a quiet-by-default print replacement bound to a logger."""

    def _print(*args, **kwargs):
        log_print(logger, *args, **kwargs)

    return _print


def checkpoint_path(suffix):
    """Return a temporary checkpoint path managed for the process lifetime."""
    return str(managed_checkpoint_path(suffix))
