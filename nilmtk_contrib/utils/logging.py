"""Logging helpers."""

import logging


def get_logger(name):
    """Return a package logger without configuring global logging."""
    return logging.getLogger(name)


def log_print(logger, *args, **kwargs):
    """Compatibility replacement for legacy print calls."""
    if kwargs.get("file") is not None:
        return
    sep = kwargs.get("sep", " ")
    message = sep.join(str(arg) for arg in args)
    logger.info(message)


def configure_logging(verbose=False):
    """Configure basic logging for scripts or notebooks that opt in."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level)
    logging.getLogger().setLevel(level)
