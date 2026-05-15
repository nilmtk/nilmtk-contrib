"""Logging helpers."""

import logging


def get_logger(name):
    """Return a package logger without configuring global logging."""
    return logging.getLogger(name)


def configure_logging(verbose=False):
    """Configure basic logging for scripts or notebooks that opt in."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level)
    logging.getLogger().setLevel(level)
