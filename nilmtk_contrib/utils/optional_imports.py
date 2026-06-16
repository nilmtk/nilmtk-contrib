"""Helpers for optional backend dependencies."""

from importlib import import_module


class OptionalDependencyError(ImportError):
    """Raised when an optional backend dependency is required but missing."""


def require_optional(package_name, extra_name, purpose):
    """Import an optional package or raise an actionable install error."""
    try:
        return import_module(package_name)
    except ModuleNotFoundError as exc:
        if exc.name != package_name:
            raise
        message = (
            f"{purpose} requires '{package_name}'. "
            f"Install nilmtk-contrib[{extra_name}]."
        )
        raise OptionalDependencyError(message) from exc
