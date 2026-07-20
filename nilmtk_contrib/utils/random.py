"""Random seed helpers."""

import random

_SUPPORTED_BACKENDS = frozenset({"python", "numpy", "torch"})


def set_random_seed(seed, backends=("python", "numpy", "torch")):
    """Set random seeds for selected backends when they are installed.

    This does not force deterministic backend modes because those can have
    significant performance and operator-availability tradeoffs.
    """
    requested_backends = tuple(backends)
    unsupported = set(requested_backends).difference(_SUPPORTED_BACKENDS)
    if unsupported:
        names = ", ".join(sorted(str(name) for name in unsupported))
        raise ValueError(f"Unsupported random-seed backend(s): {names}.")

    if seed is None:
        return

    if "python" in requested_backends:
        random.seed(seed)

    if "numpy" in requested_backends:
        try:
            import numpy as np
        except ModuleNotFoundError:
            pass
        else:
            np.random.seed(seed)

    if "torch" in requested_backends:
        try:
            import torch
        except ModuleNotFoundError:
            pass
        else:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
