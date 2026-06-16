"""Random seed helpers."""

import random


def set_random_seed(seed, backends=("python", "numpy", "torch", "tensorflow")):
    """Set random seeds for selected backends when they are installed.

    This does not force deterministic backend modes because those can have
    significant performance and operator-availability tradeoffs.
    """
    if seed is None:
        return

    if "python" in backends:
        random.seed(seed)

    if "numpy" in backends:
        try:
            import numpy as np
        except ModuleNotFoundError:
            pass
        else:
            np.random.seed(seed)

    if "torch" in backends:
        try:
            import torch
        except ModuleNotFoundError:
            pass
        else:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    if "tensorflow" in backends:
        try:
            import tensorflow as tf
        except ModuleNotFoundError:
            pass
        else:
            tf.random.set_seed(seed)
