import logging
import random

import numpy as np

from nilmtk_contrib.utils.logging import configure_logging, get_logger
from nilmtk_contrib.utils.random import set_random_seed


def test_set_random_seed_controls_python_and_numpy():
    set_random_seed(123, backends=("python", "numpy"))
    first_python = random.random()
    first_numpy = np.random.rand()

    set_random_seed(123, backends=("python", "numpy"))
    second_python = random.random()
    second_numpy = np.random.rand()

    assert first_python == second_python
    assert first_numpy == second_numpy


def test_set_random_seed_ignores_none_seed():
    set_random_seed(None, backends=("python", "numpy"))


def test_get_logger_returns_named_logger():
    logger = get_logger("nilmtk_contrib.test")

    assert logger.name == "nilmtk_contrib.test"


def test_configure_logging_sets_expected_root_level():
    configure_logging(verbose=True)
    assert logging.getLogger().level <= logging.INFO

    configure_logging(verbose=False)
    assert logging.getLogger().level <= logging.WARNING
