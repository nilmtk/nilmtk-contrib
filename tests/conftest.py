from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_addoption(parser):
    group = parser.getgroup("nilmtk-contrib smoke")
    group.addoption(
        "--run-model-smoke",
        action="store_true",
        default=False,
        help="Run optional per-model synthetic training/disaggregation smoke tests.",
    )
    group.addoption(
        "--model-smoke-backend",
        action="append",
        choices=("classical", "tensorflow", "torch"),
        default=[],
        help=(
            "Limit --run-model-smoke to one or more backends. "
            "Defaults to all backends."
        ),
    )
    group.addoption(
        "--model-smoke-epochs",
        type=int,
        default=2,
        help="Epoch count for neural model smoke tests.",
    )
    group.addoption(
        "--real-dataset-path",
        default=None,
        help="Optional NILMTK-compatible HDF5 dataset path, for example ukdale.h5.",
    )
    group.addoption(
        "--real-dataset-building",
        type=int,
        default=1,
        help="Building id used by optional real-dataset smoke tests.",
    )
    group.addoption(
        "--real-dataset-appliance",
        default="fridge",
        help="Appliance name used by optional real-dataset smoke tests.",
    )
    group.addoption(
        "--run-gpu-tests",
        action="store_true",
        default=False,
        help="Run optional GPU availability and backend placement tests.",
    )


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def model_smoke_config(pytestconfig):
    backends = set(pytestconfig.getoption("--model-smoke-backend") or [])
    return {
        "enabled": pytestconfig.getoption("--run-model-smoke"),
        "backends": backends or {"classical", "tensorflow", "torch"},
        "epochs": pytestconfig.getoption("--model-smoke-epochs"),
        "real_dataset_path": pytestconfig.getoption("--real-dataset-path"),
        "real_dataset_building": pytestconfig.getoption("--real-dataset-building"),
        "real_dataset_appliance": pytestconfig.getoption("--real-dataset-appliance"),
    }


@pytest.fixture(scope="session")
def gpu_tests_enabled(pytestconfig):
    return pytestconfig.getoption("--run-gpu-tests")
