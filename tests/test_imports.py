import importlib
import json
import subprocess
import sys

import pytest

from nilmtk_contrib.utils.optional_imports import OptionalDependencyError, require_optional


BACKEND_MODULES = {"tensorflow", "torch", "cvxpy", "hmmlearn", "nilmtk", "pandas"}


def _imported_modules_after(statement):
    code = (
        "import json, sys\n"
        f"{statement}\n"
        f"print(json.dumps(sorted({BACKEND_MODULES!r}.intersection(sys.modules))))"
    )
    output = subprocess.check_output([sys.executable, "-c", code], text=True)
    return set(json.loads(output))


def test_top_level_import_is_lightweight():
    imported = _imported_modules_after("import nilmtk_contrib")
    assert imported == set()


def test_disaggregate_package_import_is_lightweight():
    imported = _imported_modules_after("import nilmtk_contrib.disaggregate")
    assert imported == set()


def test_torch_package_import_is_lightweight():
    imported = _imported_modules_after("import nilmtk_contrib.torch")
    assert imported == set()


def test_mains_stats_import_does_not_import_nilmtk():
    imported = _imported_modules_after("import nilmtk_contrib.mains_stats")
    assert imported == set()


def test_require_optional_error_message():
    with pytest.raises(OptionalDependencyError) as exc_info:
        require_optional(
            "definitely_missing_nilmtk_contrib_dependency",
            "dev",
            "Import test",
        )

    assert str(exc_info.value) == (
        "Import test requires 'definitely_missing_nilmtk_contrib_dependency'. "
        "Install nilmtk-contrib[dev]."
    )


@pytest.mark.parametrize(
    ("package_name", "class_name"),
    [
        ("nilmtk_contrib.disaggregate", "DAE"),
        ("nilmtk_contrib.disaggregate", "AFHMM"),
        ("nilmtk_contrib.torch", "DAE"),
    ],
)
def test_backend_exports_succeed_or_raise_optional_dependency_message(
    package_name,
    class_name,
):
    package = importlib.import_module(package_name)

    try:
        getattr(package, class_name)
    except OptionalDependencyError as exc:
        message = str(exc)
        assert f"{class_name} requires '" in message
        assert "Install nilmtk-contrib[" in message
    except ImportError as exc:
        pytest.fail(f"Unexpected non-optional import failure: {exc}")
