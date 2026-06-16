import importlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

import nilmtk_contrib.disaggregate as disaggregate_exports
import nilmtk_contrib.torch as torch_exports
import nilmtk_contrib.utils.optional_imports as optional_imports
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


def test_require_optional_returns_imported_module():
    module = require_optional("json", "dev", "Import test")

    assert module is json


def test_require_optional_reraises_nested_module_not_found(monkeypatch):
    def fake_import_module(package_name):
        raise ModuleNotFoundError("missing nested", name="nested_dependency")

    monkeypatch.setattr(optional_imports, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        require_optional("outer_package", "dev", "Nested import test")

    assert exc_info.value.name == "nested_dependency"


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


def test_lazy_exports_reject_unknown_attributes():
    with pytest.raises(AttributeError, match="NoSuchModel"):
        getattr(disaggregate_exports, "NoSuchModel")

    with pytest.raises(AttributeError, match="NoSuchModel"):
        getattr(torch_exports, "NoSuchModel")


def test_disaggregate_disaggregator_export_reports_missing_nilmtk(monkeypatch):
    def fake_import_module(module_name):
        assert module_name == "nilmtk.disaggregate"
        raise ModuleNotFoundError("missing nilmtk", name="nilmtk")

    disaggregate_exports.__dict__.pop("Disaggregator", None)
    monkeypatch.setattr(disaggregate_exports, "import_module", fake_import_module)

    with pytest.raises(OptionalDependencyError) as exc_info:
        disaggregate_exports.__getattr__("Disaggregator")

    assert str(exc_info.value) == (
        "Disaggregator requires 'nilmtk'. Install nilmtk-contrib[nilm]."
    )


def test_disaggregate_disaggregator_export_caches_loaded_value(monkeypatch):
    sentinel = object()

    def fake_import_module(module_name):
        assert module_name == "nilmtk.disaggregate"
        return SimpleNamespace(Disaggregator=sentinel)

    disaggregate_exports.__dict__.pop("Disaggregator", None)
    monkeypatch.setattr(disaggregate_exports, "import_module", fake_import_module)

    assert disaggregate_exports.__getattr__("Disaggregator") is sentinel
    assert disaggregate_exports.Disaggregator is sentinel
