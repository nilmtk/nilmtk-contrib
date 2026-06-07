import importlib
import importlib.util

import pytest

from nilmtk_contrib.utils.optional_imports import OptionalDependencyError

from model_smoke_helpers import (
    ALL_MODEL_SPECS,
    DISAGGREGATE_MODEL_SPECS,
    TORCH_MODEL_SPECS,
)


def test_disaggregate_registry_matches_comprehensive_model_specs():
    package = importlib.import_module("nilmtk_contrib.disaggregate")
    expected = {spec.class_name for spec in DISAGGREGATE_MODEL_SPECS}
    actual = set(package._EXPORTS)
    assert actual == expected
    assert set(package.__all__) == expected | {"Disaggregator"}


def test_torch_registry_matches_comprehensive_model_specs():
    package = importlib.import_module("nilmtk_contrib.torch")
    expected = {
        spec.class_name
        for spec in TORCH_MODEL_SPECS
        if spec.module_name != "nilmtk_contrib.torch.msdc_without_crf"
    }
    actual = set(package._EXPORTS)
    assert actual == expected
    assert set(package.__all__) == expected


def test_msdc_without_crf_module_is_covered_even_though_not_exported():
    spec = next(
        item
        for item in TORCH_MODEL_SPECS
        if item.module_name == "nilmtk_contrib.torch.msdc_without_crf"
    )
    assert spec.class_name == "MSDC"
    assert spec.module_name.endswith("msdc_without_crf")


@pytest.mark.parametrize(
    "spec",
    ALL_MODEL_SPECS,
    ids=lambda spec: f"{spec.backend}:{spec.module_name}.{spec.class_name}",
)
def test_model_class_access_succeeds_or_reports_optional_dependency(spec):
    package = importlib.import_module(spec.package)

    try:
        if spec.module_name == "nilmtk_contrib.torch.msdc_without_crf":
            if importlib.util.find_spec("nilmtk") is None:
                pytest.skip("msdc_without_crf direct import requires nilmtk")
            if importlib.util.find_spec("torch") is None:
                pytest.skip("msdc_without_crf direct import requires torch")
            module = importlib.import_module(spec.module_name)
            cls = getattr(module, spec.class_name)
        else:
            cls = getattr(package, spec.class_name)
    except OptionalDependencyError as exc:
        message = str(exc)
        assert "requires '" in message
        assert "Install nilmtk-contrib[" in message
    else:
        assert cls.__name__ == spec.class_name


@pytest.mark.parametrize(
    "spec",
    ALL_MODEL_SPECS,
    ids=lambda spec: f"{spec.backend}:{spec.module_name}",
)
def test_each_model_module_exists_on_disk(spec):
    assert importlib.util.find_spec(spec.module_name) is not None


def test_model_specs_cover_every_model_module_file(repo_root):
    model_files = set()
    for package_dir in (
        repo_root / "nilmtk_contrib" / "disaggregate",
        repo_root / "nilmtk_contrib" / "torch",
    ):
        for path in package_dir.glob("*.py"):
            if path.name in {"__init__.py", "preprocessing.py"}:
                continue
            module_name = ".".join(path.relative_to(repo_root).with_suffix("").parts)
            model_files.add(module_name)

    covered = {spec.module_name for spec in ALL_MODEL_SPECS}
    assert covered == model_files
