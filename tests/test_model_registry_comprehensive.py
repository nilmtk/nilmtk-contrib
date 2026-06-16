import ast
import importlib
import importlib.util

import pytest

from nilmtk_contrib.metadata import MODEL_CATALOG, ModelCatalogEntry, model_catalog_by_module
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


def test_public_model_catalog_matches_comprehensive_model_specs():
    catalog_by_module = model_catalog_by_module()
    specs_by_module = {spec.module_name: spec for spec in ALL_MODEL_SPECS}

    assert set(catalog_by_module) == set(specs_by_module)
    assert all(isinstance(entry, ModelCatalogEntry) for entry in MODEL_CATALOG)

    for module_name, spec in specs_by_module.items():
        entry = catalog_by_module[module_name]
        assert entry.backend == spec.backend
        assert entry.module_path == spec.module_name
        assert entry.class_name == spec.class_name
        if module_name == "nilmtk_contrib.torch.msdc_without_crf":
            assert entry.exported_from is None
            assert entry.name == "MSDC without CRF"
        else:
            assert entry.exported_from == spec.package


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
            package.__dict__.pop(spec.class_name, None)
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


def test_model_code_does_not_unpack_appliance_params_by_dict_order(repo_root):
    model_dirs = [
        repo_root / "nilmtk_contrib" / "disaggregate",
        repo_root / "nilmtk_contrib" / "torch",
    ]
    offenders = []
    for model_dir in model_dirs:
        for path in model_dir.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            if "appliance_params" in source and ".values()" in source:
                offenders.append(str(path.relative_to(repo_root)))

    assert offenders == []


def test_model_constructors_accept_params_argument(repo_root):
    offenders = []
    for model_dir in (
        repo_root / "nilmtk_contrib" / "disaggregate",
        repo_root / "nilmtk_contrib" / "torch",
    ):
        for path in model_dir.glob("*.py"):
            if path.name in {"__init__.py", "preprocessing.py"}:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            module_name = ".".join(path.relative_to(repo_root).with_suffix("").parts)
            expected_class_names = [
                spec.class_name
                for spec in ALL_MODEL_SPECS
                if spec.module_name == module_name
            ]
            for class_node in [
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name in expected_class_names
            ]:
                init_methods = [
                    node
                    for node in class_node.body
                    if isinstance(node, ast.FunctionDef) and node.name == "__init__"
                ]
                if not init_methods:
                    offenders.append(f"{path.name}:{class_node.name}:missing __init__")
                    continue
                args = [arg.arg for arg in init_methods[0].args.args]
                if args[:2] != ["self", "params"]:
                    offenders.append(f"{path.name}:{class_node.name}:{args}")

    assert offenders == []


def test_model_modules_define_custom_error_classes_that_they_raise(repo_root):
    offenders = []
    for spec in ALL_MODEL_SPECS:
        if spec.backend == "classical":
            continue
        module_path = repo_root.joinpath(*spec.module_name.split(".")).with_suffix(".py")
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
        class_names = {
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        }
        raised_errors = {
            error_name
            for error_name in {"SequenceLengthError", "ApplianceNotFoundError"}
            if f"raise {error_name}" in source or f"raise ({error_name})" in source
        }
        missing = raised_errors - class_names
        if missing:
            offenders.append(f"{module_path.name}:{sorted(missing)}")

    assert offenders == []


def test_classification_plot_hook_is_defined_when_called(repo_root):
    offenders = []
    for model_dir in (
        repo_root / "nilmtk_contrib" / "disaggregate",
        repo_root / "nilmtk_contrib" / "torch",
    ):
        for path in model_dir.glob("*classification.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for class_node in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
                methods = {
                    node.name
                    for node in class_node.body
                    if isinstance(node, ast.FunctionDef)
                }
                calls_plot_hook = any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "classification_output_plot"
                    for node in ast.walk(class_node)
                )
                if calls_plot_hook and "classification_output_plot" not in methods:
                    offenders.append(f"{path.name}:{class_node.name}")

    assert offenders == []


def _class_method_calls(class_node):
    nested_classes = {
        nested
        for nested in ast.walk(class_node)
        if isinstance(nested, ast.ClassDef) and nested is not class_node
    }
    calls = set()
    for node in ast.walk(class_node):
        if node in nested_classes:
            continue
        if any(parent in nested_classes for parent in getattr(node, "_parents", ())):
            continue
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            calls.add(node.func.attr)
    return calls


def _attach_ast_parents(tree):
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parents = (*getattr(parent, "_parents", ()), parent)


def test_model_self_method_calls_are_defined_or_known_runtime_methods(repo_root):
    known_runtime_methods = {
        "apply",
        "load_model",
        "modules",
        "named_parameters",
        "save_model",
    }
    offenders = []
    for model_dir in (
        repo_root / "nilmtk_contrib" / "disaggregate",
        repo_root / "nilmtk_contrib" / "torch",
    ):
        for path in model_dir.glob("*.py"):
            if path.name == "__init__.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            _attach_ast_parents(tree)
            for class_node in [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]:
                methods = {
                    node.name
                    for node in class_node.body
                    if isinstance(node, ast.FunctionDef)
                }
                assigned_attrs = {
                    target.attr
                    for node in ast.walk(class_node)
                    if isinstance(node, ast.Assign)
                    for target in node.targets
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    )
                }
                missing = sorted(
                    call
                    for call in _class_method_calls(class_node)
                    if call not in methods
                    and call not in assigned_attrs
                    and call not in known_runtime_methods
                )
                if missing:
                    offenders.append(f"{path.name}:{class_node.name}:{missing}")

    assert offenders == []
