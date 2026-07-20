import ast
import importlib.util

from nilmtk_contrib.metadata import MODEL_CATALOG


RETIRED_MODULES = {
    "nilmtk_contrib.disaggregate.WindowGRU",
    "nilmtk_contrib.disaggregate.bert",
    "nilmtk_contrib.disaggregate.dae",
    "nilmtk_contrib.disaggregate.resnet",
    "nilmtk_contrib.disaggregate.resnet_classification",
    "nilmtk_contrib.disaggregate.rnn",
    "nilmtk_contrib.disaggregate.rnn_attention",
    "nilmtk_contrib.disaggregate.rnn_attention_classification",
    "nilmtk_contrib.disaggregate.seq2point",
    "nilmtk_contrib.disaggregate.seq2seq",
}


def test_retired_tensorflow_modules_are_absent():
    missing = {
        module_name
        for module_name in RETIRED_MODULES
        if importlib.util.find_spec(module_name) is not None
    }
    assert missing == set()


def test_package_source_has_no_tensorflow_or_keras_imports(repo_root):
    offenders = []
    for path in (repo_root / "nilmtk_contrib").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = [node.module]
            for module_name in imported:
                if module_name.split(".", 1)[0] in {"keras", "tensorflow"}:
                    offenders.append(
                        f"{path.relative_to(repo_root)}:{node.lineno}:{module_name}"
                    )

    assert offenders == []


def test_catalog_has_no_tensorflow_backend():
    assert {entry.backend for entry in MODEL_CATALOG}.isdisjoint(
        {"keras", "tensorflow"}
    )


def test_packaging_and_container_configs_have_no_tensorflow_surface(repo_root):
    config_paths = (
        repo_root / "pyproject.toml",
        repo_root / "uv.lock",
        repo_root / "Dockerfile",
        repo_root / ".github" / "workflows" / "docker-publish.yaml",
    )
    offenders = []
    for path in config_paths:
        source = path.read_text(encoding="utf-8").lower()
        for retired_name in ("keras", "tensorflow"):
            if retired_name in source:
                offenders.append(
                    f"{path.relative_to(repo_root)} contains {retired_name}"
                )
    assert offenders == []
