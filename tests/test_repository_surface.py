import ast
import json
import py_compile
import tomllib


IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "sample_notebooks",
}

IGNORED_NAMES = {
    ".coverage",
    "coverage.xml",
}

KNOWN_ROOT_DOTFILES = {
    ".dockerignore",
    ".gitattributes",
    ".gitignore",
}

SOURCE_ROOTS = {
    "nilmtk_contrib",
    "tests",
    "scripts",
    "docs",
}


def is_ignored_generated_file(path):
    if path.name in IGNORED_NAMES:
        return True
    if path.name.startswith(".coverage."):
        return True
    if path.name.startswith(".") and path.name not in KNOWN_ROOT_DOTFILES:
        return True
    if any(part.endswith(".egg-info") for part in path.parts):
        return True
    return False


def is_repository_source_file(path, repo_root):
    relative = path.relative_to(repo_root)
    if len(relative.parts) == 1:
        return True
    return relative.parts[0] in SOURCE_ROOTS


def repository_files(repo_root):
    return [
        path
        for path in repo_root.rglob("*")
        if (
            path.is_file()
            and is_repository_source_file(path, repo_root)
            and not is_ignored_generated_file(path)
            and not IGNORED_PARTS.intersection(path.parts)
        )
    ]


def test_every_repository_file_has_a_validation_path(repo_root):
    files = repository_files(repo_root)
    assert files, "repository should contain files"

    validated = set()
    for path in files:
        suffix = path.suffix.lower()
        name = path.name

        if suffix == ".py":
            py_compile.compile(str(path), doraise=True)
            ast.parse(path.read_text(encoding="utf-8"))
        elif suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif suffix == ".toml":
            tomllib.loads(path.read_text(encoding="utf-8"))
        elif suffix in {".md", ".txt", ".lock", ".sh"} or name in {
            ".dockerignore",
            ".gitattributes",
            ".gitignore",
            "Dockerfile",
            "LICENSE",
        }:
            assert path.read_text(encoding="utf-8").strip()
        else:
            raise AssertionError(f"No validation rule for repository file: {path}")

        validated.add(path)

    assert validated == set(files)


def test_generated_runtime_files_are_not_repository_sources(repo_root):
    generated_names = [
        ".coverage",
        ".coverage.host.123.random",
        "coverage.xml",
        ".nfs0000000000000001",
    ]
    for name in generated_names:
        path = repo_root / name
        assert is_ignored_generated_file(path)


def test_known_root_dotfiles_remain_repository_sources(repo_root):
    for name in KNOWN_ROOT_DOTFILES:
        path = repo_root / name
        assert not is_ignored_generated_file(path)
        assert is_repository_source_file(path, repo_root)


def test_nested_non_source_files_are_excluded(repo_root):
    assert not is_repository_source_file(
        repo_root / "sample_notebooks" / "historical.ipynb",
        repo_root,
    )
    assert is_repository_source_file(
        repo_root / "nilmtk_contrib" / "version.py",
        repo_root,
    )


def test_python_packages_have_init_files(repo_root):
    package_dirs = [
        repo_root / "nilmtk_contrib",
        repo_root / "nilmtk_contrib" / "disaggregate",
        repo_root / "nilmtk_contrib" / "preprocessing",
        repo_root / "nilmtk_contrib" / "torch",
        repo_root / "nilmtk_contrib" / "utils",
    ]
    for package_dir in package_dirs:
        assert (package_dir / "__init__.py").exists()


def test_readme_documents_all_public_model_exports(repo_root):
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    expected_names = [
        "AFHMM",
        "AFHMM_SAC",
        "BERT",
        "ConvLSTM",
        "DAE",
        "DSC",
        "MSDC",
        "NILMFormer",
        "RNN",
        "RNN_attention",
        "RNN_attention_classification",
        "Reformer",
        "ResNet",
        "ResNet_classification",
        "Seq2Point",
        "Seq2PointTorch",
        "Seq2Seq",
        "TCN",
        "WindowGRU",
    ]
    for name in expected_names:
        assert name in readme
