#!/usr/bin/env bash
set -Eeuo pipefail

IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
UV_BIN="${UV_BIN:-uv}"
EXTRAS="${EXTRAS:-dev}"
COVERAGE_FAIL_UNDER="${COVERAGE_FAIL_UNDER:-0}"
STRICT_BACKENDS="${STRICT_BACKENDS:-1}"
RUN_NOTEBOOK_EXECUTION="${RUN_NOTEBOOK_EXECUTION:-0}"
RUN_DOCKER_BUILD="${RUN_DOCKER_BUILD:-0}"
CLEAN_BUILD_ARTIFACTS="${CLEAN_BUILD_ARTIFACTS:-1}"

export STRICT_BACKENDS

log() {
  printf '\n[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

run() {
  log "$*"
  "$@"
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

cleanup_generated_artifacts() {
  if [[ "$CLEAN_BUILD_ARTIFACTS" != "1" ]]; then
    return
  fi

  rm -rf build dist .pytest_cache .ruff_cache .coverage .coverage.* coverage.xml htmlcov
  find nilmtk_contrib tests -type d -name __pycache__ -prune -exec rm -rf {} +
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command '$1' was not found."
}

python() {
  "$PYTHON_BIN" "$@"
}

uv_run() {
  "$UV_BIN" run python "$@"
}

trap 'printf "\nFAILED at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR

log "Starting exhaustive validation in $ROOT_DIR"

require_command git
require_command "$PYTHON_BIN"

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  die "uv is required. Install it first, or set UV_BIN to a uv executable."
fi

python - <<'PY'
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"Python 3.11 is required by this repository; found {sys.version.split()[0]}"
    )
print(sys.version)
PY

log "Syncing validation environment with extras: $EXTRAS"
sync_args=()
for extra in $EXTRAS; do
  sync_args+=(--extra "$extra")
done
run "$UV_BIN" sync "${sync_args[@]}"

log "Repository hygiene"
run git status --short
run git ls-files

tracked_md="$(git ls-files '*.md')"
if [[ "$tracked_md" != "README.md" ]]; then
  printf '%s\n' "$tracked_md"
  die "README.md must be the only tracked Markdown file."
fi

untracked_md="$(git ls-files --others --exclude-standard '*.md')"
if [[ -n "$untracked_md" ]]; then
  printf '%s\n' "$untracked_md"
  die "Untracked Markdown files are visible to git; add expected local notes to .gitignore."
fi

tracked_pyc="$(git ls-files '*.pyc')"
if [[ -n "$tracked_pyc" ]]; then
  printf '%s\n' "$tracked_pyc"
  die "Tracked Python bytecode files are not allowed."
fi

if grep -RInE \
  'docs/|action-plan|repo-audit-baseline|server-validation|report\.md|exact implementation|all the state-of-the-art algorithms|paper_faithful_unverified|fast_checks_only|not_run|dev note|todo|fixme|tests/torch|tests/tensorflow|tests/classical' \
  README.md pyproject.toml; then
  die "Production-facing README or package metadata contains forbidden planning/test-path content."
fi

log "Project metadata validation"
uv_run - <<'PY'
import tomllib
from pathlib import Path

project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
version_ns = {}
exec(Path("nilmtk_contrib/version.py").read_text(encoding="utf-8"), version_ns)

assert project["project"]["name"] == "nilmtk-contrib"
assert project["project"]["readme"] == "README.md"
assert project["project"]["requires-python"] == ">=3.11,<3.12"
assert project["project"]["version"] == version_ns["short_version"]
assert version_ns["version"].startswith(version_ns["short_version"])

extras = project["project"]["optional-dependencies"]
for required in ("tensorflow", "torch", "classical", "nilm", "all", "dev"):
    assert required in extras, required

readme = Path("README.md").read_text(encoding="utf-8")
for required in (
    "AFHMM",
    "DAE",
    "Seq2Point",
    "NILMFormer",
    "https://doi.org/10.1145/3360322.3360844",
    "https://github.com/adrienpetralia/NILMFormer",
):
    assert required in readme, required

print("metadata ok")
PY

log "Static validation"
run "$UV_BIN" run python -m ruff check .
run "$UV_BIN" run python -m compileall -q nilmtk_contrib tests

uv_run - <<'PY'
import ast
from pathlib import Path

for path in sorted(Path(".").glob("**/*.py")):
    if any(part in {".venv", "build", "dist"} for part in path.parts):
        continue
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("python AST parse ok")
PY

log "Unit and coverage tests"
run "$UV_BIN" run python -m pytest -q
run "$UV_BIN" run python -m pytest -q --cov=nilmtk_contrib --cov-report=term-missing --cov-fail-under="$COVERAGE_FAIL_UNDER"

log "Lightweight import side-effect validation"
uv_run - <<'PY'
import json
import subprocess
import sys

backend_modules = {"tensorflow", "torch", "cvxpy", "hmmlearn", "nilmtk", "pandas"}
statements = [
    "import nilmtk_contrib",
    "import nilmtk_contrib.disaggregate",
    "import nilmtk_contrib.torch",
    "import nilmtk_contrib.mains_stats",
]

for statement in statements:
    code = (
        "import json, sys\n"
        f"{statement}\n"
        f"print(json.dumps(sorted({backend_modules!r}.intersection(sys.modules))))"
    )
    output = subprocess.check_output([sys.executable, "-c", code], text=True)
    imported = json.loads(output)
    if imported:
        raise SystemExit(f"{statement!r} imported backend modules unexpectedly: {imported}")

print("lightweight imports ok")
PY

log "Public model catalog and export validation"
uv_run - <<'PY'
import importlib
import pkgutil
import os

from nilmtk_contrib.metadata import MODEL_CATALOG, model_catalog_by_module
from nilmtk_contrib.utils.optional_imports import OptionalDependencyError

STRICT_BACKENDS = os.environ.get("STRICT_BACKENDS", "1")

assert len(MODEL_CATALOG) >= 29
catalog_by_module = model_catalog_by_module()
assert len(catalog_by_module) == len({entry.module_path for entry in MODEL_CATALOG})

for entry in MODEL_CATALOG:
    module = importlib.import_module(entry.module_path)
    assert module is not None, entry.module_path

for package_name in ("nilmtk_contrib.disaggregate", "nilmtk_contrib.torch"):
    package = importlib.import_module(package_name)
    for export_name in package.__all__:
        if export_name == "Disaggregator":
            continue
        try:
            getattr(package, export_name)
        except OptionalDependencyError:
            if STRICT_BACKENDS == "1":
                raise
        except Exception as exc:
            raise RuntimeError(f"{package_name}.{export_name} export failed") from exc

root = importlib.import_module("nilmtk_contrib")
for module_info in pkgutil.walk_packages(root.__path__, root.__name__ + "."):
    name = module_info.name
    if name.endswith(".__pycache__"):
        continue
    try:
        importlib.import_module(name)
    except OptionalDependencyError:
        if STRICT_BACKENDS == "1":
            raise
    except Exception as exc:
        raise RuntimeError(f"module import failed: {name}") from exc

print("public exports and module imports ok")
PY

log "Model constructor smoke validation"
uv_run - <<'PY'
import importlib
import os

from nilmtk_contrib.utils.optional_imports import OptionalDependencyError

STRICT_BACKENDS = os.environ.get("STRICT_BACKENDS", "1")

COMMON_PARAMS = {
    "sequence_length": 99,
    "n_epochs": 0,
    "batch_size": 2,
    "mains_mean": 1000.0,
    "mains_std": 600.0,
    "appliance_params": {
        "fridge": {
            "mean": 50.0,
            "std": 10.0,
            "min": 0.0,
            "max": 150.0,
        }
    },
    "save_model_path": None,
    "pretrained_model_path": None,
    "chunk_wise_training": False,
    "seed": 123,
    "verbose": False,
    "device": "cpu",
    "learning_rate": 0.001,
    "appliances": ["fridge"],
    "appliance": "fridge",
    "time_period": 720,
    "iterations": 1,
    "num_of_states": 2,
    "num_states": 2,
    "on_power_threshold": {"fridge": 10.0},
    "loss_weights": {"fridge": 1.0},
}

targets = [
    ("nilmtk_contrib.disaggregate", "AFHMM"),
    ("nilmtk_contrib.disaggregate", "AFHMM_SAC"),
    ("nilmtk_contrib.disaggregate", "BERT"),
    ("nilmtk_contrib.disaggregate", "DAE"),
    ("nilmtk_contrib.disaggregate", "DSC"),
    ("nilmtk_contrib.disaggregate", "RNN"),
    ("nilmtk_contrib.disaggregate", "RNN_attention"),
    ("nilmtk_contrib.disaggregate", "RNN_attention_classification"),
    ("nilmtk_contrib.disaggregate", "ResNet"),
    ("nilmtk_contrib.disaggregate", "ResNet_classification"),
    ("nilmtk_contrib.disaggregate", "Seq2Point"),
    ("nilmtk_contrib.disaggregate", "Seq2Seq"),
    ("nilmtk_contrib.disaggregate", "WindowGRU"),
    ("nilmtk_contrib.torch", "BERT"),
    ("nilmtk_contrib.torch", "ConvLSTM"),
    ("nilmtk_contrib.torch", "DAE"),
    ("nilmtk_contrib.torch", "MSDC"),
    ("nilmtk_contrib.torch", "NILMFormer"),
    ("nilmtk_contrib.torch", "Reformer"),
    ("nilmtk_contrib.torch", "ResNet"),
    ("nilmtk_contrib.torch", "ResNet_classification"),
    ("nilmtk_contrib.torch", "RNN"),
    ("nilmtk_contrib.torch", "RNN_attention"),
    ("nilmtk_contrib.torch", "RNN_attention_classification"),
    ("nilmtk_contrib.torch", "Seq2PointTorch"),
    ("nilmtk_contrib.torch", "Seq2Seq"),
    ("nilmtk_contrib.torch", "TCN"),
    ("nilmtk_contrib.torch", "WindowGRU"),
]

failures = []
for module_name, class_name in targets:
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
        instance = cls(dict(COMMON_PARAMS))
        assert instance is not None
    except OptionalDependencyError:
        if STRICT_BACKENDS == "1":
            failures.append((module_name, class_name, "missing optional dependency"))
    except Exception as exc:
        failures.append((module_name, class_name, repr(exc)))

if failures:
    for module_name, class_name, error in failures:
        print(f"{module_name}.{class_name}: {error}")
    raise SystemExit("model constructor smoke validation failed")

print("model constructors ok")
PY

log "Notebook structure validation"
uv_run - <<'PY'
import json
from pathlib import Path

notebooks = sorted(Path("sample_notebooks").glob("*.ipynb"))
assert notebooks, "sample_notebooks contains no notebooks"
for path in notebooks:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("cells"), path
    assert data.get("metadata") is not None, path
    print(path)
PY

if [[ "$RUN_NOTEBOOK_EXECUTION" == "1" ]]; then
  log "Executing notebooks because RUN_NOTEBOOK_EXECUTION=1"
  run "$UV_BIN" pip install nbconvert ipykernel
  while IFS= read -r notebook; do
    run "$UV_BIN" run jupyter nbconvert --to notebook --execute --inplace "$notebook"
  done < <(find sample_notebooks -name '*.ipynb' -print)
else
  log "Skipping notebook execution. Set RUN_NOTEBOOK_EXECUTION=1 to execute notebooks; this may require datasets."
fi

log "Package build and artifact validation"
rm -rf dist build
run "$UV_BIN" run python -m build

uv_run - <<'PY'
import tarfile
import zipfile
from pathlib import Path

dist = Path("dist")
sdists = sorted(dist.glob("*.tar.gz"))
wheels = sorted(dist.glob("*.whl"))
assert len(sdists) == 1, sdists
assert len(wheels) == 1, wheels

for sdist in sdists:
    with tarfile.open(sdist) as tf:
        names = tf.getnames()
    assert any(name.endswith("README.md") for name in names), sdist
    forbidden = ("action-plan.md", "report.md", "docs/models.md", "docs/references.md", "docs/repo-audit-baseline.md", "docs/server-validation.md")
    assert not any(any(name.endswith(item) for item in forbidden) for name in names), sdist

for wheel in wheels:
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    assert "nilmtk_contrib/__init__.py" in names, wheel
    assert "nilmtk_contrib/version.py" in names, wheel
    assert any(name.endswith(".dist-info/METADATA") for name in names), wheel

print("artifacts ok")
PY

tmp_install_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_install_dir"; printf "\nFAILED at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR
run "$PYTHON_BIN" -m venv "$tmp_install_dir/venv"
run "$tmp_install_dir/venv/bin/python" -m pip install --upgrade pip
run "$tmp_install_dir/venv/bin/python" -m pip install dist/*.whl
run "$tmp_install_dir/venv/bin/python" - <<'PY'
import nilmtk_contrib

assert nilmtk_contrib.__version__ == "0.1.2"
print(nilmtk_contrib.__version__)
PY
rm -rf "$tmp_install_dir"

if [[ "$RUN_DOCKER_BUILD" == "1" ]]; then
  require_command docker
  log "Building Docker image because RUN_DOCKER_BUILD=1"
  run docker build -t nilmtk-contrib-validation .
else
  log "Skipping Docker build. Set RUN_DOCKER_BUILD=1 to build the Docker image."
fi

log "Final diff and generated-artifact checks"
run git diff --check
cleanup_generated_artifacts

remaining_generated="$(find . -maxdepth 2 \( -name dist -o -name build -o -name .pytest_cache -o -name .ruff_cache -o -name .coverage -o -name '.coverage.*' -o -name coverage.xml -o -name htmlcov \) -print)"
if [[ -n "$remaining_generated" ]]; then
  printf '%s\n' "$remaining_generated"
  die "Generated validation artifacts remain."
fi

log "Exhaustive validation completed successfully."
