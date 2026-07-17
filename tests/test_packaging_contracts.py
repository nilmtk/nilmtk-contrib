from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_EXTRAS = {"all", "classical", "nilm", "tensorflow", "torch"}
EXPECTED_UPSTREAMS = {
    "nilmtk": (
        "nilmtk @ git+https://github.com/nilmtk/nilmtk.git"
        "@6b239d466556fed981f01c442f8ec471c5e2ddc7"
    )
}
EXPECTED_BUILD_BACKEND = ["hatchling==1.31.0"]


def test_runtime_extras_pin_upstream_repositories_to_commits():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        optional_dependencies = tomllib.load(handle)["project"][
            "optional-dependencies"
        ]

    assert RUNTIME_EXTRAS <= optional_dependencies.keys()
    for extra in RUNTIME_EXTRAS:
        requirements = {
            requirement.partition(" @ ")[0]: requirement
            for requirement in optional_dependencies[extra]
            if " @ " in requirement
        }
        assert requirements == EXPECTED_UPSTREAMS, extra


def test_build_backend_and_container_install_are_frozen():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()

    assert project["build-system"]["requires"] == EXPECTED_BUILD_BACKEND
    assert "COPY pyproject.toml uv.lock README.md LICENSE ./" in dockerfile
    assert "uv export --frozen" in dockerfile
    assert "uv pip install --system --requirements" in dockerfile
