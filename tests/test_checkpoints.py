import json

import pytest

from nilmtk_contrib.utils.checkpoints import (
    SCHEMA_VERSION,
    build_metadata,
    collect_dependencies,
    load_metadata,
    save_metadata,
    temporary_checkpoint,
    unsupported_persistence,
)


def test_temporary_checkpoint_removes_parent_directory_after_exit():
    with temporary_checkpoint(".pt") as path:
        parent = path.parent
        path.write_text("checkpoint", encoding="utf-8")
        assert path.exists()

    assert not parent.exists()


def test_build_save_and_load_metadata(tmp_path):
    metadata = build_metadata(
        model_class="DAE",
        backend="torch",
        sequence_length=99,
        appliance_params={"fridge": {"mean": 10, "std": 2}},
        mains_mean=1000,
        mains_std=600,
        dependencies={"torch": "2.0.0"},
    )

    save_metadata(tmp_path, metadata)
    loaded = load_metadata(
        tmp_path,
        expected_model_class="DAE",
        expected_backend="torch",
    )

    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["model_class"] == "DAE"
    assert loaded["backend"] == "torch"
    assert loaded["sequence_length"] == 99
    assert loaded["appliance_params"] == {"fridge": {"mean": 10, "std": 2}}
    assert loaded["mains_mean"] == 1000
    assert loaded["mains_std"] == 600
    assert loaded["dependencies"] == {"torch": "2.0.0"}
    assert "created_at" in loaded


def test_load_metadata_rejects_missing_fields(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing metadata fields"):
        load_metadata(tmp_path)


def test_load_metadata_rejects_schema_mismatch(tmp_path):
    metadata = build_metadata(
        model_class="DAE",
        backend="torch",
        sequence_length=99,
        appliance_params={},
        mains_mean=1000,
        mains_std=600,
    )
    metadata["schema_version"] = 999
    save_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="Unsupported metadata schema_version"):
        load_metadata(tmp_path)


def test_load_metadata_rejects_wrong_model_or_backend(tmp_path):
    metadata = build_metadata(
        model_class="DAE",
        backend="torch",
        sequence_length=99,
        appliance_params={},
        mains_mean=1000,
        mains_std=600,
    )
    save_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="Expected model_class"):
        load_metadata(tmp_path, expected_model_class="Seq2Point")

    with pytest.raises(ValueError, match="Expected backend"):
        load_metadata(tmp_path, expected_backend="tensorflow")


def test_collect_dependencies_marks_missing_package_as_none():
    dependencies = collect_dependencies(["definitely-missing-nilmtk-contrib-package"])

    assert dependencies == {"definitely-missing-nilmtk-contrib-package": None}


def test_unsupported_persistence_raises_with_model_name():
    with pytest.raises(NotImplementedError, match="AFHMM"):
        unsupported_persistence("AFHMM")
