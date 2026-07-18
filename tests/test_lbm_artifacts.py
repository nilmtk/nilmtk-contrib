from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json

import pytest


torch = pytest.importorskip("torch")

from nilmtk_contrib.torch._lbm_artifacts import (  # noqa: E402
    ARTIFACT_FILENAME,
    ARTIFACT_SCHEMA_VERSION,
    ARTIFACT_TYPE,
    LBMTrainingBundle,
    _payload_digest,
    load_lbm_training_bundle,
    save_lbm_training_bundle,
)
from nilmtk_contrib.torch._lbm_training import (  # noqa: E402
    ApplianceTrainingWindow,
    fit_lbm_appliance,
)
from lbm_training_data import (  # noqa: E402
    POWER_WINDOWS,
    source_window,
    training_windows,
)


def _evaluation_source(building="2"):
    return source_window(100, building=building)


def _training_result(*, evaluation_sources=(), allow_temporal_evaluation=False):
    return fit_lbm_appliance(
        training_windows(),
        evaluation_sources=evaluation_sources,
        allow_temporal_evaluation=allow_temporal_evaluation,
    )


def _read_envelope(path):
    return json.loads((path / ARTIFACT_FILENAME).read_text(encoding="utf-8"))


def _write_envelope(path, envelope, *, resign=True):
    if resign:
        payload = {
            key: value for key, value in envelope.items() if key != "payload_sha256"
        }
        envelope["payload_sha256"] = _payload_digest(payload)
    (path / ARTIFACT_FILENAME).write_text(
        json.dumps(envelope, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_bundle_round_trip_is_deterministic_read_only_and_auditable(tmp_path):
    result = _training_result(evaluation_sources=(_evaluation_source(),))
    bundle = LBMTrainingBundle({"fridge": result})

    target = save_lbm_training_bundle(tmp_path, bundle)
    first = target.read_bytes()
    loaded = load_lbm_training_bundle(tmp_path)
    save_lbm_training_bundle(tmp_path, loaded)

    assert target.read_bytes() == first
    assert target.stat().st_mode & 0o777 == 0o644
    assert list(tmp_path.glob(f".{ARTIFACT_FILENAME}.*")) == []
    assert loaded.metadata() == bundle.metadata()
    assert loaded.appliances["fridge"].metadata() == result.metadata()
    with pytest.raises(TypeError):
        loaded.appliances["kettle"] = result


def test_bundle_writes_sorted_appliances_and_checksums_the_payload(tmp_path):
    result = _training_result()
    bundle = LBMTrainingBundle({"washing machine": result, "fridge": result})

    save_lbm_training_bundle(tmp_path, bundle)
    envelope = _read_envelope(tmp_path)
    payload = {key: value for key, value in envelope.items() if key != "payload_sha256"}

    assert list(envelope["appliances"]) == ["fridge", "washing machine"]
    assert envelope["artifact_type"] == ARTIFACT_TYPE
    assert envelope["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert envelope["payload_sha256"] == _payload_digest(payload)


def test_loader_rejects_corruption_before_reconstructing_models(tmp_path):
    save_lbm_training_bundle(
        tmp_path,
        LBMTrainingBundle({"fridge": _training_result()}),
    )
    envelope = _read_envelope(tmp_path)
    envelope["appliances"]["fridge"]["state_means"][1] += 1
    _write_envelope(tmp_path, envelope, resign=False)

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_lbm_training_bundle(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda metadata: metadata["initial_probabilities"].__setitem__(
                slice(None), [0.5, 0.5]
            ),
            "initial_probabilities.*raw counts",
        ),
        (
            lambda metadata: metadata["state_counts"].__setitem__(
                0, metadata["state_counts"][0] + 1
            ),
            "state_counts do not sum",
        ),
        (
            lambda metadata: metadata["summaries"][0].__setitem__(
                "induced_mean", metadata["summaries"][0]["induced_mean"] + 1
            ),
            "induced moments do not match",
        ),
        (
            lambda metadata: metadata["evaluation_sources"].append(
                metadata["sources"][0]
            ),
            "Training/evaluation leakage",
        ),
    ],
)
def test_loader_rejects_resigned_but_semantically_false_artifacts(
    tmp_path, mutation, message
):
    save_lbm_training_bundle(
        tmp_path,
        LBMTrainingBundle({"fridge": _training_result()}),
    )
    envelope = _read_envelope(tmp_path)
    mutation(envelope["appliances"]["fridge"])
    _write_envelope(tmp_path, envelope)

    with pytest.raises(ValueError, match=message):
        load_lbm_training_bundle(tmp_path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "fields do not match schema"),
        (
            {
                "artifact_type": "wrong",
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "appliances": {},
                "payload_sha256": "sha256:invalid",
            },
            "Unsupported LBM artifact_type",
        ),
    ],
)
def test_loader_rejects_invalid_envelopes(tmp_path, payload, message):
    _write_envelope(tmp_path, payload, resign=False)

    with pytest.raises(ValueError, match=message):
        load_lbm_training_bundle(tmp_path)


def test_loader_rejects_malformed_json_and_oversized_files(tmp_path, monkeypatch):
    target = tmp_path / ARTIFACT_FILENAME
    target.write_bytes(b"\xffnot-json")
    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        load_lbm_training_bundle(tmp_path)

    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("nilmtk_contrib.torch._lbm_artifacts.MAX_ARTIFACT_BYTES", 1)
    with pytest.raises(ValueError, match="safety limit"):
        load_lbm_training_bundle(tmp_path)


def test_loader_rejects_excessive_json_nesting_with_a_validation_error(tmp_path):
    depth = 2_000
    (tmp_path / ARTIFACT_FILENAME).write_text(
        "[" * depth + "0" + "]" * depth,
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_lbm_training_bundle(tmp_path)


def test_bundle_rejects_cross_appliance_contract_mismatches():
    result = _training_result(evaluation_sources=(_evaluation_source(),))
    different_evaluation = replace(
        result,
        evaluation_sources=(_evaluation_source(building="3"),),
    )
    temporal_policy = replace(result, allow_temporal_evaluation=True)

    with pytest.raises(ValueError, match="same evaluation sources"):
        LBMTrainingBundle({"fridge": result, "kettle": different_evaluation})
    with pytest.raises(ValueError, match="same leakage policy"):
        LBMTrainingBundle({"fridge": result, "kettle": temporal_policy})


def test_bundle_rejects_different_sample_periods_and_window_lengths():
    result = _training_result()
    thirty_second_windows = tuple(
        ApplianceTrainingWindow(
            source_window(index, sample_period_seconds=30),
            power,
        )
        for index, power in enumerate(POWER_WINDOWS)
    )
    thirty_second_result = fit_lbm_appliance(thirty_second_windows)

    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    longer_windows = tuple(
        ApplianceTrainingWindow(
            source_window(
                index,
                start=start + timedelta(hours=index),
                end=start + timedelta(hours=index, minutes=7),
            ),
            [*power, 0],
        )
        for index, power in enumerate(POWER_WINDOWS)
    )
    longer_result = fit_lbm_appliance(longer_windows)

    with pytest.raises(ValueError, match="same sample period"):
        LBMTrainingBundle({"fridge": result, "kettle": thirty_second_result})
    with pytest.raises(ValueError, match="same window length"):
        LBMTrainingBundle({"fridge": result, "kettle": longer_result})


def test_bundle_rejects_ambiguous_names_and_non_results():
    result = _training_result()

    with pytest.raises(ValueError, match="Duplicate normalized"):
        LBMTrainingBundle({"fridge": result, " fridge ": result})
    with pytest.raises(ValueError, match="non-empty strings"):
        LBMTrainingBundle({" ": result})
    with pytest.raises(TypeError, match="LBMTrainingResult"):
        LBMTrainingBundle({"fridge": object()})


def test_save_requires_a_validated_bundle(tmp_path):
    with pytest.raises(TypeError, match="LBMTrainingBundle"):
        save_lbm_training_bundle(tmp_path, {"fridge": _training_result()})


def test_bundle_freezes_fitted_numeric_sequences():
    bundle = LBMTrainingBundle({"fridge": _training_result()})
    probabilities = bundle.appliances["fridge"].appliance.initial_probabilities

    assert isinstance(probabilities, tuple)
    with pytest.raises(TypeError):
        probabilities[0] = 0.5
