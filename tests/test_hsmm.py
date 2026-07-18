import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
hsmm_module = pytest.importorskip("nilmtk_contrib.torch._hsmm")
HSMM = hsmm_module.HSMM


def _params(**overrides):
    return {
        "device": "cpu",
        "num_states": 2,
        "max_duration": 8,
        "pseudocount": 1.0,
        "variance_floor": 0.1,
        "kmeans_max_iterations": 50,
        "seed": 7,
    } | overrides


def _chunks(*, length=24, start="2026-01-01", background=20.0):
    index = pd.date_range(start, periods=length, freq="1min", tz="UTC")
    states = np.tile(
        np.array([0, 0, 0, 100, 100, 100], dtype=np.float32),
        (length + 5) // 6,
    )[:length]
    residual = background + 0.1 * np.sin(np.arange(length, dtype=np.float32))
    mains = states + residual
    return (
        pd.DataFrame({"power": mains}, index=index),
        pd.DataFrame({"power": states}, index=index),
    )


def _training_chunks():
    first_main, first_target = _chunks(length=24)
    second_main, second_target = _chunks(length=18, start="2026-02-01", background=22.0)
    return [first_main, second_main], [("fridge", [first_target, second_target])]


def test_fit_learns_normalized_state_transition_and_duration_contracts():
    mains, appliances = _training_chunks()
    model = HSMM(_params())

    model.partial_fit(mains, appliances)
    fitted = model.models["fridge"]

    assert fitted.state_means == pytest.approx((0.0, 100.0))
    assert np.subtract(fitted.aggregate_means, fitted.state_means) == pytest.approx(
        (21.0, 21.0), abs=0.2
    )
    assert sum(fitted.state_counts) == fitted.num_samples == 42
    assert sum(fitted.initial_counts) == fitted.num_chunks == 2
    assert sum(sum(row) for row in fitted.duration_counts) == fitted.num_segments
    assert (
        sum(sum(row) for row in fitted.transition_counts)
        == fitted.num_segments - fitted.num_chunks
    )
    assert fitted.right_censored_segments == 0
    assert sum(fitted.initial_probabilities) == pytest.approx(1.0)
    assert all(
        sum(row) == pytest.approx(1.0) for row in fitted.transition_probabilities
    )
    assert all(sum(row) == pytest.approx(1.0) for row in fitted.duration_probabilities)
    assert all(fitted.transition_probabilities[state][state] == 0 for state in range(2))


def test_train_infer_recovers_clean_states_and_preserves_index_and_dtype():
    mains, appliances = _training_chunks()
    model = HSMM(_params())
    model.partial_fit(mains, appliances)

    prediction = model.disaggregate_chunk(mains)[0]

    assert prediction.index.equals(mains[0].index)
    assert prediction.columns.tolist() == ["fridge"]
    assert prediction["fridge"].dtype == np.float32
    assert prediction["fridge"].to_numpy() == pytest.approx(
        appliances[0][1][0]["power"].to_numpy()
    )


def test_inference_decodes_long_chunks_without_artificial_state_resets(monkeypatch):
    mains, appliances = _training_chunks()
    model = HSMM(_params(max_duration=5))
    model.partial_fit(mains, appliances)
    lengths = []
    original = hsmm_module.semi_markov_viterbi

    def recording_decoder(emission, *args):
        lengths.append(len(emission))
        return original(emission, *args)

    monkeypatch.setattr(hsmm_module, "semi_markov_viterbi", recording_decoder)
    prediction = model.disaggregate_chunk([mains[0].iloc[:13]])[0]

    assert lengths == [13]
    assert len(prediction) == 13
    assert np.isfinite(prediction["fridge"]).all()


def test_long_training_runs_are_right_censored_at_the_inference_cap():
    index = pd.date_range("2026-01-01", periods=20, freq="1min", tz="UTC")
    target = np.array([0.0] * 5 + [100.0] * 5, dtype=np.float32)
    target = np.tile(target, 2)
    mains = pd.DataFrame({"power": target + 20}, index=index)
    appliance = pd.DataFrame({"power": target}, index=index)
    model = HSMM(_params(max_duration=3))

    model.partial_fit([mains], [("fridge", [appliance])])
    fitted = model.models["fridge"]

    assert fitted.right_censored_segments == 4
    assert fitted.duration_counts[0][-1] == 2
    assert fitted.duration_counts[1][-1] == 2


def test_repeated_fits_produce_identical_parameter_records():
    mains, appliances = _training_chunks()
    first = HSMM(_params(seed=1))
    second = HSMM(_params(seed=999))

    first.partial_fit(mains, appliances)
    second.partial_fit(mains, appliances)

    assert first.models == second.models


def test_empty_inference_chunk_preserves_empty_index_and_columns():
    mains, appliances = _training_chunks()
    model = HSMM(_params())
    model.partial_fit(mains, appliances)
    empty = mains[0].iloc[:0]

    prediction = model.disaggregate_chunk([empty])[0]

    assert prediction.index.equals(empty.index)
    assert prediction.columns.tolist() == ["fridge"]
    assert prediction.empty


def test_multiple_appliances_are_fitted_without_partial_state_on_failure():
    mains, appliances = _training_chunks()
    kettle = [frame.copy() for frame in appliances[0][1]]
    for frame in kettle:
        frame["power"] *= 2
    model = HSMM(_params())
    model.partial_fit(mains, appliances)
    original = model.models

    with pytest.raises(ValueError, match="unique values"):
        model.partial_fit(
            mains,
            [
                ("fridge", appliances[0][1]),
                (
                    "constant",
                    [
                        pd.DataFrame({"power": np.ones(len(frame))}, index=frame.index)
                        for frame in mains
                    ],
                ),
            ],
        )

    assert model.models is original
    model.partial_fit(mains, [("fridge", appliances[0][1]), ("kettle", kettle)])
    assert list(model.models) == ["fridge", "kettle"]


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"num_states": 1}, ValueError, "at least two"),
        ({"num_states": True}, ValueError, "positive integer"),
        ({"max_duration": 0}, ValueError, "positive integer"),
        ({"pseudocount": 0}, ValueError, "positive finite"),
        ({"variance_floor": float("nan")}, ValueError, "positive finite"),
        ({"kmeans_max_iterations": 0}, ValueError, "positive integer"),
        ({"seed": True}, ValueError, "integer or None"),
        ({"verbose": "yes"}, ValueError, "boolean"),
        ({"chunk_wise_training": True}, ValueError, "does not support"),
    ],
)
def test_invalid_configuration_fails_at_construction(overrides, error, message):
    with pytest.raises(error, match=message):
        HSMM(_params(**overrides))


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        (
            lambda mains, targets: (mains, []),
            ValueError,
            "chunks but mains",
        ),
        (
            lambda mains, targets: (mains, [targets[0].iloc[:-1], targets[1]]),
            ValueError,
            "length does not match",
        ),
        (
            lambda mains, targets: (
                mains,
                [
                    targets[0].set_axis(targets[0].index + pd.Timedelta(days=1)),
                    targets[1],
                ],
            ),
            ValueError,
            "index does not match",
        ),
        (
            lambda mains, targets: (
                mains,
                [targets[0].assign(power=-1), targets[1]],
            ),
            ValueError,
            "non-negative",
        ),
    ],
)
def test_training_rejects_misaligned_or_invalid_targets(mutation, error, message):
    mains, appliances = _training_chunks()
    mutated_mains, mutated_targets = mutation(mains, appliances[0][1])
    model = HSMM(_params())

    with pytest.raises(error, match=message):
        model.partial_fit(mutated_mains, [("fridge", mutated_targets)])


def test_training_and_inference_contracts_reject_invalid_containers():
    mains, appliances = _training_chunks()
    model = HSMM(_params())

    with pytest.raises(ValueError, match="at least one mains"):
        model.partial_fit([], appliances)
    with pytest.raises(ValueError, match="at least one appliance"):
        model.partial_fit(mains, [])
    with pytest.raises(ValueError, match="Duplicate appliance"):
        model.partial_fit(mains, [appliances[0], appliances[0]])
    with pytest.raises(RuntimeError, match="trained or loaded"):
        model.disaggregate_chunk(mains)
    with pytest.raises(ValueError, match="boolean"):
        model.partial_fit(mains, appliances, do_preprocessing="yes")


def test_checkpoint_roundtrip_restores_config_models_and_predictions(tmp_path):
    mains, appliances = _training_chunks()
    model = HSMM(_params(save_model_path=str(tmp_path)))
    model.partial_fit(mains, appliances)
    expected = model.disaggregate_chunk(mains)

    loaded = HSMM(
        _params(
            num_states=3,
            max_duration=3,
            pseudocount=2.0,
            variance_floor=2.0,
            kmeans_max_iterations=2,
            pretrained_model_path=str(tmp_path),
        )
    )
    actual = loaded.disaggregate_chunk(mains)

    assert loaded.num_states == 2
    assert loaded.max_duration == 8
    assert loaded.pseudocount == 1.0
    assert loaded.variance_floor == 0.1
    assert loaded.kmeans_max_iterations == 50
    assert loaded.models == model.models
    assert len(actual) == len(expected)
    for actual_frame, expected_frame in zip(actual, expected):
        assert np.array_equal(actual_frame.to_numpy(), expected_frame.to_numpy())


def test_untrained_model_cannot_publish_an_artifact(tmp_path):
    with pytest.raises(RuntimeError, match="trained or loaded"):
        HSMM(_params()).save_model(tmp_path)


def test_persistence_requires_an_explicit_directory():
    model = HSMM(_params())

    with pytest.raises(ValueError, match="checkpoint directory"):
        model.save_model()
    with pytest.raises(ValueError, match="checkpoint directory"):
        model.load_model()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema_version=999), "schema_version"),
        (lambda payload: payload.update(schema_version=True), "schema_version"),
        (lambda payload: payload.update(model_class="Other"), "model_class"),
        (lambda payload: payload.update(extra=True), "top-level"),
        (
            lambda payload: payload["models"]["fridge"][
                "initial_probabilities"
            ].__setitem__(0, float("nan")),
            "valid HSMM artifact",
        ),
        (
            lambda payload: payload["models"]["fridge"]["state_counts"].__setitem__(
                0, 0
            ),
            "state_counts",
        ),
        (
            lambda payload: payload["models"]["fridge"][
                "transition_probabilities"
            ][0].__setitem__(0, 0.1),
            "zero diagonal",
        ),
    ],
)
def test_malformed_artifacts_fail_closed_without_mutating_model(
    tmp_path, mutation, message
):
    mains, appliances = _training_chunks()
    source = tmp_path / "source"
    model = HSMM(_params(save_model_path=str(source)))
    model.partial_fit(mains, appliances)
    artifact = source / "hsmm.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    mutation(payload)
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "hsmm.json").write_text(json.dumps(payload), encoding="utf-8")
    original_models = model.models
    original_config = (model.num_states, model.max_duration)

    with pytest.raises(ValueError, match=message):
        model.load_model(corrupt)

    assert model.models is original_models
    assert (model.num_states, model.max_duration) == original_config


def test_duplicate_artifact_keys_are_rejected(tmp_path):
    mains, appliances = _training_chunks()
    source = tmp_path / "source"
    model = HSMM(_params(save_model_path=str(source)))
    model.partial_fit(mains, appliances)
    artifact = source / "hsmm.json"
    content = artifact.read_text(encoding="utf-8")
    content = content.replace(
        '"schema_version": 1',
        '"schema_version": 1, "schema_version": 1',
        1,
    )
    artifact.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate JSON key"):
        model.load_model(source)


def test_public_export_catalog_and_defaults_are_complete():
    import nilmtk_contrib.torch as torch_models
    from nilmtk_contrib.metadata import model_catalog_by_module

    default = torch_models.HSMM({"device": "cpu"})
    entry = model_catalog_by_module()["nilmtk_contrib.torch.hsmm"]

    assert torch_models.HSMM is HSMM
    assert default.num_states == 2
    assert default.max_duration == 720
    assert entry.class_name == "HSMM"
    assert entry.exported_from == "nilmtk_contrib.torch"


def test_implementation_uses_shared_exact_core_without_classical_solver():
    source = Path(inspect.getfile(HSMM)).read_text(encoding="utf-8")

    assert "semi_markov_viterbi" in source
    for dependency in ("cvxpy", "hmmlearn", "mosek", "scipy"):
        assert dependency not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_inference_matches_cpu():
    mains, appliances = _training_chunks()
    cpu = HSMM(_params(device="cpu"))
    cuda = HSMM(_params(device="cuda"))
    cpu.partial_fit(mains, appliances)
    cuda.partial_fit(mains, appliances)

    cpu_prediction = cpu.disaggregate_chunk(mains)[0]
    cuda_prediction = cuda.disaggregate_chunk(mains)[0]

    assert np.array_equal(cpu_prediction.to_numpy(), cuda_prediction.to_numpy())
