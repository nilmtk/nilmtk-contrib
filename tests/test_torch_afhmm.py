import inspect
from itertools import pairwise
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
afhmm_module = pytest.importorskip("nilmtk_contrib.torch._afhmm")
TorchAFHMM = afhmm_module.TorchAFHMM


def _params(**overrides):
    return {
        "device": "cpu",
        "num_states": 2,
        "pseudocount": 1.0,
        "kmeans_max_iterations": 50,
        "kmeans_tolerance": 1e-9,
        "noise_std": 1.0,
        "inference_max_iterations": 20,
        "seed": 7,
    } | overrides


def _chunk(length, start):
    index = pd.date_range(start, periods=length, freq="1min", tz="UTC")
    fridge = np.resize(np.array([0, 0, 0, 80, 80, 80], dtype=np.float32), length)
    kettle = np.resize(np.array([0, 30, 30, 0, 30, 0], dtype=np.float32), length)
    mains = fridge + kettle
    return (
        pd.DataFrame({"power": mains}, index=index),
        pd.DataFrame({"power": fridge}, index=index),
        pd.DataFrame({"power": kettle}, index=index),
    )


def _training_chunks():
    first = _chunk(24, "2026-01-01")
    second = _chunk(18, "2026-02-01")
    mains = [first[0], second[0]]
    appliances = [
        ("kettle", [first[2], second[2]]),
        ("fridge", [first[1], second[1]]),
    ]
    return mains, appliances


def _assert_same_predictions(actual, expected):
    assert len(actual) == len(expected)
    for actual_frame, expected_frame in zip(actual, expected, strict=True):
        assert actual_frame.index.equals(expected_frame.index)
        assert np.array_equal(actual_frame.to_numpy(), expected_frame.to_numpy())


def test_fit_learns_ordered_normalized_hmms_and_preserves_chunk_boundaries():
    mains, appliances = _training_chunks()
    model = TorchAFHMM(_params())

    model.partial_fit(mains, appliances)

    assert list(model.models) == ["fridge", "kettle"]
    assert model.models["fridge"].state_means == pytest.approx((0.0, 80.0))
    assert model.models["kettle"].state_means == pytest.approx((0.0, 30.0))
    for fitted in model.models.values():
        assert sum(fitted.initial_probabilities) == pytest.approx(1.0)
        assert all(
            sum(row) == pytest.approx(1.0) for row in fitted.transition_probabilities
        )
        assert all(value > 0 for value in fitted.initial_probabilities)
        assert all(
            value > 0 for row in fitted.transition_probabilities for value in row
        )
        assert fitted.num_samples == 42
        assert fitted.num_chunks == 2
        assert fitted.fit_converged
        assert fitted.fit_iterations == len(fitted.fit_loss_history) - 1
        assert all(right <= left for left, right in pairwise(fitted.fit_loss_history))


def test_joint_inference_recovers_both_appliances_with_diagnostics():
    mains, appliances = _training_chunks()
    model = TorchAFHMM(_params())
    model.partial_fit(mains, appliances)

    predictions = model.disaggregate_chunk(mains)

    expected_by_name = {name: frames for name, frames in appliances}
    assert len(predictions) == len(mains)
    assert len(model.last_inference_diagnostics) == len(mains)
    for chunk_index, prediction in enumerate(predictions):
        assert prediction.index.equals(mains[chunk_index].index)
        assert prediction.columns.tolist() == ["fridge", "kettle"]
        assert all(dtype == np.float32 for dtype in prediction.dtypes)
        for name in prediction:
            assert prediction[name].to_numpy() == pytest.approx(
                expected_by_name[name][chunk_index]["power"].to_numpy()
            )
        diagnostics = model.last_inference_diagnostics[chunk_index]
        assert diagnostics.samples == len(mains[chunk_index])
        assert diagnostics.converged
        assert diagnostics.iterations == len(diagnostics.score_history) - 1
        assert diagnostics.score == diagnostics.score_history[-1]
        assert all(right >= left for left, right in pairwise(diagnostics.score_history))


def test_inference_decodes_each_full_chunk_without_artificial_resets(monkeypatch):
    mains, appliances = _training_chunks()
    model = TorchAFHMM(_params())
    model.partial_fit(mains, appliances)
    lengths = []
    original = afhmm_module.factorial_hmm_coordinate_viterbi

    def recording_decoder(observations, *args, **kwargs):
        lengths.append(len(observations))
        return original(observations, *args, **kwargs)

    monkeypatch.setattr(
        afhmm_module, "factorial_hmm_coordinate_viterbi", recording_decoder
    )

    predictions = model.disaggregate_chunk([mains[0].iloc[:13], mains[1].iloc[:7]])

    assert lengths == [13, 7]
    assert [len(frame) for frame in predictions] == [13, 7]


def test_repeated_fits_are_seed_independent_and_deterministic():
    mains, appliances = _training_chunks()
    first = TorchAFHMM(_params(seed=1))
    second = TorchAFHMM(_params(seed=999))

    first.partial_fit(mains, appliances)
    second.partial_fit(mains, appliances)

    assert first.models == second.models
    _assert_same_predictions(
        first.disaggregate_chunk(mains), second.disaggregate_chunk(mains)
    )


def test_empty_inference_preserves_index_columns_and_diagnostics():
    mains, appliances = _training_chunks()
    model = TorchAFHMM(_params())
    model.partial_fit(mains, appliances)
    empty = mains[0].iloc[:0]

    prediction = model.disaggregate_chunk([empty])[0]

    assert prediction.index.equals(empty.index)
    assert prediction.columns.tolist() == ["fridge", "kettle"]
    assert prediction.empty
    assert all(dtype == np.float32 for dtype in prediction.dtypes)
    assert model.last_inference_diagnostics == (
        afhmm_module.TorchAFHMMInferenceDiagnostics(
            samples=0,
            iterations=0,
            converged=True,
            score=None,
            score_history=(),
        ),
    )


def test_failed_multi_appliance_fit_is_atomic():
    mains, appliances = _training_chunks()
    model = TorchAFHMM(_params())
    model.partial_fit(mains, appliances)
    original = model.models
    constant = [
        pd.DataFrame({"power": np.ones(len(frame))}, index=frame.index)
        for frame in mains
    ]

    with pytest.raises(ValueError, match="distinct power"):
        model.partial_fit(
            mains,
            [("fridge", appliances[1][1]), ("constant", constant)],
        )

    assert model.models is original


def test_checkpoint_write_failure_rolls_back_the_fitted_mapping(monkeypatch, tmp_path):
    mains, appliances = _training_chunks()
    model = TorchAFHMM(_params(save_model_path=str(tmp_path)))
    model.partial_fit(mains, appliances)
    original = model.models

    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(afhmm_module, "save_json_atomic", fail_write)

    with pytest.raises(OSError, match="disk full"):
        model.partial_fit(mains, appliances)

    assert model.models is original


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_states": 1}, "at least 2"),
        ({"num_states": True}, "positive integer"),
        ({"pseudocount": 0}, "positive"),
        ({"pseudocount": float("inf")}, "finite positive"),
        ({"kmeans_max_iterations": 0}, "positive integer"),
        ({"kmeans_tolerance": -1}, "nonnegative"),
        ({"kmeans_tolerance": float("nan")}, "finite nonnegative"),
        ({"noise_std": 0}, "positive"),
        ({"inference_max_iterations": 0}, "positive integer"),
        ({"fail_on_nonconvergence": "yes"}, "boolean"),
        ({"seed": True}, "integer or None"),
        ({"verbose": "yes"}, "boolean"),
        ({"chunk_wise_training": True}, "does not support"),
    ],
)
def test_invalid_configuration_fails_at_construction(overrides, message):
    with pytest.raises(ValueError, match=message):
        TorchAFHMM(_params(**overrides))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda mains, targets: (mains, []), "chunks but mains"),
        (
            lambda mains, targets: (mains, [targets[0].iloc[:-1], targets[1]]),
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
            "index does not match",
        ),
        (
            lambda mains, targets: (
                mains,
                [targets[0].assign(power=-1), targets[1]],
            ),
            "non-negative",
        ),
    ],
)
def test_training_rejects_misaligned_or_invalid_targets(mutation, message):
    mains, appliances = _training_chunks()
    mutated_mains, mutated_targets = mutation(mains, appliances[1][1])

    with pytest.raises(ValueError, match=message):
        TorchAFHMM(_params()).partial_fit(mutated_mains, [("fridge", mutated_targets)])


def test_training_and_inference_reject_invalid_inputs():
    mains, appliances = _training_chunks()
    model = TorchAFHMM(_params())

    with pytest.raises(ValueError, match="at least one mains"):
        model.partial_fit([], appliances)
    with pytest.raises(ValueError, match="at least one appliance"):
        model.partial_fit(mains, [])
    with pytest.raises(TypeError, match="must contain"):
        model.partial_fit(mains, None)
    with pytest.raises(ValueError, match=r"\(name, frames\) pair"):
        model.partial_fit(mains, [("fridge", appliances[0][1], "extra")])
    with pytest.raises(ValueError, match="Duplicate appliance"):
        model.partial_fit(mains, [appliances[0], appliances[0]])
    with pytest.raises(ValueError, match="non-empty strings"):
        model.partial_fit(mains, [("", appliances[0][1])])
    with pytest.raises(ValueError, match="surrounding whitespace"):
        model.partial_fit(mains, [(" fridge ", appliances[0][1])])
    with pytest.raises(TypeError, match="must be a sequence"):
        model.partial_fit(mains, [("fridge", iter(appliances[0][1]))])
    with pytest.raises(RuntimeError, match="trained or loaded"):
        model.disaggregate_chunk(mains)
    with pytest.raises(TypeError, match="fitted HMM parameters"):
        model.disaggregate_chunk(mains, model={"fridge": object()})
    with pytest.raises(ValueError, match="boolean"):
        model.partial_fit(mains, appliances, do_preprocessing="yes")

    model.partial_fit(mains, appliances)
    with pytest.raises(ValueError, match="non-negative"):
        model.disaggregate_chunk([mains[0].assign(power=-1)])
    with pytest.raises(ValueError, match="boolean"):
        model.disaggregate_chunk(mains, do_preprocessing="yes")


def test_external_model_mapping_is_validated_sorted_and_detached():
    mains, appliances = _training_chunks()
    trained = TorchAFHMM(_params())
    trained.partial_fit(mains, appliances)
    external = {
        "kettle": trained.models["kettle"],
        "fridge": trained.models["fridge"],
    }
    fresh = TorchAFHMM(_params())

    prediction = fresh.disaggregate_chunk([mains[0]], model=external)[0]

    assert fresh.models is not external
    assert list(fresh.models) == ["kettle", "fridge"]
    assert prediction.columns.tolist() == ["fridge", "kettle"]
    external["new"] = trained.models["fridge"]
    assert "new" not in fresh.models


def test_iteration_limit_can_warn_or_fail_on_unconfirmed_convergence():
    mains, appliances = _training_chunks()
    permissive = TorchAFHMM(
        _params(inference_max_iterations=1, fail_on_nonconvergence=False)
    )
    permissive.partial_fit(mains, appliances)

    permissive.disaggregate_chunk([mains[0]])

    diagnostics = permissive.last_inference_diagnostics[0]
    assert diagnostics.iterations == 1
    assert not diagnostics.converged
    strict = TorchAFHMM(
        _params(inference_max_iterations=1, fail_on_nonconvergence=True)
    )
    strict.models = permissive.models
    with pytest.raises(RuntimeError, match="did not converge"):
        strict.disaggregate_chunk([mains[0]])


def test_checkpoint_roundtrip_restores_config_models_and_predictions(tmp_path):
    mains, appliances = _training_chunks()
    source = tmp_path / "source"
    model = TorchAFHMM(_params(save_model_path=str(source)))
    model.partial_fit(mains, appliances)
    expected = model.disaggregate_chunk(mains)

    loaded = TorchAFHMM(
        _params(
            num_states=3,
            pseudocount=9.0,
            kmeans_max_iterations=2,
            kmeans_tolerance=0.1,
            noise_std=20.0,
            inference_max_iterations=2,
            fail_on_nonconvergence=True,
            pretrained_model_path=str(source),
        )
    )

    assert loaded.num_states == 2
    assert loaded.pseudocount == 1.0
    assert loaded.kmeans_max_iterations == 50
    assert loaded.kmeans_tolerance == 1e-9
    assert loaded.noise_std == 1.0
    assert loaded.inference_max_iterations == 20
    assert loaded.fail_on_nonconvergence is False
    assert loaded.models == model.models
    _assert_same_predictions(loaded.disaggregate_chunk(mains), expected)
    artifact = source / "torch_afhmm.json"
    assert artifact.stat().st_mode & 0o777 == 0o644
    assert not list(source.glob(".torch_afhmm.json.*"))


def test_persistence_contracts_require_trained_model_and_directory(tmp_path):
    model = TorchAFHMM(_params())

    with pytest.raises(RuntimeError, match="trained or loaded"):
        model.save_model(tmp_path)
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
        (lambda payload: payload["config"].pop("noise_std"), "configuration"),
        (
            lambda payload: payload["config"].update(fail_on_nonconvergence="yes"),
            "boolean",
        ),
        (lambda payload: payload.update(models={}), "at least one appliance"),
        (
            lambda payload: payload["models"].update(
                {" fridge ": payload["models"].pop("fridge")}
            ),
            "invalid appliance name",
        ),
        (
            lambda payload: payload["config"].update(num_states=3),
            "state_means",
        ),
        (
            lambda payload: payload["models"]["fridge"].update(
                initial_probabilities=[1.0]
            ),
            "initial_probabilities",
        ),
        (
            lambda payload: payload["models"]["fridge"].update(
                transition_probabilities=[[0.5, 0.5]]
            ),
            "transition_probabilities",
        ),
        (
            lambda payload: payload["models"]["fridge"].update(state_means=[80.0, 0.0]),
            "increasing",
        ),
        (
            lambda payload: payload["models"]["fridge"][
                "initial_probabilities"
            ].__setitem__(0, 0),
            "strictly positive",
        ),
        (
            lambda payload: payload["models"]["fridge"]["transition_probabilities"][
                0
            ].__setitem__(0, -1),
            "Invalid TorchAFHMM HMM parameters",
        ),
        (
            lambda payload: payload["models"]["fridge"]["fit_loss_history"].__setitem__(
                1, 1
            ),
            "monotonically nonincreasing",
        ),
        (
            lambda payload: payload["models"]["fridge"].update(fit_converged="yes"),
            "fit diagnostics",
        ),
        (
            lambda payload: payload["models"]["fridge"].update(fit_loss_history=[0.0]),
            "fit_loss_history",
        ),
        (
            lambda payload: payload["models"]["fridge"].update(num_samples=0),
            "positive integer",
        ),
        (
            lambda payload: payload["models"]["fridge"].update(
                num_samples=1, num_chunks=2
            ),
            "at least num_chunks",
        ),
        (
            lambda payload: payload["models"]["fridge"].update(extra=True),
            "fitted-parameter fields",
        ),
        (
            lambda payload: payload["models"].update(fridge=None),
            "fitted-parameter fields",
        ),
        (
            lambda payload: payload["models"]["fridge"].update(state_means=None),
            "fitted parameters are malformed",
        ),
    ],
)
def test_malformed_artifacts_fail_closed_without_mutating_model(
    tmp_path, mutation, message
):
    mains, appliances = _training_chunks()
    source = tmp_path / "source"
    model = TorchAFHMM(_params(save_model_path=str(source)))
    model.partial_fit(mains, appliances)
    artifact = source / "torch_afhmm.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    mutation(payload)
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "torch_afhmm.json").write_text(json.dumps(payload), encoding="utf-8")
    original_models = model.models
    original_config = (model.num_states, model.noise_std)

    with pytest.raises(ValueError, match=message):
        model.load_model(corrupt)

    assert model.models is original_models
    assert (model.num_states, model.noise_std) == original_config


def test_strict_json_loader_rejects_duplicate_keys_nan_and_bad_syntax(tmp_path):
    mains, appliances = _training_chunks()
    source = tmp_path / "source"
    model = TorchAFHMM(_params(save_model_path=str(source)))
    model.partial_fit(mains, appliances)
    artifact = source / "torch_afhmm.json"
    valid = artifact.read_text(encoding="utf-8")
    corruptions = [
        valid.replace(
            '"schema_version": 1',
            '"schema_version": 1, "schema_version": 1',
            1,
        ),
        valid.replace('"noise_std": 1.0', '"noise_std": NaN', 1),
        "{not-json",
    ]

    for content in corruptions:
        artifact.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="valid TorchAFHMM artifact"):
            model.load_model(source)


def test_public_export_catalog_defaults_and_legacy_coexistence_are_explicit():
    import nilmtk_contrib.disaggregate as legacy_models
    import nilmtk_contrib.torch as torch_models
    from nilmtk_contrib.metadata import model_catalog_by_module

    default = torch_models.TorchAFHMM({"device": "cpu"})
    entry = model_catalog_by_module()["nilmtk_contrib.torch.afhmm"]

    assert torch_models.TorchAFHMM is TorchAFHMM
    assert legacy_models._EXPORTS["AFHMM"][0] == "nilmtk_contrib.disaggregate.afhmm"
    assert TorchAFHMM.__module__ == "nilmtk_contrib.torch._afhmm"
    assert default.num_states == 2
    assert default.noise_std == 100.0
    assert default.inference_max_iterations == 20
    assert entry.class_name == "TorchAFHMM"
    assert entry.exported_from == "nilmtk_contrib.torch"


def test_mps_is_rejected_before_float64_state_space_work(monkeypatch):
    import nilmtk_contrib.torch._base as base_module

    monkeypatch.setattr(
        base_module,
        "resolve_torch_device",
        lambda _requested=None: torch.device("mps"),
    )

    with pytest.raises(ValueError, match="CPU and CUDA"):
        TorchAFHMM(_params(device="mps"))


def test_implementation_uses_shared_torch_cores_without_external_solver():
    source = Path(inspect.getfile(TorchAFHMM)).read_text(encoding="utf-8")

    assert "fit_observed_gaussian_hmm" in source
    assert "factorial_hmm_coordinate_viterbi" in source
    assert "aligned_power_windows" in source
    for dependency in (
        "cvxpy",
        "hmmlearn",
        "mosek",
        "sklearn",
        "scipy",
        "multiprocessing",
    ):
        assert dependency not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fit_and_inference_match_cpu_under_strict_determinism():
    mains, appliances = _training_chunks()
    torch.use_deterministic_algorithms(True)
    try:
        cpu = TorchAFHMM(_params(device="cpu"))
        cuda = TorchAFHMM(_params(device="cuda"))
        cpu.partial_fit(mains, appliances)
        cuda.partial_fit(mains, appliances)
        cpu_predictions = cpu.disaggregate_chunk(mains)
        cuda_predictions = cuda.disaggregate_chunk(mains)
    finally:
        torch.use_deterministic_algorithms(False)

    assert cpu.models == cuda.models
    _assert_same_predictions(cuda_predictions, cpu_predictions)
