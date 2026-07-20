from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
dsc_module = pytest.importorskip("nilmtk_contrib.torch._dsc")
DSC = dsc_module.DSC
nonnegative_sparse_code = dsc_module.nonnegative_sparse_code


def _params(**overrides):
    return {
        "device": "cpu",
        "shape": 8,
        "n_components": 2,
        "sparsity_coefficient": 0.1,
        "dictionary_iterations": 4,
        "discriminative_iterations": 4,
        "sparse_code_iterations": 300,
        "tolerance": 1e-7,
        "discriminative_learning_rate": 0.1,
        "seed": 7,
    } | overrides


def _chunks(*, length=32, start="2026-01-01"):
    index = pd.date_range(start, periods=length, freq="1min", tz="UTC")
    samples = np.arange(length)
    fridge = np.where((samples % 8 >= 2) & (samples % 8 <= 4), 80.0, 0.0)
    kettle = np.where((samples % 16 >= 11) & (samples % 16 <= 12), 180.0, 0.0)
    background = 20.0 + 0.2 * np.sin(samples)
    mains = background + fridge + kettle
    return (
        pd.DataFrame({"power": mains.astype(np.float32)}, index=index),
        pd.DataFrame({"power": fridge.astype(np.float32)}, index=index),
        pd.DataFrame({"power": kettle.astype(np.float32)}, index=index),
    )


def _training_chunks():
    first = _chunks(length=32)
    second = _chunks(length=24, start="2026-02-01")
    mains = [first[0], second[0]]
    appliances = [
        ("fridge", [first[1], second[1]]),
        ("kettle", [first[2], second[2]]),
    ]
    return mains, appliances


@pytest.fixture(scope="module")
def valid_artifact_payload(tmp_path_factory):
    mains, appliances = _training_chunks()
    path = tmp_path_factory.mktemp("dsc-artifact")
    model = DSC(_params())
    model.partial_fit(mains, appliances)
    model.save_model(path)
    return json.loads((path / "dsc.json").read_text(encoding="utf-8"))


def test_nonnegative_sparse_code_matches_identity_dictionary_closed_form():
    dictionary = torch.eye(3, dtype=torch.float64)
    observations = torch.tensor(
        [[0.1, 1.5], [2.0, 0.0], [0.8, 0.4]], dtype=torch.float64
    )

    result = nonnegative_sparse_code(
        dictionary,
        observations,
        sparsity_coefficient=0.3,
        max_iterations=50,
        tolerance=1e-12,
    )

    assert result.converged
    torch.testing.assert_close(
        result.codes,
        torch.clamp(observations - 0.3, min=0),
        atol=1e-12,
        rtol=0,
    )
    assert result.iterations <= 2


def test_nonnegative_sparse_code_matches_sklearn_lasso_cd_gold_solution():
    sklearn = pytest.importorskip("sklearn.decomposition")
    dictionary = np.array([[1.0, 0.2], [0.0, 0.98], [0.4, 0.1]], dtype=np.float64)
    dictionary /= np.linalg.norm(dictionary, axis=0)
    observations = np.array([[1.2, 0.1], [0.5, 1.5], [0.6, 0.2]], dtype=np.float64)
    coefficient = 0.15

    expected = (
        sklearn.SparseCoder(
            dictionary=dictionary.T,
            transform_algorithm="lasso_cd",
            transform_alpha=coefficient,
            positive_code=True,
        )
        .transform(observations.T)
        .T
    )
    actual = nonnegative_sparse_code(
        torch.tensor(dictionary),
        torch.tensor(observations),
        sparsity_coefficient=coefficient,
        max_iterations=500,
        tolerance=1e-10,
    )

    assert actual.converged
    assert actual.codes.cpu().numpy() == pytest.approx(expected, abs=1e-6)


def test_sparse_solver_reduces_the_zero_code_objective_and_is_deterministic():
    dictionary = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float64)
    dictionary /= torch.linalg.vector_norm(dictionary, dim=0)
    observations = torch.tensor(
        [[2.0, 0.2], [0.1, 1.5], [1.0, 1.0]], dtype=torch.float64
    )
    initial_objective = 0.5 * torch.square(observations).sum()

    first = nonnegative_sparse_code(
        dictionary,
        observations,
        sparsity_coefficient=0.1,
        max_iterations=200,
        tolerance=1e-9,
    )
    second = nonnegative_sparse_code(
        dictionary,
        observations,
        sparsity_coefficient=0.1,
        max_iterations=200,
        tolerance=1e-9,
    )

    assert first.objective < float(initial_objective)
    assert bool((first.codes >= 0).all())
    assert torch.equal(first.codes, second.codes)
    assert first.objective == second.objective
    assert first.iterations == second.iterations
    assert first.converged is second.converged


@pytest.mark.parametrize(
    ("dictionary", "observations", "kwargs", "error", "message"),
    [
        (np.ones((2, 1)), torch.ones((2, 1)), {}, TypeError, "torch.Tensor"),
        (torch.ones(2), torch.ones((2, 1)), {}, ValueError, "two-dimensional"),
        (
            torch.ones((2, 1), dtype=torch.int64),
            torch.ones((2, 1), dtype=torch.float64),
            {},
            TypeError,
            "float32 or float64",
        ),
        (
            torch.tensor([[float("nan")]], dtype=torch.float64),
            torch.ones((1, 1), dtype=torch.float64),
            {},
            ValueError,
            "finite values",
        ),
        (
            torch.ones((2, 1), dtype=torch.float32),
            torch.ones((2, 1), dtype=torch.float64),
            {},
            ValueError,
            "same dtype",
        ),
        (
            torch.ones((2, 1), dtype=torch.float64),
            torch.ones((3, 1), dtype=torch.float64),
            {},
            ValueError,
            "same row count",
        ),
        (
            torch.empty((2, 0), dtype=torch.float64),
            torch.ones((2, 1), dtype=torch.float64),
            {},
            ValueError,
            "at least one column",
        ),
        (
            torch.zeros((2, 1), dtype=torch.float64),
            torch.ones((2, 1), dtype=torch.float64),
            {},
            ValueError,
            "positive finite spectral norm",
        ),
        (
            torch.ones((2, 1), dtype=torch.float64),
            torch.tensor([[1.0], [-1.0]], dtype=torch.float64),
            {},
            ValueError,
            "non-negative",
        ),
        (
            torch.ones((2, 1), dtype=torch.float64),
            torch.ones((2, 1), dtype=torch.float64),
            {"sparsity_coefficient": -1},
            ValueError,
            "non-negative finite",
        ),
        (
            torch.ones((2, 1), dtype=torch.float64),
            torch.ones((2, 1), dtype=torch.float64),
            {"max_iterations": 0},
            ValueError,
            "positive integer",
        ),
        (
            torch.ones((2, 1), dtype=torch.float64),
            torch.ones((2, 1), dtype=torch.float64),
            {"tolerance": 0},
            ValueError,
            "positive finite",
        ),
    ],
)
def test_sparse_solver_rejects_invalid_inputs(
    dictionary, observations, kwargs, error, message
):
    defaults = {
        "sparsity_coefficient": 0.1,
        "max_iterations": 10,
        "tolerance": 1e-6,
    }
    with pytest.raises(error, match=message):
        nonnegative_sparse_code(
            dictionary,
            observations,
            **(defaults | kwargs),
        )


def test_fit_is_deterministic_and_learns_normalized_nonnegative_dictionaries():
    mains, appliances = _training_chunks()
    first = DSC(_params(seed=1))
    second = DSC(_params(seed=999))

    first.partial_fit(mains, appliances)
    second.partial_fit(mains, appliances)

    assert first.models == second.models
    assert list(first.models) == ["fridge", "kettle"]
    for fitted in first.models.values():
        reconstruction = torch.tensor(fitted.reconstruction_dictionary)
        discriminative = torch.tensor(fitted.discriminative_dictionary)
        assert bool((reconstruction >= 0).all())
        assert bool((discriminative >= 0).all())
        torch.testing.assert_close(
            torch.linalg.vector_norm(reconstruction, dim=0),
            torch.ones(2),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(discriminative, dim=0),
            torch.ones(2),
            atol=1e-6,
            rtol=0,
        )
        assert fitted.training_windows == 7
        assert np.isfinite(fitted.reconstruction_objective)
        assert fitted.reconstruction_iterations >= 1
        assert isinstance(fitted.reconstruction_converged, bool)
        assert np.isfinite(fitted.activation_error)


def test_train_infer_preserves_index_dtype_and_aggregate_conservation():
    mains, appliances = _training_chunks()
    model = DSC(_params())
    model.partial_fit(mains, appliances)

    prediction = model.disaggregate_chunk(mains)[0]

    assert prediction.index.equals(mains[0].index)
    assert prediction.columns.tolist() == ["fridge", "kettle"]
    assert all(dtype == np.float32 for dtype in prediction.dtypes)
    assert np.isfinite(prediction.to_numpy()).all()
    assert (prediction.to_numpy() >= 0).all()
    assert np.all(
        prediction.sum(axis=1).to_numpy() <= mains[0]["power"].to_numpy() + 1e-5
    )


def test_fitted_runtime_keeps_validated_power_data_in_float32(monkeypatch):
    mains, appliances = _training_chunks()
    observed_dtypes = []
    original_solver = nonnegative_sparse_code

    def recording_solver(dictionary, observations, **kwargs):
        observed_dtypes.append((dictionary.dtype, observations.dtype))
        return original_solver(dictionary, observations, **kwargs)

    monkeypatch.setattr(dsc_module, "nonnegative_sparse_code", recording_solver)
    model = DSC(_params(dictionary_iterations=0, discriminative_iterations=0))
    model.partial_fit(mains, appliances)
    model.disaggregate_chunk(mains)

    assert observed_dtypes
    assert set(observed_dtypes) == {(torch.float32, torch.float32)}


def test_partial_and_empty_inference_chunks_preserve_exact_shape_and_index():
    mains, appliances = _training_chunks()
    model = DSC(_params())
    model.partial_fit(mains, appliances)
    partial = mains[0].iloc[:13]
    empty = mains[0].iloc[:0]

    partial_prediction, empty_prediction = model.disaggregate_chunk([partial, empty])

    assert partial_prediction.index.equals(partial.index)
    assert len(partial_prediction) == 13
    assert empty_prediction.index.equals(empty.index)
    assert empty_prediction.columns.tolist() == ["fridge", "kettle"]
    assert empty_prediction.empty


def test_failed_fit_does_not_replace_a_previous_valid_model():
    mains, appliances = _training_chunks()
    model = DSC(_params())
    model.partial_fit(mains, appliances)
    original = model.models
    constant_zero = appliances[0][1][0].copy()
    constant_zero.iloc[:] = 0

    with pytest.raises(ValueError, match="positive power"):
        model.partial_fit(mains[:1], [("zero", [constant_zero])])

    assert model.models is original


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"shape": 0}, ValueError, "positive integer"),
        ({"n_components": True}, ValueError, "positive integer"),
        ({"sparsity_coefficient": -1}, ValueError, "non-negative finite"),
        ({"dictionary_iterations": -1}, ValueError, "non-negative integer"),
        ({"discriminative_iterations": -1}, ValueError, "non-negative integer"),
        ({"sparse_code_iterations": 0}, ValueError, "positive integer"),
        ({"tolerance": float("nan")}, ValueError, "positive finite"),
        (
            {"discriminative_learning_rate": 0},
            ValueError,
            "positive finite",
        ),
        ({"enforce_aggregate": "yes"}, ValueError, "boolean"),
        ({"chunk_wise_training": True}, ValueError, "does not support"),
    ],
)
def test_invalid_configuration_fails_at_construction(overrides, error, message):
    with pytest.raises(error, match=message):
        DSC(_params(**overrides))


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        (lambda mains, apps: ([], apps), ValueError, "mains chunk"),
        (lambda mains, apps: (mains, []), ValueError, "one appliance"),
        (
            lambda mains, apps: (mains, [("fridge", apps[0][1][:-1])]),
            ValueError,
            "chunks but mains",
        ),
        (
            lambda mains, apps: (
                mains,
                [("fridge", [apps[0][1][0].iloc[:-1], apps[0][1][1]])],
            ),
            ValueError,
            "length does not match",
        ),
        (
            lambda mains, apps: (
                mains,
                [
                    ("fridge", apps[0][1]),
                    ("fridge", apps[0][1]),
                ],
            ),
            ValueError,
            "Duplicate appliance",
        ),
    ],
)
def test_invalid_training_data_fails_explicitly(mutation, error, message):
    mains, appliances = _training_chunks()
    mutated_mains, mutated_appliances = mutation(mains, appliances)
    with pytest.raises(error, match=message):
        DSC(_params()).partial_fit(mutated_mains, mutated_appliances)


@pytest.mark.parametrize(
    ("train_main", "train_appliances", "error", "message"),
    [
        (None, [], TypeError, "train_main must contain"),
        ([], None, ValueError, "mains chunk"),
        ([object()], 1, TypeError, "train_appliances must contain"),
    ],
)
def test_training_rejects_non_iterable_inputs(
    train_main, train_appliances, error, message
):
    with pytest.raises(error, match=message):
        DSC(_params()).partial_fit(train_main, train_appliances)


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        (
            lambda mains, apps: (mains, [("fridge", apps[0][1], "extra")]),
            ValueError,
            r"\(name, frames\) pair",
        ),
        (
            lambda mains, apps: (mains, [(1, apps[0][1])]),
            ValueError,
            "non-empty strings",
        ),
        (
            lambda mains, apps: (mains, [(" fridge", apps[0][1])]),
            ValueError,
            "surrounding whitespace",
        ),
        (
            lambda mains, apps: (mains, [("fridge", "frames")]),
            TypeError,
            "must be a sequence",
        ),
        (
            lambda mains, apps: (
                mains,
                [
                    (
                        "fridge",
                        [
                            apps[0][1][0].set_axis(
                                apps[0][1][0].index.shift(1, freq="1min")
                            ),
                            apps[0][1][1],
                        ],
                    )
                ],
            ),
            ValueError,
            "index does not match",
        ),
        (
            lambda mains, apps: (
                mains,
                [
                    (
                        "fridge",
                        [apps[0][1][0] * 0 - 1, apps[0][1][1]],
                    )
                ],
            ),
            ValueError,
            "must be non-negative",
        ),
    ],
)
def test_training_rejects_malformed_entries(mutation, error, message):
    mains, appliances = _training_chunks()
    mutated_mains, mutated_appliances = mutation(mains, appliances)
    with pytest.raises(error, match=message):
        DSC(_params()).partial_fit(mutated_mains, mutated_appliances)


def test_preprocessing_flags_are_strict_booleans():
    with pytest.raises(ValueError, match="do_preprocessing must be a boolean"):
        DSC(_params()).partial_fit([], [], do_preprocessing=1)
    with pytest.raises(ValueError, match="do_preprocessing must be a boolean"):
        DSC(_params()).disaggregate_chunk([], do_preprocessing="yes")


def test_training_pads_each_chunk_independently_without_cross_chunk_windows():
    first = _chunks(length=13)
    second = _chunks(length=10, start="2026-02-01")
    model = DSC(_params(dictionary_iterations=0, discriminative_iterations=0))

    model.partial_fit(
        [first[0], second[0]],
        [("fridge", [first[1], second[1]])],
    )

    assert model.models["fridge"].training_windows == 4


def test_inference_requires_training_and_validates_external_models():
    mains, appliances = _training_chunks()
    model = DSC(_params())
    with pytest.raises(RuntimeError, match="trained or loaded"):
        model.disaggregate_chunk(mains)

    model.partial_fit(mains, appliances)
    with pytest.raises(ValueError, match="models must contain"):
        model.disaggregate_chunk(mains, model={})
    with pytest.raises(TypeError, match="fitted DSC parameters"):
        model.disaggregate_chunk(mains, model={"fridge": object()})


def test_save_load_round_trip_preserves_predictions_and_host_readable_json(tmp_path):
    mains, appliances = _training_chunks()
    trained = DSC(_params())
    trained.partial_fit(mains, appliances)
    expected = trained.disaggregate_chunk(mains)[0]

    trained.save_model(tmp_path)
    artifact = tmp_path / "dsc.json"
    loaded = DSC(_params(pretrained_model_path=str(tmp_path)))
    actual = loaded.disaggregate_chunk(mains)[0]

    pd.testing.assert_frame_equal(actual, expected)
    assert loaded.models == trained.models
    assert artifact.stat().st_mode & 0o777 == 0o644
    assert json.loads(artifact.read_text())["model_class"] == "DSC"


def test_configured_save_path_is_written_transactionally(tmp_path):
    mains, appliances = _training_chunks()
    model = DSC(_params(save_model_path=str(tmp_path)))

    model.partial_fit(mains, appliances)

    assert (tmp_path / "dsc.json").is_file()


def test_save_and_load_require_an_explicit_or_configured_path():
    model = DSC(_params())
    with pytest.raises(ValueError, match="save_model requires"):
        model.save_model()
    with pytest.raises(ValueError, match="load_model requires"):
        model.load_model()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload | {"schema_version": True}, "schema_version"),
        (lambda payload: payload | {"model_class": "Other"}, "model_class"),
        (lambda payload: payload | {"extra": 1}, "top-level"),
        (
            lambda payload: payload
            | {"config": payload["config"] | {"n_components": 0}},
            "positive integer",
        ),
        (lambda payload: payload | {"models": {}}, "at least one appliance"),
    ],
)
def test_load_rejects_malformed_artifacts(
    tmp_path, valid_artifact_payload, mutator, message
):
    path = tmp_path / "dsc.json"
    payload = deepcopy(valid_artifact_payload)
    path.write_text(json.dumps(mutator(payload)), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        DSC(_params(pretrained_model_path=str(tmp_path)))


def _first_fitted_payload(payload):
    return next(iter(payload["models"].values()))


def _rename_first_appliance(payload, name):
    old_name = next(iter(payload["models"]))
    payload["models"][name] = payload["models"].pop(old_name)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload["config"].pop("shape"), "configuration fields"),
        (
            lambda payload: payload["config"].__setitem__("enforce_aggregate", "yes"),
            "enforce_aggregate must be a boolean",
        ),
        (
            lambda payload: _rename_first_appliance(payload, " fridge"),
            "invalid appliance name",
        ),
        (
            lambda payload: _first_fitted_payload(payload).__setitem__("extra", 1),
            "invalid fitted-parameter fields",
        ),
        (
            lambda payload: _first_fitted_payload(payload).__setitem__(
                "reconstruction_dictionary", None
            ),
            "fitted parameters are malformed",
        ),
        (
            lambda payload: _first_fitted_payload(payload).__setitem__(
                "reconstruction_dictionary", []
            ),
            "must contain exactly",
        ),
        (
            lambda payload: _first_fitted_payload(payload)["reconstruction_dictionary"][
                0
            ].clear(),
            "rows must contain exactly",
        ),
        (
            lambda payload: _first_fitted_payload(payload).__setitem__(
                "reconstruction_dictionary",
                [[0.0, 0.0] for _ in range(payload["config"]["shape"])],
            ),
            "columns must have unit norm",
        ),
        (
            lambda payload: _first_fitted_payload(payload).__setitem__(
                "reconstruction_converged", 1
            ),
            "reconstruction_converged must be a boolean",
        ),
    ],
)
def test_load_rejects_invalid_config_and_fitted_records(
    tmp_path, valid_artifact_payload, mutator, message
):
    payload = deepcopy(valid_artifact_payload)
    mutator(payload)
    (tmp_path / "dsc.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        DSC(_params(pretrained_model_path=str(tmp_path)))


def test_load_rejects_duplicate_keys_non_finite_values_and_oversized_files(
    tmp_path, monkeypatch
):
    path = tmp_path / "dsc.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        DSC(_params(pretrained_model_path=str(tmp_path)))

    path.write_text('{"schema_version":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON constant"):
        DSC(_params(pretrained_model_path=str(tmp_path)))

    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(dsc_module, "_MAX_ARTIFACT_BYTES", 1)
    with pytest.raises(ValueError, match="safety limit"):
        DSC(_params(pretrained_model_path=str(tmp_path)))


def test_runtime_has_no_sklearn_cvxpy_or_generic_solver_dependency():
    source = Path(inspect.getsourcefile(DSC)).read_text(encoding="utf-8")
    assert "import sklearn" not in source
    assert "import cvxpy" not in source
    assert "SparseCoder" not in source
    assert "problem.solve" not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_sparse_solver_and_dsc_inference_run_on_cuda():
    dictionary = torch.eye(2, dtype=torch.float64, device="cuda")
    observations = torch.ones((2, 2), dtype=torch.float64, device="cuda")
    result = nonnegative_sparse_code(
        dictionary,
        observations,
        sparsity_coefficient=0.1,
        max_iterations=20,
        tolerance=1e-8,
    )
    assert result.codes.is_cuda

    with pytest.raises(ValueError, match="same device"):
        nonnegative_sparse_code(
            dictionary,
            observations.cpu(),
            sparsity_coefficient=0.1,
        )

    mains, appliances = _training_chunks()
    model = DSC(_params(device="cuda", dictionary_iterations=1))
    model.partial_fit(mains, appliances)
    prediction = model.disaggregate_chunk(mains)[0]
    assert np.isfinite(prediction.to_numpy()).all()
