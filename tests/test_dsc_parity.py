from collections import OrderedDict
import inspect
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

from model_smoke_helpers import load_real_dataset_chunks


torch = pytest.importorskip("torch")
sklearn_decomposition = pytest.importorskip("sklearn.decomposition")

from nilmtk_contrib.disaggregate.dsc import DSC as CompatibilityDSC  # noqa: E402
from nilmtk_contrib.torch._dsc import (  # noqa: E402
    DSC as TorchDSC,
    _DSCParameters,
    _discriminative_dictionary_step,
    _fit_discriminative_dictionary,
    nonnegative_sparse_code,
)


def _compatibility_params(**overrides):
    return {
        "device": "cpu",
        "shape": 12,
        "n_components": 2,
        "sparsity_coef": 0.1,
        "iterations": 2,
        "learning_rate": 1e-9,
        "dictionary_iterations": 20,
        "sparse_code_iterations": 500,
        "tolerance": 1e-7,
        "seed": 4,
        "verbose": False,
    } | overrides


def _torch_params(**overrides):
    return {
        "device": "cpu",
        "shape": 12,
        "n_components": 2,
        "sparsity_coefficient": 0.1,
        "dictionary_iterations": 20,
        "discriminative_iterations": 2,
        "discriminative_learning_rate": 1e-9,
        "sparse_code_iterations": 500,
        "tolerance": 1e-7,
        "seed": 4,
    } | overrides


def _normalized_columns(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.linalg.norm(values, axis=0)


def _fitted_parameters(dictionary, column):
    matrix = tuple((float(value),) for value in dictionary[:, column])
    return _DSCParameters(
        reconstruction_dictionary=matrix,
        discriminative_dictionary=matrix,
        training_windows=3,
        reconstruction_objective=0.0,
        reconstruction_iterations=1,
        reconstruction_converged=True,
        activation_error=0.0,
    )


def _synthetic_train_test(seed=1):
    rng = np.random.default_rng(seed)
    first = np.r_[
        np.zeros(2),
        np.linspace(0.2, 1.0, 4),
        np.linspace(0.8, 0.1, 4),
        np.zeros(2),
    ]
    second = np.r_[
        np.zeros(6),
        np.linspace(0.1, 1.0, 3),
        np.linspace(0.7, 0.0, 3),
    ]
    bases = np.stack([first, second], axis=1)

    def codes(scale, windows, off_probability):
        result = rng.gamma(2.0, scale, size=(2, windows))
        result *= rng.random((2, windows)) > off_probability
        return result

    def frames(first_codes, second_codes, *, start):
        first_power = (bases @ first_codes).T.reshape(-1).astype(np.float32)
        second_power = (bases @ second_codes).T.reshape(-1).astype(np.float32)
        mains = first_power + second_power + 5.0
        index = pd.date_range(start, periods=len(mains), freq="1min", tz="UTC")
        return tuple(
            pd.DataFrame({"power": values}, index=index)
            for values in (mains, first_power, second_power)
        )

    train = frames(
        codes(35.0, 30, 0.25),
        codes(55.0, 30, 0.4),
        start="2026-01-01",
    )
    test = frames(
        codes(35.0, 12, 0.25),
        codes(55.0, 12, 0.4),
        start="2026-02-01",
    )
    return train, test


def _fit_predictions(train, test):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compatibility = CompatibilityDSC(_compatibility_params())
    appliances = [("first", [train[1]]), ("second", [train[2]])]
    compatibility.partial_fit([train[0]], appliances)
    compatibility_prediction = compatibility.disaggregate_chunk([test[0]])[0]
    modern = TorchDSC(_torch_params())
    modern.partial_fit([train[0]], appliances)
    return (
        compatibility_prediction,
        modern.disaggregate_chunk([test[0]])[0],
    )


def test_compatibility_wrapper_uses_torch_with_historical_defaults():
    with pytest.warns(FutureWarning, match="maintained PyTorch implementation"):
        model = CompatibilityDSC({"device": "cpu"})

    assert isinstance(model, TorchDSC)
    assert model.shape == 120
    assert model.n_components == 10
    assert model.sparsity_coef == 20.0
    assert model.iterations == model.n_epochs == 3_000
    assert model.learning_rate == 1e-9
    model.iterations = 7
    assert model.iterations == model.n_epochs == 7
    source = Path(inspect.getsourcefile(CompatibilityDSC)).read_text(encoding="utf-8")
    for dependency in ("numpy", "pandas", "sklearn", "scipy"):
        assert dependency not in source


def test_compatibility_wrapper_rejects_non_mapping_parameters():
    with pytest.warns(FutureWarning):
        with pytest.raises(TypeError, match="mapping or None"):
            CompatibilityDSC([])


def test_compatibility_wrapper_accepts_none_with_historical_defaults():
    with pytest.warns(FutureWarning):
        model = CompatibilityDSC()

    assert model.iterations == model.n_epochs == 3_000


def test_package_level_dsc_export_resolves_to_the_compatibility_wrapper():
    import nilmtk_contrib.disaggregate as historical_models

    historical_models.__dict__.pop("DSC", None)
    exported = historical_models.DSC

    assert exported is CompatibilityDSC
    assert historical_models._EXPORTS["DSC"] == (
        "nilmtk_contrib.disaggregate.dsc",
        "DSC",
        "torch",
    )
    with pytest.warns(FutureWarning, match="maintained PyTorch implementation"):
        assert isinstance(
            exported({"device": "cpu", "discriminative_iterations": 0}),
            TorchDSC,
        )


def test_legacy_parameter_names_preserve_configuration_with_warnings():
    with pytest.warns(DeprecationWarning) as recorded:
        model = TorchDSC(
            {
                "device": "cpu",
                "sparsity_coef": 0.25,
                "iterations": 7,
                "learning_rate": 2e-8,
            }
        )

    messages = {str(item.message) for item in recorded}
    assert messages == {
        "Parameter 'iterations' is deprecated; use "
        "'discriminative_iterations' instead.",
        "Parameter 'learning_rate' is deprecated; use "
        "'discriminative_learning_rate' instead.",
        "Parameter 'sparsity_coef' is deprecated; use 'sparsity_coefficient' instead.",
    }
    assert model.sparsity_coefficient == model.sparsity_coef == 0.25
    assert model.discriminative_iterations == model.iterations == 7
    assert model.discriminative_learning_rate == model.learning_rate == 2e-8


def test_legacy_parameter_attributes_remain_mutable_and_validated():
    model = TorchDSC({"device": "cpu"})

    model.iterations = 7
    model.learning_rate = 2e-8
    model.sparsity_coef = 0.25

    assert model.discriminative_iterations == 7
    assert model.discriminative_learning_rate == 2e-8
    assert model.sparsity_coefficient == 0.25
    with pytest.raises(ValueError, match="non-negative integer"):
        model.iterations = -1
    with pytest.raises(ValueError, match="positive finite"):
        model.learning_rate = 0
    with pytest.raises(ValueError, match="non-negative finite"):
        model.sparsity_coef = float("nan")


@pytest.mark.parametrize(("seed", "coefficient"), [(1, 0.02), (2, 0.2), (3, 0.8)])
def test_sparse_codes_match_sklearn_across_conditioned_positive_lasso_fixtures(
    seed, coefficient
):
    rng = np.random.default_rng(seed)
    dictionary = 0.02 * rng.random((7, 3))
    dictionary[:3] += np.eye(3)
    dictionary = _normalized_columns(dictionary)
    true_codes = rng.gamma(2.0, 0.8, size=(3, 6))
    observations = dictionary @ true_codes + 0.002 * rng.random((7, 6))

    expected = (
        sklearn_decomposition.SparseCoder(
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
        max_iterations=2_000,
        tolerance=1e-11,
    )

    assert actual.converged
    np.testing.assert_allclose(actual.codes.numpy(), expected, rtol=2e-5, atol=2e-5)


def test_sparse_code_regularization_monotonically_reduces_mass_and_support():
    """Higher L1 penalties must produce genuinely sparser positive codes."""
    dictionary = torch.eye(4, dtype=torch.float64)
    observations = torch.tensor(
        [[0.05, 0.8], [0.2, 1.1], [0.6, 1.6], [1.4, 2.2]],
        dtype=torch.float64,
    )
    coefficients = (0.0, 0.25, 0.75, 1.5)
    results = [
        nonnegative_sparse_code(
            dictionary,
            observations,
            sparsity_coefficient=coefficient,
            max_iterations=50,
            tolerance=1e-12,
        )
        for coefficient in coefficients
    ]

    for coefficient, result in zip(coefficients, results, strict=True):
        assert result.converged
        torch.testing.assert_close(
            result.codes,
            torch.clamp(observations - coefficient, min=0),
            atol=1e-12,
            rtol=0,
        )

    l1_mass = [float(result.codes.sum()) for result in results]
    support = [int(torch.count_nonzero(result.codes)) for result in results]
    assert all(left >= right for left, right in zip(l1_mass, l1_mass[1:]))
    assert all(left >= right for left, right in zip(support, support[1:]))
    assert l1_mass[0] > l1_mass[-1]
    assert support[0] > support[-1]


def test_discriminative_basis_step_matches_legacy_t1_minus_t2_equation():
    dictionary = _normalized_columns([[1.0, 0.2], [0.1, 1.0], [0.4, 0.3], [0.2, 0.6]])
    aggregate = np.array(
        [[1.4, 0.2, 1.1], [0.3, 1.5, 0.8], [0.7, 0.4, 0.9], [0.5, 1.0, 0.6]]
    )
    predicted_codes = np.array([[1.0, 0.1, 0.7], [0.2, 1.1, 0.4]])
    target_codes = np.array([[0.8, 0.2, 0.9], [0.3, 1.0, 0.2]])
    learning_rate = 0.03

    t1 = (aggregate - dictionary @ predicted_codes) @ predicted_codes.T
    t2 = (aggregate - dictionary @ target_codes) @ target_codes.T
    expected = np.clip(dictionary - learning_rate * (t1 - t2), 0, None)
    expected /= np.linalg.norm(expected, axis=0)
    actual = _discriminative_dictionary_step(
        torch.tensor(aggregate),
        torch.tensor(dictionary),
        torch.tensor(predicted_codes),
        torch.tensor(target_codes),
        learning_rate=learning_rate,
        fallback=torch.tensor(dictionary).mean(dim=1),
    )

    assert actual is not None
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)


def test_discriminative_fit_uses_all_windows_when_twenty_percent_rounds_to_zero(
    monkeypatch,
):
    import nilmtk_contrib.torch._dsc as dsc_module

    dictionary = torch.tensor(
        [[1.0, 0.2], [0.1, 1.0], [0.4, 0.3]], dtype=torch.float64
    )
    dictionary /= torch.linalg.vector_norm(dictionary, dim=0)
    target_codes = torch.tensor(
        [[1.0, 0.2, 0.8, 0.4], [0.1, 1.1, 0.3, 0.9]], dtype=torch.float64
    )
    aggregate = dictionary @ target_codes + 0.05
    observed_windows = []
    original_solver = nonnegative_sparse_code

    def recording_solver(dictionary_value, observations, **kwargs):
        observed_windows.append(observations.shape[1])
        return original_solver(dictionary_value, observations, **kwargs)

    monkeypatch.setattr(dsc_module, "nonnegative_sparse_code", recording_solver)
    fitted, error = _fit_discriminative_dictionary(
        aggregate,
        dictionary,
        target_codes,
        sparsity_coefficient=0.05,
        discriminative_iterations=2,
        sparse_code_iterations=500,
        tolerance=1e-9,
        learning_rate=1e-3,
    )

    assert observed_windows
    assert set(observed_windows) == {4}
    assert torch.isfinite(fitted).all()
    assert np.isfinite(error)


def test_fixed_dictionary_partial_inference_matches_canonical_torch_path():
    shape = 4
    coefficient = 0.05
    joint_dictionary = _normalized_columns(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    )
    codes = np.array([[2.0, 4.0, 1.0], [3.0, 1.0, 5.0]])
    mains_values = (joint_dictionary @ codes).T.reshape(-1)[:10].astype(np.float32)
    index = pd.date_range("2026-03-01", periods=10, freq="1min", tz="UTC")
    mains = pd.DataFrame({"power": mains_values}, index=index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compatibility = CompatibilityDSC(
            _compatibility_params(
                device="cpu",
                shape=shape,
                n_components=1,
                sparsity_coef=coefficient,
                iterations=0,
                sparse_code_iterations=2_000,
                tolerance=1e-9,
            )
        )
    modern = TorchDSC(
        _torch_params(
            shape=shape,
            n_components=1,
            sparsity_coefficient=coefficient,
            sparse_code_iterations=2_000,
            tolerance=1e-9,
        )
    )

    models = OrderedDict(
        (
            ("first", _fitted_parameters(joint_dictionary, 0)),
            ("second", _fitted_parameters(joint_dictionary, 1)),
        )
    )

    compatibility_prediction = compatibility.disaggregate_chunk(
        [mains], model=models
    )[0]
    modern_prediction = modern.disaggregate_chunk([mains], model=models)[0]

    np.testing.assert_array_equal(
        modern_prediction.to_numpy(), compatibility_prediction.to_numpy()
    )
    assert modern_prediction.index.equals(index)
    assert compatibility_prediction.index.equals(index)


def test_compatibility_load_keeps_legacy_epoch_alias_in_sync(tmp_path):
    dictionary = _normalized_columns([[1.0], [0.5], [0.2], [0.1]])
    canonical = TorchDSC(
        _torch_params(
            shape=4,
            n_components=1,
            discriminative_iterations=3,
        )
    )
    canonical.models = OrderedDict(
        (("fridge", _fitted_parameters(dictionary, 0)),)
    )
    canonical.save_model(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compatibility = CompatibilityDSC(
            _compatibility_params(
                shape=4,
                n_components=1,
                iterations=7,
            )
        )

    compatibility.load_model(tmp_path)

    assert compatibility.iterations == compatibility.n_epochs == 3


@pytest.mark.parametrize("fixture_seed", [1, 7, 19])
def test_compatibility_and_canonical_dsc_match_on_deterministic_nilm_fixture(
    fixture_seed,
):
    train, test = _synthetic_train_test(seed=fixture_seed)
    compatibility_prediction, modern_prediction = _fit_predictions(train, test)
    truth = np.column_stack((test[1]["power"], test[2]["power"]))
    baseline = np.broadcast_to(np.mean(truth, axis=0), truth.shape)
    modern_mae = float(np.mean(np.abs(modern_prediction.to_numpy() - truth)))
    baseline_mae = float(np.mean(np.abs(baseline - truth)))

    assert np.isfinite([modern_mae, baseline_mae]).all()
    np.testing.assert_array_equal(
        compatibility_prediction.to_numpy(), modern_prediction.to_numpy()
    )
    assert compatibility_prediction.index.equals(modern_prediction.index)
    assert modern_mae <= baseline_mae


def test_compatibility_and_canonical_dsc_match_on_supplied_real_dataset(
    model_smoke_config,
):
    if not model_smoke_config["enabled"]:
        pytest.skip("real DSC parity requires --run-model-smoke")
    path = model_smoke_config["real_dataset_path"]
    if not path:
        pytest.skip("real DSC parity requires --real-dataset-path")
    appliance = model_smoke_config["real_dataset_appliance"]
    mains, appliances = load_real_dataset_chunks(
        path,
        model_smoke_config["real_dataset_building"],
        appliance,
        sequence_length=20,
        max_samples=800,
    )
    target = appliances[0][1][0]
    split = (int(len(mains[0]) * 0.7) // 20) * 20
    # Keep this quality comparison to full windows. Partial-window behavior has
    # its own exact parity test and must not change the model-quality sample set.
    test_length = ((len(mains[0]) - split) // 20) * 20
    if split < 100 or test_length < 40:
        pytest.skip("real DSC parity needs at least 140 aligned samples")
    train = (mains[0].iloc[:split], target.iloc[:split])
    test = (
        mains[0].iloc[split : split + test_length],
        target.iloc[split : split + test_length],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compatibility = CompatibilityDSC(
            _compatibility_params(
                shape=20,
                n_components=2,
                iterations=2,
                dictionary_iterations=8,
            )
        )
    compatibility.partial_fit([train[0]], [(appliance, [train[1]])])
    compatibility_values = compatibility.disaggregate_chunk([test[0]])[0][
        appliance
    ].to_numpy()
    modern = TorchDSC(
        _torch_params(
            shape=20,
            n_components=2,
            dictionary_iterations=8,
            discriminative_iterations=2,
        )
    )
    modern.partial_fit([train[0]], [(appliance, [train[1]])])
    modern_values = modern.disaggregate_chunk([test[0]])[0][appliance].to_numpy()
    truth = test[1]["power"].to_numpy()
    compatibility_mae = float(np.mean(np.abs(compatibility_values - truth)))
    modern_mae = float(np.mean(np.abs(modern_values - truth)))

    assert np.isfinite([compatibility_mae, modern_mae]).all()
    np.testing.assert_array_equal(compatibility_values, modern_values)
    assert compatibility_mae == modern_mae
