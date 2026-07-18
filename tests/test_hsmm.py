import inspect
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


def test_inference_splits_long_and_partial_chunks_at_the_duration_cap(monkeypatch):
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

    assert lengths == [5, 5, 3]
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
    with pytest.raises(RuntimeError, match="trained model"):
        model.disaggregate_chunk(mains)
    with pytest.raises(ValueError, match="boolean"):
        model.partial_fit(mains, appliances, do_preprocessing="yes")


def test_persistence_is_explicitly_fail_closed_until_artifact_pr(tmp_path):
    mains, appliances = _training_chunks()
    model = HSMM(_params())
    model.partial_fit(mains, appliances)

    with pytest.raises(NotImplementedError, match="HSMM"):
        model.save_model(tmp_path)
    with pytest.raises(NotImplementedError, match="HSMM"):
        model.load_model(tmp_path)


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
