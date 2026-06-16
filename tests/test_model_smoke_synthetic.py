import pytest

from model_smoke_helpers import (
    ALL_MODEL_SPECS,
    assert_prediction_frames,
    import_or_skip,
    smoke_params,
    synthetic_nilmtk_chunks,
)


def _skip_if_smoke_disabled_or_backend_unselected(config, spec):
    if not config["enabled"]:
        pytest.skip("synthetic model smoke tests require --run-model-smoke")
    if spec.backend not in config["backends"]:
        pytest.skip(f"{spec.backend} backend not selected")


def _params_for_spec(spec, epochs):
    params = smoke_params(spec, epochs)
    if spec.backend == "classical":
        params.update(
            {
                "time_period": 20,
                "default_num_states": 2,
                "max_workers": 1,
                "shape": 20,
                "iterations": 1,
                "n_components": 2,
                "learning_rate": 1e-9,
                "sparsity_coef": 0.1,
            }
        )
    return params


@pytest.mark.parametrize(
    "spec",
    ALL_MODEL_SPECS,
    ids=lambda spec: f"{spec.backend}:{spec.module_name}.{spec.class_name}",
)
def test_model_contract_on_synthetic_dataset(spec, model_smoke_config):
    _skip_if_smoke_disabled_or_backend_unselected(model_smoke_config, spec)
    cls = import_or_skip(spec)
    params = _params_for_spec(spec, model_smoke_config["epochs"])

    model = cls(params)
    for method_name in ("partial_fit", "disaggregate_chunk"):
        assert callable(getattr(model, method_name, None))

    mains, appliances = synthetic_nilmtk_chunks(spec.min_sequence_length)
    model.partial_fit(mains, appliances)
    predictions = model.disaggregate_chunk(mains)
    assert_prediction_frames(predictions)
