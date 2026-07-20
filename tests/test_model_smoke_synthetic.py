import pytest

from model_smoke_helpers import (
    ALL_MODEL_SPECS,
    assert_prediction_frames,
    instantiate_for_smoke,
    smoke_params,
    synthetic_nilmtk_chunks,
)


def _skip_if_smoke_disabled_or_backend_unselected(config, spec):
    if not config["enabled"]:
        pytest.skip("synthetic model smoke tests require --run-model-smoke")
    if spec.backend not in config["backends"]:
        pytest.skip(f"{spec.backend} backend not selected")


@pytest.mark.parametrize(
    "spec",
    ALL_MODEL_SPECS,
    ids=lambda spec: f"{spec.backend}:{spec.module_name}.{spec.class_name}",
)
def test_model_contract_on_synthetic_dataset(spec, model_smoke_config):
    _skip_if_smoke_disabled_or_backend_unselected(model_smoke_config, spec)
    params = smoke_params(spec, model_smoke_config["epochs"])

    model = instantiate_for_smoke(spec, params)
    for method_name in ("partial_fit", "disaggregate_chunk"):
        assert callable(getattr(model, method_name, None))

    mains, appliances = synthetic_nilmtk_chunks(spec.min_sequence_length)
    model.partial_fit(mains, appliances)
    predictions = model.disaggregate_chunk(mains)
    assert_prediction_frames(predictions)
