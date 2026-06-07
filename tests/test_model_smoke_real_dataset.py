import pytest

from model_smoke_helpers import (
    ALL_MODEL_SPECS,
    assert_prediction_frames,
    import_or_skip,
    load_real_dataset_chunks,
    smoke_params,
)


def _skip_if_real_smoke_unavailable(config, spec):
    if not config["enabled"]:
        pytest.skip("real-dataset model smoke tests require --run-model-smoke")
    if not config["real_dataset_path"]:
        pytest.skip("real-dataset model smoke tests require --real-dataset-path")
    if spec.backend not in config["backends"]:
        pytest.skip(f"{spec.backend} backend not selected")
    if not spec.real_dataset:
        pytest.skip(f"{spec.class_name} is not configured for real-dataset smoke")


@pytest.mark.parametrize(
    "spec",
    ALL_MODEL_SPECS,
    ids=lambda spec: f"{spec.backend}:{spec.module_name}.{spec.class_name}",
)
def test_model_contract_on_real_dataset(spec, model_smoke_config):
    _skip_if_real_smoke_unavailable(model_smoke_config, spec)
    appliance = model_smoke_config["real_dataset_appliance"]
    cls = import_or_skip(spec)
    params = smoke_params(spec, model_smoke_config["epochs"])
    params["appliance_params"] = {
        appliance: {
            "mean": 100.0,
            "std": 50.0,
            "min": 0.0,
            "max": 300.0,
            "on_power_threshold": 40.0,
            "threshold": 40.0,
        }
    }
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

    mains, appliances = load_real_dataset_chunks(
        model_smoke_config["real_dataset_path"],
        model_smoke_config["real_dataset_building"],
        appliance,
        spec.min_sequence_length,
    )
    model = cls(params)
    model.partial_fit(mains, appliances)
    predictions = model.disaggregate_chunk(mains)
    assert_prediction_frames(predictions, appliance=appliance)
