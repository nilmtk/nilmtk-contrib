"""Classification label helpers."""

import numpy as np


def make_on_off_labels(values, threshold):
    """Create binary on/off labels using an explicit power threshold."""
    if threshold is None:
        raise ValueError("threshold must be explicit.")
    return (np.asarray(values) >= threshold).astype(int)


def appliance_threshold(appliance_params, appliance_name, default_threshold=None):
    """Return an explicit on/off threshold for one appliance."""
    params = appliance_params.get(appliance_name, {}) if appliance_params else {}
    threshold = params.get("on_power_threshold", params.get("threshold", default_threshold))
    if threshold is None:
        raise ValueError(f"Missing on/off threshold for appliance {appliance_name!r}.")
    return threshold


def classification_metadata(appliance_params, default_threshold=None):
    """Return serializable threshold metadata for classification models."""
    metadata = {
        "default_threshold": default_threshold,
        "appliances": {},
    }
    for appliance_name in sorted((appliance_params or {}).keys()):
        metadata["appliances"][appliance_name] = {
            "on_power_threshold": appliance_threshold(
                appliance_params,
                appliance_name,
                default_threshold,
            )
        }
    return metadata


def loss_weight_metadata(regression_weight=1.0, classification_weight=1.0):
    """Return serializable loss weight metadata for dual-output models."""
    if regression_weight <= 0:
        raise ValueError("regression_weight must be positive.")
    if classification_weight <= 0:
        raise ValueError("classification_weight must be positive.")
    return {
        "regression": regression_weight,
        "classification": classification_weight,
    }
