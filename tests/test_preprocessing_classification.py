import pytest

from nilmtk_contrib.preprocessing.classification import (
    appliance_threshold,
    classification_metadata,
    loss_weight_metadata,
    make_on_off_labels,
)


def test_appliance_threshold_prefers_appliance_specific_value():
    params = {"fridge": {"on_power_threshold": 25}}

    assert appliance_threshold(params, "fridge", default_threshold=15) == 25


def test_appliance_threshold_requires_explicit_threshold():
    with pytest.raises(ValueError, match="Missing on/off threshold"):
        appliance_threshold({}, "fridge")


def test_classification_metadata_is_serializable():
    metadata = classification_metadata(
        {
            "fridge": {"on_power_threshold": 25},
            "kettle": {"threshold": 1000},
        },
        default_threshold=15,
    )

    assert metadata == {
        "default_threshold": 15,
        "appliances": {
            "fridge": {"on_power_threshold": 25},
            "kettle": {"on_power_threshold": 1000},
        },
    }


def test_loss_weight_metadata_rejects_non_positive_weights():
    assert loss_weight_metadata(2.0, 0.5) == {
        "regression": 2.0,
        "classification": 0.5,
    }
    with pytest.raises(ValueError, match="regression_weight"):
        loss_weight_metadata(0, 1)
    with pytest.raises(ValueError, match="classification_weight"):
        loss_weight_metadata(1, 0)


def test_make_on_off_labels_uses_explicit_threshold():
    assert make_on_off_labels([1, 15, 16], threshold=15).tolist() == [0, 1, 1]
