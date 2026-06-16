import pytest

from nilmtk_contrib.utils.params import (
    get_param,
    normalize_common_params,
    require_odd_sequence_length,
    validate_non_negative_int,
    validate_positive_int,
    validate_positive_number,
)


DEFAULTS = {
    "sequence_length": 99,
    "n_epochs": 10,
    "batch_size": 512,
    "mains_mean": 1000,
    "mains_std": 600,
    "appliance_params": {},
    "save_model_path": None,
    "pretrained_model_path": None,
    "chunk_wise_training": False,
    "seed": None,
    "verbose": False,
    "device": None,
}


def test_get_param_prefers_canonical_name_over_alias():
    value = get_param(
        {"sequence_length": 101, "seq_len": 99},
        "sequence_length",
        aliases=("seq_len",),
    )

    assert value == 101


def test_get_param_alias_warns():
    with pytest.warns(DeprecationWarning, match="save-model-path"):
        value = get_param(
            {"save-model-path": "old-path"},
            "save_model_path",
            aliases=("save-model-path",),
        )

    assert value == "old-path"


def test_get_param_required_missing_fails():
    with pytest.raises(ValueError, match="Missing required parameter 'sequence_length'"):
        get_param({}, "sequence_length", required=True)


def test_normalize_common_params_uses_defaults():
    params = normalize_common_params({}, DEFAULTS)

    assert params.sequence_length == 99
    assert params.n_epochs == 10
    assert params.batch_size == 512
    assert params.mains_mean == 1000
    assert params.mains_std == 600
    assert params.appliance_params == {}
    assert params.save_model_path is None
    assert params.pretrained_model_path is None
    assert params.chunk_wise_training is False
    assert params.seed is None
    assert params.verbose is False
    assert params.device is None


def test_normalize_common_params_accepts_canonical_names():
    params = normalize_common_params(
        {
            "sequence_length": 101,
            "n_epochs": 0,
            "batch_size": 64,
            "mains_mean": 500,
            "mains_std": 250,
            "appliance_params": {"fridge": {"mean": 75, "std": 25}},
            "save_model_path": "save",
            "pretrained_model_path": "load",
            "chunk_wise_training": True,
            "seed": 123,
            "verbose": True,
            "device": "cpu",
        },
        DEFAULTS,
    )

    assert params.sequence_length == 101
    assert params.n_epochs == 0
    assert params.batch_size == 64
    assert params.mains_mean == 500
    assert params.mains_std == 250
    assert params.appliance_params == {"fridge": {"mean": 75, "std": 25}}
    assert params.save_model_path == "save"
    assert params.pretrained_model_path == "load"
    assert params.chunk_wise_training is True
    assert params.seed == 123
    assert params.verbose is True
    assert params.device == "cpu"


def test_normalize_common_params_accepts_legacy_path_aliases():
    with pytest.warns(DeprecationWarning) as warnings:
        params = normalize_common_params(
            {
                "save-model-path": "save",
                "pretrained-model-path": "load",
            },
            DEFAULTS,
        )

    assert params.save_model_path == "save"
    assert params.pretrained_model_path == "load"
    assert len(warnings) == 2


@pytest.mark.parametrize("alias", ["load_model_path", "load-model-path"])
def test_normalize_common_params_accepts_load_model_aliases(alias):
    with pytest.warns(DeprecationWarning, match=alias):
        params = normalize_common_params({alias: "load"}, DEFAULTS)

    assert params.pretrained_model_path == "load"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sequence_length", 0, "sequence_length must be a positive integer"),
        ("sequence_length", 99.5, "sequence_length must be a positive integer"),
        ("n_epochs", -1, "n_epochs must be a non-negative integer"),
        ("batch_size", 0, "batch_size must be a positive integer"),
        ("mains_std", 0, "mains_std must not be zero"),
    ],
)
def test_normalize_common_params_validates_common_values(field, value, message):
    with pytest.raises(ValueError, match=message):
        normalize_common_params({field: value}, DEFAULTS)


def test_normalize_common_params_validates_appliance_std():
    with pytest.raises(ValueError, match=r"appliance_params\['fridge'\]\['std'\]"):
        normalize_common_params(
            {"appliance_params": {"fridge": {"mean": 75, "std": 0}}},
            DEFAULTS,
        )


def test_require_odd_sequence_length_accepts_odd_values():
    require_odd_sequence_length(99)


def test_require_odd_sequence_length_rejects_even_values():
    with pytest.raises(ValueError, match="sequence_length must be odd"):
        require_odd_sequence_length(100)


def test_model_specific_parameter_validators():
    assert validate_positive_int("time_period", 720) == 720
    assert validate_non_negative_int("iterations", 0) == 0
    assert validate_positive_number("learning_rate", 1e-9) == 1e-9

    with pytest.raises(ValueError, match="time_period"):
        validate_positive_int("time_period", 0)
    with pytest.raises(ValueError, match="iterations"):
        validate_non_negative_int("iterations", -1)
    with pytest.raises(ValueError, match="learning_rate"):
        validate_positive_number("learning_rate", 0)
