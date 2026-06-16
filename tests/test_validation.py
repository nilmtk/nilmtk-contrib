import numpy as np
import pandas as pd

from nilmtk_contrib.utils.validation import (
    safe_train_test_split,
    should_train,
    train_validation_split,
)


def test_should_train_reports_skip_reason():
    decision = should_train(num_samples=1, min_samples=2)

    assert decision.should_train is False
    assert decision.num_samples == 1
    assert decision.min_samples == 2
    assert "below" in decision.reason


def test_should_train_reports_trainable_input():
    decision = should_train(num_samples=2, min_samples=2)

    assert decision.should_train is True
    assert decision.reason == "enough samples to train."


def test_tail_split_guarantees_validation_sample():
    split = train_validation_split(
        np.arange(10),
        np.arange(10) + 100,
        validation_fraction=0.01,
    )

    assert split.metadata.should_train is True
    assert split.metadata.validation_enabled is True
    assert split.metadata.train_size == 9
    assert split.metadata.validation_size == 1
    assert split.X_train.tolist() == list(range(9))
    assert split.X_val.tolist() == [9]
    assert split.y_val.tolist() == [109]


def test_tiny_dataset_skips_when_validation_is_required():
    split = train_validation_split(
        np.asarray([1]),
        np.asarray([10]),
        min_train=1,
        min_val=1,
        allow_no_validation=False,
    )

    assert split.metadata.should_train is False
    assert split.metadata.validation_enabled is False
    assert split.X_train is None
    assert "min_train + min_val" in split.metadata.reason


def test_tiny_dataset_can_train_without_validation_when_allowed():
    split = train_validation_split(
        np.asarray([1]),
        np.asarray([10]),
        min_train=1,
        min_val=1,
        allow_no_validation=True,
    )

    assert split.metadata.should_train is True
    assert split.metadata.validation_enabled is False
    assert split.metadata.train_size == 1
    assert split.metadata.validation_size == 0
    assert split.X_train.tolist() == [1]
    assert split.X_val.size == 0


def test_empty_dataset_skips_even_when_no_validation_allowed():
    split = train_validation_split(
        np.asarray([]),
        np.asarray([]),
        min_train=1,
        min_val=1,
        allow_no_validation=True,
    )

    assert split.metadata.should_train is False
    assert split.metadata.train_size == 0
    assert split.X_train is None


def test_random_split_is_deterministic_with_seed():
    first = train_validation_split(
        np.arange(20),
        np.arange(20),
        validation_fraction=0.25,
        strategy="random",
        seed=123,
    )
    second = train_validation_split(
        np.arange(20),
        np.arange(20),
        validation_fraction=0.25,
        strategy="random",
        seed=123,
    )

    assert first.X_train.tolist() == second.X_train.tolist()
    assert first.X_val.tolist() == second.X_val.tolist()
    assert first.metadata.validation_size == 5


def test_split_preserves_pandas_objects_and_indices():
    X = pd.DataFrame({"mains": [1, 2, 3, 4]}, index=list("abcd"))
    y = pd.Series([10, 20, 30, 40], index=list("abcd"), name="fridge")

    split = train_validation_split(X, y, validation_fraction=0.25)

    assert isinstance(split.X_train, pd.DataFrame)
    assert isinstance(split.y_val, pd.Series)
    assert split.X_train.index.tolist() == ["a", "b", "c"]
    assert split.y_val.index.tolist() == ["d"]
    assert split.y_val.name == "fridge"


def test_split_supports_plain_lists():
    split = train_validation_split(
        ["a", "b", "c", "d"],
        [1, 2, 3, 4],
        validation_fraction=0.5,
    )

    assert split.X_train == ["a", "b"]
    assert split.y_train == [1, 2]
    assert split.X_val == ["c", "d"]
    assert split.y_val == [3, 4]


def test_split_rejects_invalid_arguments():
    invalid_cases = [
        {"strategy": "middle"},
        {"validation_fraction": 0},
        {"validation_fraction": 1},
        {"min_train": 0},
        {"min_val": 0},
    ]

    for kwargs in invalid_cases:
        try:
            train_validation_split(np.arange(3), np.arange(3), **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {kwargs}")


def test_split_rejects_length_mismatch():
    try:
        train_validation_split(np.arange(3), np.arange(2))
    except ValueError as exc:
        assert "same number of samples" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_split_rejects_unsized_inputs():
    try:
        train_validation_split(1, 1)
    except ValueError as exc:
        assert "sized collections" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_safe_train_test_split_guarantees_validation_when_possible():
    train_x, val_x, train_y, val_y = safe_train_test_split(
        np.arange(3),
        np.arange(3) + 10,
        test_size=0.15,
        random_state=1,
    )

    assert len(train_x) == 2
    assert len(val_x) == 1
    assert len(train_y) == 2
    assert len(val_y) == 1


def test_safe_train_test_split_handles_single_sample():
    train_x, val_x = safe_train_test_split(np.asarray([1]), test_size=0.15)

    assert train_x.tolist() == [1]
    assert val_x.size == 0


def test_safe_train_test_split_supports_unshuffled_integer_test_size():
    train_x, val_x, train_y, val_y = safe_train_test_split(
        ["a", "b", "c", "d"],
        [1, 2, 3, 4],
        test_size=2,
        shuffle=False,
    )

    assert train_x == ["a", "b"]
    assert val_x == ["c", "d"]
    assert train_y == [1, 2]
    assert val_y == [3, 4]


def test_safe_train_test_split_rejects_missing_arrays_and_length_mismatch():
    for args in [(), (np.arange(2), np.arange(3))]:
        try:
            safe_train_test_split(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {args}")


def test_safe_train_test_split_preserves_tuple_type():
    train_x, val_x = safe_train_test_split(("a", "b", "c"), test_size=1, shuffle=False)

    assert train_x == ("a", "b")
    assert val_x == ("c",)
