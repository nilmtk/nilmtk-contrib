"""Safe train/validation splitting helpers."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainingDecision:
    should_train: bool
    reason: str
    num_samples: int
    min_samples: int


@dataclass(frozen=True)
class SplitMetadata:
    should_train: bool
    reason: str
    num_samples: int
    train_size: int
    validation_size: int
    validation_enabled: bool
    validation_fraction: float
    strategy: str
    seed: int | None


@dataclass(frozen=True)
class TrainValidationSplit:
    X_train: object
    y_train: object
    X_val: object | None
    y_val: object | None
    metadata: SplitMetadata


def should_train(num_samples, min_samples):
    """Return a structured training decision for a sample count."""
    if num_samples < min_samples:
        return TrainingDecision(
            should_train=False,
            reason=f"num_samples={num_samples} is below min_samples={min_samples}.",
            num_samples=num_samples,
            min_samples=min_samples,
        )

    return TrainingDecision(
        should_train=True,
        reason="enough samples to train.",
        num_samples=num_samples,
        min_samples=min_samples,
    )


def _length(values):
    try:
        return len(values)
    except TypeError as exc:
        raise ValueError("X and y must be sized collections.") from exc


def _take(values, indices):
    if values is None:
        return None
    if hasattr(values, "iloc"):
        return values.iloc[indices]
    if isinstance(values, (list, tuple)):
        return type(values)(values[int(index)] for index in indices)
    return values[indices]


def _empty_like(values):
    if values is None:
        return None
    return _take(values, np.asarray([], dtype=int))


def train_validation_split(
    X,
    y,
    validation_fraction=0.15,
    strategy="tail",
    seed=None,
    min_train=1,
    min_val=1,
    allow_no_validation=False,
):
    """Split arrays safely, avoiding empty train or validation sets."""
    if strategy not in {"tail", "random"}:
        raise ValueError("strategy must be one of 'tail' or 'random'.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if min_train < 1:
        raise ValueError("min_train must be at least 1.")
    if min_val < 1:
        raise ValueError("min_val must be at least 1.")

    num_samples = _length(X)
    if _length(y) != num_samples:
        raise ValueError("X and y must contain the same number of samples.")

    min_samples_with_validation = min_train + min_val
    if num_samples < min_samples_with_validation:
        if not allow_no_validation:
            metadata = SplitMetadata(
                should_train=False,
                reason=(
                    f"num_samples={num_samples} is below the required "
                    f"min_train + min_val={min_samples_with_validation}."
                ),
                num_samples=num_samples,
                train_size=0,
                validation_size=0,
                validation_enabled=False,
                validation_fraction=validation_fraction,
                strategy=strategy,
                seed=seed,
            )
            return TrainValidationSplit(None, None, None, None, metadata)

        decision = should_train(num_samples, min_train)
        metadata = SplitMetadata(
            should_train=decision.should_train,
            reason=(
                "training without validation because there are not enough "
                "samples for a validation split."
                if decision.should_train
                else decision.reason
            ),
            num_samples=num_samples,
            train_size=num_samples if decision.should_train else 0,
            validation_size=0,
            validation_enabled=False,
            validation_fraction=validation_fraction,
            strategy=strategy,
            seed=seed,
        )
        if not decision.should_train:
            return TrainValidationSplit(None, None, None, None, metadata)
        indices = np.arange(num_samples)
        return TrainValidationSplit(
            _take(X, indices),
            _take(y, indices),
            _empty_like(X),
            _empty_like(y),
            metadata,
        )

    validation_size = max(min_val, int(round(num_samples * validation_fraction)))
    validation_size = min(validation_size, num_samples - min_train)
    train_size = num_samples - validation_size

    if strategy == "tail":
        train_indices = np.arange(train_size)
        validation_indices = np.arange(train_size, num_samples)
    else:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(num_samples)
        validation_indices = np.sort(indices[:validation_size])
        train_indices = np.sort(indices[validation_size:])

    metadata = SplitMetadata(
        should_train=True,
        reason="using train/validation split.",
        num_samples=num_samples,
        train_size=len(train_indices),
        validation_size=len(validation_indices),
        validation_enabled=True,
        validation_fraction=validation_fraction,
        strategy=strategy,
        seed=seed,
    )
    return TrainValidationSplit(
        _take(X, train_indices),
        _take(y, train_indices),
        _take(X, validation_indices),
        _take(y, validation_indices),
        metadata,
    )


def safe_train_test_split(*arrays, test_size=0.15, random_state=None, shuffle=True, **_):
    """Small sklearn-compatible split wrapper with non-empty validation when possible."""
    if not arrays:
        raise ValueError("At least one array is required.")
    num_samples = _length(arrays[0])
    for array in arrays[1:]:
        if _length(array) != num_samples:
            raise ValueError("All arrays must contain the same number of samples.")

    if num_samples < 2:
        train_indices = np.arange(num_samples)
        validation_indices = np.asarray([], dtype=int)
    else:
        if isinstance(test_size, float):
            validation_size = max(1, int(round(num_samples * test_size)))
        else:
            validation_size = int(test_size)
        validation_size = min(validation_size, num_samples - 1)

        if shuffle:
            rng = np.random.default_rng(random_state)
            indices = rng.permutation(num_samples)
            validation_indices = np.sort(indices[:validation_size])
            train_indices = np.sort(indices[validation_size:])
        else:
            train_size = num_samples - validation_size
            train_indices = np.arange(train_size)
            validation_indices = np.arange(train_size, num_samples)

    split_arrays = []
    for array in arrays:
        split_arrays.append(_take(array, train_indices))
        split_arrays.append(_take(array, validation_indices))
    return tuple(split_arrays)
