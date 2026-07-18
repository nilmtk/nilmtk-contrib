"""Safe, checksummed persistence for private LBM training artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch

from nilmtk_contrib.torch._lbm import (
    GaussianRatioSummary,
    StructuredAppliance,
    _finite_real,
    _positive_int,
    _prepare_appliances,
)
from nilmtk_contrib.torch._lbm_training import (
    LBMTrainingResult,
    SourceWindow,
    _markov_reward_moments,
    _windows_overlap,
    assert_disjoint_sources,
)
from nilmtk_contrib.utils.checkpoints import save_json_atomic


ARTIFACT_FILENAME = "lbm-training.json"
ARTIFACT_TYPE = "nilmtk-contrib-lbm-training"
ARTIFACT_SCHEMA_VERSION = 1
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_TRAINING_ALGORITHM = "deterministic-1d-kmeans+hmm-counts"
_SUMMARY_NAMES = ("energy_wh", "duration_minutes")


def _canonical_json(payload: dict) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(
            "LBM artifact payload must be finite and JSON-serializable."
        ) from exc
    return encoded.encode("utf-8")


def _payload_digest(payload: dict) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest()


def _exact_keys(name: str, value, expected: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    actual = set(value)
    if actual != expected:
        missing = ", ".join(sorted(expected - actual)) or "none"
        unknown = ", ".join(sorted(actual - expected)) or "none"
        raise ValueError(
            f"{name} fields do not match schema; missing: {missing}; unknown: {unknown}."
        )
    return value


def _nonnegative_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _count_vector(name: str, value, length: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} counts.")
    return tuple(
        _nonnegative_int(f"{name}[{index}]", item) for index, item in enumerate(value)
    )


def _count_matrix(name: str, value, size: int) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must contain exactly {size} rows.")
    return tuple(
        _count_vector(f"{name}[{index}]", row, size) for index, row in enumerate(value)
    )


def _sources(name: str, value) -> tuple[SourceWindow, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array.")
    sources = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{name}[{index}] must be a JSON object.")
        try:
            sources.append(SourceWindow(**item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {name}[{index}]: {exc}") from exc
    return tuple(sources)


def _probabilities_from_counts(
    counts: torch.Tensor, pseudocount: float
) -> torch.Tensor:
    smoothed = counts.to(torch.float64) + pseudocount
    return smoothed / smoothed.sum(dim=-1, keepdim=True)


def _require_close(name: str, actual, expected) -> None:
    actual_tensor = torch.as_tensor(actual, dtype=torch.float64)
    expected_tensor = torch.as_tensor(expected, dtype=torch.float64)
    if actual_tensor.shape != expected_tensor.shape or not torch.allclose(
        actual_tensor,
        expected_tensor,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError(f"{name} do not match the persisted raw counts.")


_TRAINING_FIELDS = {
    "schema_version",
    "algorithm",
    "sources",
    "evaluation_sources",
    "num_samples",
    "window_length",
    "sample_period_seconds",
    "state_means",
    "initial_probabilities",
    "transition_probabilities",
    "off_state",
    "cycle_probabilities",
    "state_counts",
    "initial_counts",
    "transition_counts",
    "cycle_counts",
    "emission_variance",
    "pseudocount",
    "minimum_windows_per_cycle",
    "variance_floor",
    "kmeans_max_iterations",
    "allow_temporal_evaluation",
    "summaries",
}

_SUMMARY_FIELDS = {
    "name",
    "state_weights",
    "conditional_means",
    "conditional_variance",
    "induced_mean",
    "induced_variance",
    "weight",
}


def _training_result_from_metadata(metadata, *, label: str) -> LBMTrainingResult:
    metadata = _exact_keys(label, metadata, _TRAINING_FIELDS)
    if metadata["schema_version"] != 1:
        raise ValueError(f"{label}.schema_version must be 1.")
    if metadata["algorithm"] != _TRAINING_ALGORITHM:
        raise ValueError(f"{label}.algorithm is unsupported.")
    sources = _sources(f"{label}.sources", metadata["sources"])
    if not sources:
        raise ValueError(f"{label}.sources must be non-empty.")
    evaluation_sources = _sources(
        f"{label}.evaluation_sources", metadata["evaluation_sources"]
    )
    num_samples = _positive_int(f"{label}.num_samples", metadata["num_samples"])
    window_length = _positive_int(f"{label}.window_length", metadata["window_length"])
    sample_period = _positive_int(
        f"{label}.sample_period_seconds", metadata["sample_period_seconds"]
    )
    pseudocount = _finite_real(
        f"{label}.pseudocount", metadata["pseudocount"], positive=True
    )
    minimum_windows_per_cycle = _positive_int(
        f"{label}.minimum_windows_per_cycle",
        metadata["minimum_windows_per_cycle"],
    )
    variance_floor = _finite_real(
        f"{label}.variance_floor", metadata["variance_floor"], positive=True
    )
    kmeans_max_iterations = _positive_int(
        f"{label}.kmeans_max_iterations", metadata["kmeans_max_iterations"]
    )
    allow_temporal = metadata["allow_temporal_evaluation"]
    if not isinstance(allow_temporal, bool):
        raise ValueError(f"{label}.allow_temporal_evaluation must be a boolean.")
    emission_variance = _finite_real(
        f"{label}.emission_variance", metadata["emission_variance"], positive=True
    )

    try:
        state_means = torch.as_tensor(metadata["state_means"], dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{label}.state_means must be numeric.") from exc
    if state_means.ndim != 1 or state_means.numel() < 2:
        raise ValueError(f"{label}.state_means must contain at least two states.")
    if not bool(torch.isfinite(state_means).all()):
        raise ValueError(f"{label}.state_means must be finite.")
    state_count = int(state_means.numel())
    off_state = _nonnegative_int(f"{label}.off_state", metadata["off_state"])
    if off_state >= state_count or off_state != int(torch.argmin(state_means)):
        raise ValueError(f"{label}.off_state must identify the lowest-mean state.")

    cycle_values = metadata["cycle_probabilities"]
    if not isinstance(cycle_values, list) or not cycle_values:
        raise ValueError(f"{label}.cycle_probabilities must be a non-empty array.")
    cycle_categories = len(cycle_values)
    state_counts = _count_vector(
        f"{label}.state_counts", metadata["state_counts"], state_count
    )
    initial_counts = _count_vector(
        f"{label}.initial_counts", metadata["initial_counts"], state_count
    )
    transition_counts = _count_matrix(
        f"{label}.transition_counts", metadata["transition_counts"], state_count
    )
    cycle_counts = _count_vector(
        f"{label}.cycle_counts", metadata["cycle_counts"], cycle_categories
    )
    window_count = len(sources)
    if num_samples != window_count * window_length:
        raise ValueError(f"{label}.num_samples does not match source/window counts.")
    if sum(state_counts) != num_samples:
        raise ValueError(f"{label}.state_counts do not sum to num_samples.")
    if sum(initial_counts) != window_count:
        raise ValueError(f"{label}.initial_counts do not sum to the window count.")
    expected_transitions = window_count * max(0, window_length - 1)
    if sum(sum(row) for row in transition_counts) != expected_transitions:
        raise ValueError(f"{label}.transition_counts do not match window boundaries.")
    if sum(cycle_counts) != window_count:
        raise ValueError(f"{label}.cycle_counts do not sum to the window count.")
    if any(count < minimum_windows_per_cycle for count in cycle_counts):
        raise ValueError(f"{label}.cycle_counts violate minimum_windows_per_cycle.")

    expected_seconds = window_length * sample_period
    for index, source in enumerate(sources):
        if source.sample_period_seconds != sample_period:
            raise ValueError(f"{label}.sources[{index}] has a different sample period.")
        start, end = source.interval
        if not math.isclose(end - start, expected_seconds, abs_tol=1e-6, rel_tol=0):
            raise ValueError(f"{label}.sources[{index}] duration is inconsistent.")
        for previous in sources[:index]:
            if _windows_overlap(previous, source):
                raise ValueError(f"{label}.sources contain overlapping evidence.")
    assert_disjoint_sources(
        sources,
        evaluation_sources,
        allow_temporal_split=allow_temporal,
    )

    summaries_value = metadata["summaries"]
    if not isinstance(summaries_value, list) or len(summaries_value) != 2:
        raise ValueError(f"{label}.summaries must contain energy and duration.")
    summaries = []
    for index, expected_name in enumerate(_SUMMARY_NAMES):
        item = _exact_keys(
            f"{label}.summaries[{index}]",
            summaries_value[index],
            _SUMMARY_FIELDS,
        )
        if item["name"] != expected_name:
            raise ValueError(
                f"{label}.summaries[{index}].name must be {expected_name!r}."
            )
        summaries.append(
            GaussianRatioSummary(
                state_weights=tuple(float(value) for value in item["state_weights"]),
                conditional_means=tuple(
                    float(value) for value in item["conditional_means"]
                ),
                conditional_variance=item["conditional_variance"],
                induced_mean=item["induced_mean"],
                induced_variance=item["induced_variance"],
                weight=item["weight"],
            )
        )
    appliance = StructuredAppliance(
        state_means=tuple(float(value) for value in state_means),
        initial_probabilities=tuple(
            float(value) for value in metadata["initial_probabilities"]
        ),
        transition_probabilities=tuple(
            tuple(float(value) for value in row)
            for row in metadata["transition_probabilities"]
        ),
        off_state=off_state,
        cycle_probabilities=tuple(float(value) for value in cycle_values),
        summaries=tuple(summaries),
    )
    _prepare_appliances(
        [appliance],
        time_points=window_length,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    expected_initial = _probabilities_from_counts(
        torch.tensor(initial_counts), pseudocount
    )
    expected_transition = _probabilities_from_counts(
        torch.tensor(transition_counts), pseudocount
    )
    expected_cycles = _probabilities_from_counts(
        torch.tensor(cycle_counts), pseudocount
    )
    _require_close(
        f"{label}.initial_probabilities",
        metadata["initial_probabilities"],
        expected_initial,
    )
    _require_close(
        f"{label}.transition_probabilities",
        metadata["transition_probabilities"],
        expected_transition,
    )
    _require_close(
        f"{label}.cycle_probabilities",
        cycle_values,
        expected_cycles,
    )

    initial_tensor = torch.as_tensor(
        metadata["initial_probabilities"], dtype=torch.float64
    )
    transition_tensor = torch.as_tensor(
        metadata["transition_probabilities"], dtype=torch.float64
    )
    for index, summary in enumerate(summaries):
        induced_mean, induced_variance = _markov_reward_moments(
            initial_tensor,
            transition_tensor,
            torch.as_tensor(summary.state_weights, dtype=torch.float64),
            window_length,
        )
        if not math.isclose(
            induced_mean, summary.induced_mean, rel_tol=1e-10, abs_tol=1e-10
        ) or not math.isclose(
            induced_variance,
            summary.induced_variance,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError(
                f"{label}.summaries[{index}] induced moments do not match the HMM."
            )

    result = LBMTrainingResult(
        appliance=appliance,
        sources=sources,
        evaluation_sources=evaluation_sources,
        num_samples=num_samples,
        window_length=window_length,
        sample_period_seconds=sample_period,
        state_counts=state_counts,
        initial_counts=initial_counts,
        transition_counts=transition_counts,
        cycle_counts=cycle_counts,
        emission_variance=emission_variance,
        pseudocount=pseudocount,
        minimum_windows_per_cycle=minimum_windows_per_cycle,
        variance_floor=variance_floor,
        kmeans_max_iterations=kmeans_max_iterations,
        allow_temporal_evaluation=allow_temporal,
    )
    if result.metadata() != metadata:
        raise ValueError(f"{label} is not in canonical artifact form.")
    return result


@dataclass(frozen=True)
class LBMTrainingBundle:
    """Immutable, schema-validated collection of fitted LBM appliances."""

    appliances: Mapping[str, LBMTrainingResult]

    def __post_init__(self):
        if not isinstance(self.appliances, Mapping) or not self.appliances:
            raise ValueError("appliances must be a non-empty mapping.")
        validated = {}
        for raw_name, raw_result in self.appliances.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError("appliance names must be non-empty strings.")
            name = raw_name.strip()
            if name in validated:
                raise ValueError(f"Duplicate normalized appliance name {name!r}.")
            if not isinstance(raw_result, LBMTrainingResult):
                raise TypeError(f"appliances[{name!r}] must be an LBMTrainingResult.")
            validated[name] = _training_result_from_metadata(
                raw_result.metadata(),
                label=f"appliances[{name!r}]",
            )
        results = tuple(validated.values())
        if len({result.window_length for result in results}) != 1:
            raise ValueError("All LBM appliances must use the same window length.")
        if len({result.sample_period_seconds for result in results}) != 1:
            raise ValueError("All LBM appliances must use the same sample period.")
        if len({result.evaluation_sources for result in results}) != 1:
            raise ValueError(
                "All LBM appliances must declare the same evaluation sources."
            )
        if len({result.allow_temporal_evaluation for result in results}) != 1:
            raise ValueError("All LBM appliances must use the same leakage policy.")
        object.__setattr__(
            self,
            "appliances",
            MappingProxyType(dict(sorted(validated.items()))),
        )

    def metadata(self) -> dict:
        return {
            "artifact_type": ARTIFACT_TYPE,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "appliances": {
                name: result.metadata() for name, result in self.appliances.items()
            },
        }


def _artifact_envelope(bundle: LBMTrainingBundle) -> dict:
    payload = bundle.metadata()
    return {**payload, "payload_sha256": _payload_digest(payload)}


def save_lbm_training_bundle(path, bundle: LBMTrainingBundle) -> Path:
    """Atomically save a deterministic, non-executable LBM JSON artifact."""
    if not isinstance(bundle, LBMTrainingBundle):
        raise TypeError("bundle must be an LBMTrainingBundle.")
    # Revalidate at the persistence boundary rather than trusting a long-lived
    # in-memory object graph.
    bundle = LBMTrainingBundle(bundle.appliances)
    target = Path(path) / ARTIFACT_FILENAME
    save_json_atomic(target, _artifact_envelope(bundle))
    return target


def load_lbm_training_bundle(path) -> LBMTrainingBundle:
    """Load and semantically validate a checksummed LBM JSON artifact."""
    target = Path(path) / ARTIFACT_FILENAME
    if target.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError(
            f"LBM artifact exceeds the {MAX_ARTIFACT_BYTES}-byte safety limit."
        )
    try:
        with target.open(encoding="utf-8") as handle:
            envelope = json.load(handle)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("LBM artifact is not valid UTF-8 JSON.") from exc
    envelope = _exact_keys(
        "artifact",
        envelope,
        {"artifact_type", "schema_version", "appliances", "payload_sha256"},
    )
    if envelope["artifact_type"] != ARTIFACT_TYPE:
        raise ValueError("Unsupported LBM artifact_type.")
    if envelope["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported LBM artifact schema_version.")
    supplied_digest = envelope["payload_sha256"]
    if not isinstance(supplied_digest, str):
        raise ValueError("LBM artifact payload_sha256 must be a string.")
    payload = {key: value for key, value in envelope.items() if key != "payload_sha256"}
    expected_digest = _payload_digest(payload)
    if not hmac.compare_digest(supplied_digest, expected_digest):
        raise ValueError("LBM artifact checksum mismatch; the file is corrupt.")
    appliances_value = payload["appliances"]
    if not isinstance(appliances_value, dict) or not appliances_value:
        raise ValueError("LBM artifact appliances must be a non-empty object.")
    try:
        results = {
            name: _training_result_from_metadata(
                metadata,
                label=f"appliances[{name!r}]",
            )
            for name, metadata in appliances_value.items()
        }
    except (TypeError, RuntimeError, OverflowError) as exc:
        raise ValueError(f"LBM artifact contains invalid model data: {exc}") from exc
    bundle = LBMTrainingBundle(results)
    if bundle.metadata() != payload:
        raise ValueError("LBM artifact is not in canonical form.")
    return bundle


__all__ = [
    "ARTIFACT_FILENAME",
    "ARTIFACT_SCHEMA_VERSION",
    "ARTIFACT_TYPE",
    "LBMTrainingBundle",
    "load_lbm_training_bundle",
    "save_lbm_training_bundle",
]
