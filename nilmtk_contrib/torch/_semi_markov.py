"""Exact, differentiable semi-Markov dynamic programming primitives.

The score of a segmentation is the sum of per-sample emission scores, one
duration score per segment, an initial-state score, and transition scores
between adjacent segments. Transition diagonals are forbidden so each state
sequence has one unambiguous maximal-segment representation.

This module is private. The classical explicit-duration HSMM and its neural
semi-Markov counterpart share it so their decoding semantics cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


_FLOAT_DTYPES = frozenset({torch.float32, torch.float64})
_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
)


@dataclass(frozen=True)
class SemiMarkovSegment:
    """One half-open constant-state segment ``[start, end)``."""

    state: int
    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class SemiMarkovViterbiResult:
    """Highest-scoring state path and its maximal constant-state segments."""

    states: torch.Tensor
    segments: tuple[SemiMarkovSegment, ...]
    score: float


def _score_tensor(name, value, *, shape=None, reference=None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.dtype not in _FLOAT_DTYPES or value.is_complex():
        raise TypeError(f"{name} must use torch.float32 or torch.float64.")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}.")
    if reference is not None:
        if value.dtype != reference.dtype:
            raise ValueError(f"{name} must use dtype {reference.dtype}.")
        if value.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}.")
    if bool(torch.isnan(value).any()) or bool(torch.isposinf(value).any()):
        raise ValueError(f"{name} must not contain NaN or positive infinity.")
    return value


def _validate_scores(
    emission_scores,
    initial_scores,
    transition_scores,
    duration_scores,
) -> tuple[int, int, int]:
    emissions = _score_tensor("emission_scores", emission_scores)
    if emissions.ndim != 2 or emissions.shape[0] < 1 or emissions.shape[1] < 2:
        raise ValueError(
            "emission_scores must have shape (time, states) with time >= 1 "
            "and states >= 2."
        )
    if not bool(torch.isfinite(emissions).all()):
        raise ValueError("emission_scores must contain only finite values.")
    time_points, states = emissions.shape
    initial = _score_tensor(
        "initial_scores",
        initial_scores,
        shape=(states,),
        reference=emissions,
    )
    transitions = _score_tensor(
        "transition_scores",
        transition_scores,
        shape=(states, states),
        reference=emissions,
    )
    durations = _score_tensor(
        "duration_scores",
        duration_scores,
        reference=emissions,
    )
    if durations.ndim != 2 or durations.shape[0] != states or durations.shape[1] < 1:
        raise ValueError(f"duration_scores must have shape ({states}, max_duration).")
    if not bool(torch.isfinite(initial).any()):
        raise ValueError("initial_scores must allow at least one state.")
    if not bool(torch.isneginf(torch.diagonal(transitions)).all()):
        raise ValueError(
            "transition_scores diagonal must be negative infinity; "
            "durations, not self-transitions, represent state persistence."
        )
    if not bool(torch.isfinite(durations).any(dim=1).all()):
        raise ValueError("Every state must allow at least one duration.")
    return int(time_points), int(states), int(durations.shape[1])


def _emission_prefix(emission_scores: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            emission_scores.new_zeros((1, emission_scores.shape[1])),
            torch.cumsum(emission_scores, dim=0),
        ),
        dim=0,
    )


def _safe_logsumexp(values: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Keep gradients finite for slices whose every path is forbidden."""
    impossible = torch.isneginf(values).all(dim=dim)
    safe_values = torch.where(
        impossible.unsqueeze(dim),
        torch.zeros_like(values),
        values,
    )
    reduced = torch.logsumexp(safe_values, dim=dim)
    return torch.where(
        impossible,
        torch.full_like(reduced, -torch.inf),
        reduced,
    )


def semi_markov_log_partition(
    emission_scores,
    initial_scores,
    transition_scores,
    duration_scores,
) -> torch.Tensor:
    """Return the exact log-partition over all valid segmentations.

    The result remains differentiable with respect to every finite score.
    Runtime is ``O(T * K * D * K)`` in arithmetic but only ``O(T * K)`` in
    stored dynamic-programming state, where ``D`` is the duration cap.
    """

    time_points, states, max_duration = _validate_scores(
        emission_scores,
        initial_scores,
        transition_scores,
        duration_scores,
    )
    prefix = _emission_prefix(emission_scores)
    rows: list[torch.Tensor | None] = [None]
    for end in range(1, time_points + 1):
        length = min(max_duration, end)
        starts = end - torch.arange(
            1,
            length + 1,
            device=emission_scores.device,
        )
        earliest_positive_start = max(1, end - length)
        if earliest_positive_start < end:
            predecessors = torch.stack(rows[earliest_positive_start:end]).flip(0)
        else:
            predecessors = None

        end_scores = []
        for state in range(states):
            segment = (
                prefix[end, state]
                - prefix[starts, state]
                + duration_scores[state, :length]
            )
            candidates = []
            if predecessors is not None:
                transitioned = _safe_logsumexp(
                    predecessors + transition_scores[:, state],
                    dim=1,
                )
                candidates.append(transitioned + segment[: transitioned.numel()])
            if end <= max_duration:
                candidates.append((initial_scores[state] + segment[-1]).reshape(1))
            end_scores.append(_safe_logsumexp(torch.cat(candidates), dim=0))
        rows.append(torch.stack(end_scores))

    result = _safe_logsumexp(rows[-1], dim=0)
    if not bool(torch.isfinite(result)):
        raise ValueError("No valid semi-Markov segmentation reaches the sequence end.")
    return result


@torch.no_grad()
def semi_markov_viterbi(
    emission_scores,
    initial_scores,
    transition_scores,
    duration_scores,
) -> SemiMarkovViterbiResult:
    """Decode the exact highest-scoring semi-Markov state sequence."""

    time_points, states, max_duration = _validate_scores(
        emission_scores,
        initial_scores,
        transition_scores,
        duration_scores,
    )
    prefix = _emission_prefix(emission_scores)
    best = emission_scores.new_full((time_points + 1, states), -torch.inf)
    previous_state = torch.full(
        (time_points + 1, states),
        -1,
        dtype=torch.int64,
        device=emission_scores.device,
    )
    best_duration = torch.zeros_like(previous_state)

    for end in range(1, time_points + 1):
        length = min(max_duration, end)
        starts = end - torch.arange(
            1,
            length + 1,
            device=emission_scores.device,
        )
        earliest_positive_start = max(1, end - length)
        predecessor_rows = best[earliest_positive_start:end].flip(0)
        for state in range(states):
            segment = (
                prefix[end, state]
                - prefix[starts, state]
                + duration_scores[state, :length]
            )
            candidate_scores = []
            candidate_states = []
            if predecessor_rows.shape[0]:
                transitioned, sources = torch.max(
                    predecessor_rows + transition_scores[:, state],
                    dim=1,
                )
                candidate_scores.append(transitioned + segment[: transitioned.numel()])
                candidate_states.append(sources)
            if end <= max_duration:
                candidate_scores.append(
                    (initial_scores[state] + segment[-1]).reshape(1)
                )
                candidate_states.append(
                    torch.full(
                        (1,),
                        -1,
                        dtype=torch.int64,
                        device=emission_scores.device,
                    )
                )
            scores = torch.cat(candidate_scores)
            sources = torch.cat(candidate_states)
            candidate = int(torch.argmax(scores))
            best[end, state] = scores[candidate]
            previous_state[end, state] = sources[candidate]
            best_duration[end, state] = candidate + 1

    final_state = int(torch.argmax(best[time_points]))
    final_score = best[time_points, final_state]
    if not bool(torch.isfinite(final_score)):
        raise ValueError("No valid semi-Markov segmentation reaches the sequence end.")

    decoded = torch.empty(
        time_points,
        dtype=torch.int64,
        device=emission_scores.device,
    )
    segments = []
    end = time_points
    state = final_state
    while end:
        duration = int(best_duration[end, state])
        if duration < 1 or duration > end:
            raise RuntimeError(
                "Semi-Markov backtracking encountered an invalid duration."
            )
        start = end - duration
        decoded[start:end] = state
        segments.append(SemiMarkovSegment(state=state, start=start, end=end))
        state = int(previous_state[end, state])
        end = start
        if end and state < 0:
            raise RuntimeError("Semi-Markov backtracking lost its predecessor state.")
    segments.reverse()
    return SemiMarkovViterbiResult(
        states=decoded,
        segments=tuple(segments),
        score=float(final_score),
    )


def semi_markov_path_score(
    states,
    emission_scores,
    initial_scores,
    transition_scores,
    duration_scores,
) -> torch.Tensor:
    """Score one labeled path under the same maximal-segment convention."""

    time_points, state_count, max_duration = _validate_scores(
        emission_scores,
        initial_scores,
        transition_scores,
        duration_scores,
    )
    if not isinstance(states, torch.Tensor):
        raise TypeError("states must be a torch.Tensor.")
    if states.dtype not in _INTEGER_DTYPES or states.ndim != 1:
        raise TypeError("states must be a one-dimensional integer tensor.")
    if states.device != emission_scores.device:
        raise ValueError(f"states must be on {emission_scores.device}.")
    if states.numel() != time_points:
        raise ValueError(f"states must contain exactly {time_points} labels.")
    if bool(((states < 0) | (states >= state_count)).any()):
        raise ValueError(f"states labels must be between 0 and {state_count - 1}.")

    labels = states.detach().cpu().tolist()
    segments = []
    start = 0
    for end in range(1, time_points + 1):
        if end == time_points or labels[end] != labels[start]:
            segments.append(SemiMarkovSegment(labels[start], start, end))
            start = end
    if any(segment.duration > max_duration for segment in segments):
        raise ValueError("states contain a segment longer than max_duration.")

    first = segments[0]
    score = initial_scores[first.state]
    previous = None
    for segment in segments:
        if previous is not None:
            score = score + transition_scores[previous.state, segment.state]
        score = (
            score
            + emission_scores[segment.start : segment.end, segment.state].sum()
            + duration_scores[segment.state, segment.duration - 1]
        )
        previous = segment
    if not bool(torch.isfinite(score)):
        raise ValueError("states describe a path forbidden by the model scores.")
    return score


def semi_markov_nll(
    states,
    emission_scores,
    initial_scores,
    transition_scores,
    duration_scores,
) -> torch.Tensor:
    """Return exact conditional negative log likelihood for one labeled path."""

    log_partition = semi_markov_log_partition(
        emission_scores,
        initial_scores,
        transition_scores,
        duration_scores,
    )
    gold_score = semi_markov_path_score(
        states,
        emission_scores,
        initial_scores,
        transition_scores,
        duration_scores,
    )
    return log_partition - gold_score


__all__ = [
    "SemiMarkovSegment",
    "SemiMarkovViterbiResult",
    "semi_markov_log_partition",
    "semi_markov_nll",
    "semi_markov_path_score",
    "semi_markov_viterbi",
]
