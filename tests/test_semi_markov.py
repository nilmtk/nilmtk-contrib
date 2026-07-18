from itertools import product
import inspect
import math
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from nilmtk_contrib.torch._semi_markov import (  # noqa: E402
    SemiMarkovSegment,
    semi_markov_log_partition,
    semi_markov_nll,
    semi_markov_path_score,
    semi_markov_viterbi,
)


def _scores(*, dtype=torch.float64, device="cpu"):
    return (
        torch.tensor(
            [
                [0.8, -0.2],
                [0.5, 0.1],
                [-0.4, 0.9],
                [0.2, 0.6],
            ],
            dtype=dtype,
            device=device,
        ),
        torch.tensor([0.2, -0.1], dtype=dtype, device=device),
        torch.tensor(
            [[-torch.inf, 0.3], [-0.2, -torch.inf]],
            dtype=dtype,
            device=device,
        ),
        torch.tensor(
            [[0.1, 0.4, -0.3], [0.2, -0.1, 0.3]],
            dtype=dtype,
            device=device,
        ),
    )


def _enumerated_paths(scores):
    emission, initial, transition, duration = scores
    valid = []
    for labels in product(range(emission.shape[1]), repeat=emission.shape[0]):
        states = torch.tensor(labels, dtype=torch.int64, device=emission.device)
        try:
            score = semi_markov_path_score(
                states,
                emission,
                initial,
                transition,
                duration,
            )
        except ValueError:
            continue
        valid.append((labels, score))
    return valid


def test_forward_and_viterbi_match_exhaustive_enumeration():
    scores = _scores()
    paths = _enumerated_paths(scores)
    expected_partition = torch.logsumexp(
        torch.stack([score for _, score in paths]), dim=0
    )
    expected_labels, expected_score = max(paths, key=lambda item: float(item[1]))

    actual_partition = semi_markov_log_partition(*scores)
    result = semi_markov_viterbi(*scores)

    assert actual_partition == pytest.approx(float(expected_partition), abs=1e-12)
    assert result.score == pytest.approx(float(expected_score), abs=1e-12)
    assert tuple(result.states.tolist()) == expected_labels
    assert result.segments == (
        SemiMarkovSegment(state=0, start=0, end=2),
        SemiMarkovSegment(state=1, start=2, end=4),
    )


@pytest.mark.parametrize(
    ("time_points", "states", "max_duration", "seed"),
    [
        (1, 2, 1, 1),
        (2, 2, 1, 2),
        (3, 2, 2, 3),
        (4, 2, 4, 4),
        (3, 3, 2, 5),
        (4, 3, 3, 6),
        (5, 3, 2, 7),
    ],
)
def test_random_small_problems_match_exhaustive_oracle(
    time_points, states, max_duration, seed
):
    generator = torch.Generator().manual_seed(seed)
    emission = torch.randn(
        (time_points, states), dtype=torch.float64, generator=generator
    )
    initial = torch.randn(states, dtype=torch.float64, generator=generator)
    transition = torch.randn((states, states), dtype=torch.float64, generator=generator)
    transition.fill_diagonal_(-torch.inf)
    duration = torch.randn(
        (states, max_duration), dtype=torch.float64, generator=generator
    )
    scores = (emission, initial, transition, duration)
    paths = _enumerated_paths(scores)
    expected_partition = torch.logsumexp(
        torch.stack([score for _, score in paths]), dim=0
    )
    expected_score = max(float(score) for _, score in paths)

    partition = semi_markov_log_partition(*scores)
    result = semi_markov_viterbi(*scores)

    assert partition == pytest.approx(float(expected_partition), abs=1e-12)
    assert result.score == pytest.approx(expected_score, abs=1e-12)
    assert float(partition) >= result.score


def test_path_score_uses_maximal_segments_and_duration_scores():
    emission, initial, transition, duration = _scores()
    states = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    actual = semi_markov_path_score(
        states,
        emission,
        initial,
        transition,
        duration,
    )
    expected = (
        initial[0]
        + emission[:2, 0].sum()
        + duration[0, 1]
        + transition[0, 1]
        + emission[2:, 1].sum()
        + duration[1, 1]
    )

    assert actual == expected


def test_duration_scores_change_the_best_path_without_postprocessing():
    emission = torch.zeros((4, 2), dtype=torch.float64)
    initial = torch.tensor([0.0, -10.0], dtype=torch.float64)
    transition = torch.tensor(
        [[-torch.inf, 0.0], [0.0, -torch.inf]], dtype=torch.float64
    )
    duration = torch.tensor(
        [[-10.0, 4.0, -10.0, -10.0], [-10.0, 4.0, -10.0, -10.0]],
        dtype=torch.float64,
    )

    result = semi_markov_viterbi(emission, initial, transition, duration)

    assert result.states.tolist() == [0, 0, 1, 1]
    assert [segment.duration for segment in result.segments] == [2, 2]


def test_single_segment_sequence_and_duration_cap_are_handled_exactly():
    emission, initial, transition, duration = _scores()
    one_sample = semi_markov_viterbi(
        emission[:1],
        initial,
        transition,
        duration,
    )

    assert one_sample.states.tolist() == [0]
    assert one_sample.segments == (SemiMarkovSegment(0, 0, 1),)
    with pytest.raises(ValueError, match="longer than max_duration"):
        semi_markov_path_score(
            torch.zeros(4, dtype=torch.int64),
            emission,
            initial,
            transition,
            duration,
        )


def test_log_partition_and_nll_are_differentiable_and_nonnegative():
    emission, initial, transition, duration = _scores()
    emission.requires_grad_()
    initial.requires_grad_()
    duration.requires_grad_()
    states = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    loss = semi_markov_nll(
        states,
        emission,
        initial,
        transition,
        duration,
    )
    loss.backward()

    assert float(loss) >= 0
    for value in (emission, initial, duration):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_temporarily_unreachable_states_do_not_create_nan_gradients():
    emission = torch.zeros((3, 2), dtype=torch.float64, requires_grad=True)
    initial = torch.tensor([0.0, -torch.inf], dtype=torch.float64)
    transition = torch.tensor(
        [[-torch.inf, 0.0], [0.0, -torch.inf]], dtype=torch.float64
    )
    duration = torch.tensor(
        [[0.0, -torch.inf], [-torch.inf, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    partition = semi_markov_log_partition(
        emission,
        initial,
        transition,
        duration,
    )
    partition.backward()

    assert torch.isfinite(partition)
    assert emission.grad is not None
    assert duration.grad is not None
    assert torch.isfinite(emission.grad).all()
    assert torch.isfinite(duration.grad).all()


def test_log_partition_passes_float64_autograd_gradcheck():
    emission, initial, _, duration = _scores()
    off_diagonal = torch.tensor([0.3, -0.2], dtype=torch.float64)
    values = tuple(
        value.clone().requires_grad_()
        for value in (emission, initial, off_diagonal, duration)
    )

    def partition(emission_value, initial_value, off_value, duration_value):
        negative_infinity = emission_value.new_tensor(-torch.inf)
        transition_value = torch.stack(
            (
                torch.stack((negative_infinity, off_value[0])),
                torch.stack((off_value[1], negative_infinity)),
            )
        )
        return semi_markov_log_partition(
            emission_value,
            initial_value,
            transition_value,
            duration_value,
        )

    assert torch.autograd.gradcheck(partition, values, atol=1e-6, rtol=1e-5)


def test_impossible_models_fail_instead_of_returning_negative_infinity():
    emission = torch.zeros((2, 2), dtype=torch.float64)
    initial = torch.zeros(2, dtype=torch.float64)
    transition = torch.full((2, 2), -torch.inf, dtype=torch.float64)
    duration = torch.zeros((2, 1), dtype=torch.float64)

    with pytest.raises(ValueError, match="No valid"):
        semi_markov_log_partition(emission, initial, transition, duration)
    with pytest.raises(ValueError, match="No valid"):
        semi_markov_viterbi(emission, initial, transition, duration)


@pytest.mark.parametrize(
    ("field", "replacement", "error", "message"),
    [
        ("emission", [[1.0, 2.0]], TypeError, "torch.Tensor"),
        (
            "emission",
            torch.ones((3, 1), dtype=torch.float64),
            ValueError,
            "states >= 2",
        ),
        (
            "emission",
            torch.ones((3, 2), dtype=torch.int64),
            TypeError,
            "float32 or torch.float64",
        ),
        (
            "emission",
            torch.tensor([[0.0, math.nan]], dtype=torch.float64),
            ValueError,
            "NaN",
        ),
        (
            "initial",
            torch.zeros(3, dtype=torch.float64),
            ValueError,
            "shape",
        ),
        (
            "transition",
            torch.zeros((2, 2), dtype=torch.float64),
            ValueError,
            "diagonal",
        ),
        (
            "duration",
            torch.empty((2, 0), dtype=torch.float64),
            ValueError,
            "max_duration",
        ),
        (
            "duration",
            torch.tensor([[-torch.inf, -torch.inf], [0.0, 0.0]], dtype=torch.float64),
            ValueError,
            "Every state",
        ),
    ],
)
def test_score_contract_rejects_invalid_inputs(field, replacement, error, message):
    emission, initial, transition, duration = _scores()
    values = {
        "emission": emission,
        "initial": initial,
        "transition": transition,
        "duration": duration,
    }
    values[field] = replacement

    with pytest.raises(error, match=message):
        semi_markov_viterbi(
            values["emission"],
            values["initial"],
            values["transition"],
            values["duration"],
        )


@pytest.mark.parametrize(
    ("states", "error", "message"),
    [
        ([0, 0, 1, 1], TypeError, "torch.Tensor"),
        (torch.tensor([0.0, 0.0, 1.0, 1.0]), TypeError, "integer"),
        (torch.tensor([[0, 0, 1, 1]]), TypeError, "one-dimensional"),
        (torch.tensor([0, 1]), ValueError, "exactly 4"),
        (torch.tensor([0, 0, 1, 2]), ValueError, "between 0 and 1"),
    ],
)
def test_path_score_rejects_invalid_state_labels(states, error, message):
    with pytest.raises(error, match=message):
        semi_markov_path_score(states, *_scores())


def test_float32_and_longer_sequences_remain_finite_and_deterministic():
    generator = torch.Generator().manual_seed(42)
    emission = torch.randn((128, 3), generator=generator)
    initial = torch.randn(3, generator=generator)
    transition = torch.randn((3, 3), generator=generator)
    transition.fill_diagonal_(-torch.inf)
    duration = torch.randn((3, 32), generator=generator)

    first = semi_markov_viterbi(emission, initial, transition, duration)
    second = semi_markov_viterbi(emission, initial, transition, duration)

    assert first.score == second.score
    assert torch.equal(first.states, second.states)
    assert first.segments == second.segments
    assert torch.isfinite(
        semi_markov_log_partition(emission, initial, transition, duration)
    )


def test_core_has_no_solver_or_array_runtime_dependency():
    source = Path(inspect.getfile(semi_markov_viterbi)).read_text(encoding="utf-8")

    for dependency in ("cvxpy", "mosek", "numpy", "scipy"):
        assert dependency not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_matches_cpu_with_float64():
    cpu_scores = _scores()
    cuda_scores = tuple(value.cuda() for value in cpu_scores)

    cpu_partition = semi_markov_log_partition(*cpu_scores)
    cuda_partition = semi_markov_log_partition(*cuda_scores).cpu()
    cpu_path = semi_markov_viterbi(*cpu_scores)
    cuda_path = semi_markov_viterbi(*cuda_scores)

    assert cuda_partition == pytest.approx(float(cpu_partition), abs=1e-10)
    assert cuda_path.score == pytest.approx(cpu_path.score, abs=1e-10)
    assert cuda_path.states.cpu().tolist() == cpu_path.states.tolist()
