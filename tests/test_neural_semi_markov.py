import inspect
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from nilmtk_contrib.torch._neural_semi_markov import (  # noqa: E402
    NeuralSemiMarkovNetwork,
    TemporalResidualBlock,
)


def _network(**overrides):
    params = {
        "num_states": 2,
        "max_duration": 6,
        "hidden_channels": 8,
        "num_blocks": 2,
        "kernel_size": 3,
        "dropout": 0.0,
    }
    params.update(overrides)
    return NeuralSemiMarkovNetwork(**params)


def _batch(length=6):
    inputs = torch.tensor(
        [
            [[0.0, 0.1, -0.1, 1.0, 1.1, 0.9]],
            [[1.0, 0.9, 1.1, 0.0, 0.1, -0.1]],
        ],
        dtype=torch.float32,
    )[:, :, :length]
    states = torch.tensor(
        [[0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0]],
        dtype=torch.int64,
    )[:, :length]
    return inputs, states


def test_network_emissions_preserve_batch_time_and_state_axes():
    network = _network(num_states=3)
    inputs = torch.randn(4, 1, 11)

    scores = network.emission_scores(inputs)

    assert scores.shape == (4, 11, 3)
    assert torch.isfinite(scores).all()
    assert sum(parameter.numel() for parameter in network.parameters()) < 5_000


def test_structure_scores_are_normalized_and_forbid_self_transitions():
    network = _network(num_states=3, max_duration=5)
    initial, transitions, durations = network.structure_scores()

    assert float(torch.exp(initial).sum()) == pytest.approx(1.0)
    assert torch.isneginf(torch.diagonal(transitions)).all()
    assert torch.allclose(
        torch.exp(transitions).sum(dim=1), torch.ones(3), atol=1e-7, rtol=0
    )
    assert torch.allclose(
        torch.exp(durations).sum(dim=1), torch.ones(3), atol=1e-7, rtol=0
    )


def test_classical_probabilities_initialize_the_same_structure_scores():
    network = _network(max_duration=3)
    initial = [0.8, 0.2]
    transitions = [[0.0, 1.0], [1.0, 0.0]]
    durations = [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]]

    network.initialize_structure(initial, transitions, durations)
    actual_initial, actual_transitions, actual_durations = network.structure_scores()

    assert torch.exp(actual_initial).tolist() == pytest.approx(initial)
    assert torch.allclose(
        torch.exp(actual_transitions), torch.tensor(transitions), atol=1e-7, rtol=0
    )
    assert torch.allclose(
        torch.exp(actual_durations), torch.tensor(durations), atol=1e-7, rtol=0
    )


def test_exact_nll_is_finite_nonnegative_and_backpropagates_every_score_family():
    torch.manual_seed(4)
    network = _network()
    inputs, states = _batch()

    loss = network.negative_log_likelihood(inputs, states)
    loss.backward()

    assert float(loss) >= 0
    for name, parameter in network.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert torch.all(torch.diagonal(network.transition_logits.grad) == 0)


def test_a_small_optimizer_run_reduces_the_structured_training_loss():
    torch.manual_seed(9)
    network = _network(dropout=0.0)
    inputs, states = _batch()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.02)
    initial_loss = float(network.negative_log_likelihood(inputs, states))

    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss = network.negative_log_likelihood(inputs, states)
        loss.backward()
        optimizer.step()

    final_loss = float(network.negative_log_likelihood(inputs, states))
    assert final_loss < initial_loss


def test_decode_is_deterministic_and_restores_training_mode():
    torch.manual_seed(12)
    network = _network(dropout=0.3)
    inputs, _ = _batch()
    network.train()

    first = network.decode(inputs)
    second = network.decode(inputs)

    assert network.training
    assert len(first) == len(inputs)
    assert all(result.states.shape == (inputs.shape[2],) for result in first)
    assert all(result.segments for result in first)
    assert [result.score for result in first] == [result.score for result in second]
    assert all(
        torch.equal(left.states, right.states) for left, right in zip(first, second)
    )


def test_residual_block_preserves_length_and_propagates_gradients():
    block = TemporalResidualBlock(channels=4, kernel_size=3, dilation=4, dropout=0.0)
    inputs = torch.randn(2, 4, 17, requires_grad=True)

    output = block(inputs)
    output.square().mean().backward()

    assert output.shape == inputs.shape
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"num_states": 1}, ValueError, "at least two"),
        ({"max_duration": 0}, ValueError, "positive integer"),
        ({"hidden_channels": True}, ValueError, "positive integer"),
        ({"num_blocks": 0}, ValueError, "positive integer"),
        ({"kernel_size": 4}, ValueError, "odd"),
        ({"dropout": -0.1}, ValueError, "non-negative"),
        ({"dropout": 1.0}, ValueError, "less than one"),
    ],
)
def test_invalid_architecture_configuration_fails(overrides, error, message):
    with pytest.raises(error, match=message):
        _network(**overrides)


@pytest.mark.parametrize(
    ("inputs", "error", "message"),
    [
        ([[[1.0]]], TypeError, "torch.Tensor"),
        (torch.ones(2, 6), ValueError, "shape"),
        (torch.ones(2, 2, 6), ValueError, "shape"),
        (torch.ones(0, 1, 6), ValueError, "batch/time >= 1"),
        (torch.ones(2, 1, 0), ValueError, "batch/time >= 1"),
        (torch.ones(2, 1, 6, dtype=torch.int64), TypeError, "floating"),
        (
            torch.full((2, 1, 6), float("nan")),
            ValueError,
            "finite",
        ),
    ],
)
def test_invalid_network_inputs_fail_closed(inputs, error, message):
    with pytest.raises(error, match=message):
        _network().emission_scores(inputs)


@pytest.mark.parametrize(
    ("states", "error", "message"),
    [
        ([[0, 0, 0, 1, 1, 1]], TypeError, "torch.Tensor"),
        (torch.zeros((2, 6), dtype=torch.float32), TypeError, "torch.int64"),
        (torch.zeros(12, dtype=torch.int64), TypeError, "two-dimensional"),
        (torch.zeros((1, 6), dtype=torch.int64), ValueError, "shape"),
        (
            torch.tensor([[0, 0, 0, 1, 1, 2], [1, 1, 1, 0, 0, 0]]),
            ValueError,
            "between 0 and 1",
        ),
    ],
)
def test_invalid_training_labels_fail_closed(states, error, message):
    inputs, _ = _batch()
    with pytest.raises(error, match=message):
        _network().negative_log_likelihood(inputs, states)


def test_gold_segment_longer_than_duration_support_is_rejected():
    inputs, _ = _batch()
    states = torch.zeros((2, inputs.shape[2]), dtype=torch.int64)

    with pytest.raises(ValueError, match="longer than max_duration"):
        _network(max_duration=3).negative_log_likelihood(inputs, states)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (([0.5, 0.5], [[0.5, 0.5], [1.0, 0.0]], [[0.5] * 6] * 2), "zero diagonal"),
        (([1.0, 0.0], [[0.0, 1.0], [1.0, 0.0]], [[0.5] * 6] * 2), "positive and sum"),
        (
            ([0.5, 0.5], [[0.0, 1.0], [1.0, 0.0]], [[1.0, 0, 0, 0, 0, 0]] * 2),
            "positive",
        ),
    ],
)
def test_invalid_classical_initialization_fails(values, message):
    with pytest.raises(ValueError, match=message):
        _network().initialize_structure(*values)


def test_network_reuses_shared_dynamic_program_instead_of_copying_it():
    source = Path(inspect.getfile(NeuralSemiMarkovNetwork)).read_text(encoding="utf-8")

    assert "semi_markov_nll" in source
    assert "semi_markov_viterbi" in source
    assert "logsumexp" not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_loss_and_decode_are_finite():
    network = _network().cuda()
    inputs, states = _batch()
    inputs = inputs.cuda()
    states = states.cuda()

    loss = network.negative_log_likelihood(inputs, states)
    decoded = network.decode(inputs)

    assert torch.isfinite(loss)
    assert all(result.states.device.type == "cuda" for result in decoded)
