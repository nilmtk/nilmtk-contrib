import pytest


torch = pytest.importorskip("torch")

from nilmtk_contrib.torch._state_fitting import (  # noqa: E402
    assign_states,
    fit_state_means,
)


def test_state_fitting_is_ordered_deterministic_and_window_invariant():
    windows = (
        torch.tensor([0.0, 1.0, 100.0], dtype=torch.float64),
        torch.tensor([99.0, 2.0, 101.0], dtype=torch.float64),
    )

    first = fit_state_means(windows, num_states=2, max_iterations=20)
    second = fit_state_means(tuple(reversed(windows)), num_states=2, max_iterations=20)

    assert first.tolist() == pytest.approx([1.0, 100.0])
    assert torch.equal(first, second)


def test_assignment_breaks_exact_ties_toward_the_lower_ordered_state():
    states = assign_states(
        torch.tensor([0.0, 5.0, 10.0], dtype=torch.float64),
        torch.tensor([0.0, 10.0], dtype=torch.float64),
    )

    assert states.tolist() == [0, 0, 1]


def test_state_fitting_rejects_insufficient_support_and_nonconvergence():
    with pytest.raises(ValueError, match="unique values"):
        fit_state_means(
            (torch.ones(4, dtype=torch.float64),),
            num_states=2,
            max_iterations=10,
        )
    with pytest.raises(RuntimeError, match="did not converge"):
        fit_state_means(
            (torch.tensor([0.0, 1.0, 9.0, 10.0], dtype=torch.float64),),
            num_states=2,
            max_iterations=1,
        )
