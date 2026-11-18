"""Shape and differentiability contracts for a pointwise correction."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid.neural import NeuralCorrection


def test_pointwise_batch_and_gradients():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = NeuralCorrection()
        X = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    got = model(X)
    assert got.shape == X.shape and got.dtype == torch.float64
    torch.testing.assert_close(got.flatten(), model(X.flatten()))
    got.sum().backward()
    assert X.grad is not None and torch.isfinite(X.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_neural_correction_through_integrator():
    from hybrid import coarse
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = NeuralCorrection()
        X = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    # A constant forcing correction must equal adding it to F at every RK stage.
    torch.testing.assert_close(
        coarse.rk4_step(X, 0.001, correction=lambda x: torch.full_like(x, 2.0)),
        coarse.rk4_step(X, 0.001, F=22.0))
    trajectory = coarse.integrate(X, 0.001, 5, correction=model)
    assert trajectory.shape == (6, 2, 3, 8)
    assert torch.isfinite(trajectory).all()
    trajectory[-1].square().mean().backward()
    assert torch.isfinite(X.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_rollout_loss_matches_trajectory_and_updates_network():
    from hybrid import coarse
    from hybrid.train_rollout import make_windows, rollout_loss
    with torch.random.fork_rng():
        torch.manual_seed(44)
        model = NeuralCorrection()
        X = torch.randn(2, 8, dtype=torch.float64)
    target = coarse.integrate(X, 0.001, 3).detach().transpose(0, 1)
    loss = rollout_loss(model, target, 0.001)
    assert loss.item() > 0
    loss.backward()
    assert any(p.grad.abs().sum() > 0 for p in model.parameters())
    timeline = torch.arange(80, dtype=torch.float64).reshape(10, 8)
    windows = make_windows(timeline, 3, 2)
    torch.testing.assert_close(windows[1], timeline[2:6])


def test_neighbor_correction_is_cyclic_and_checkpoint_roundtrip(tmp_path):
    from hybrid.neural import load_model
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = NeuralCorrection([8, 8], activation='silu', offsets=[-1, 0, 1])
        X = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(model(torch.roll(X, 2, dims=-1)),
                               torch.roll(model(X), 2, dims=-1))
    model(X).sum().backward()
    assert torch.isfinite(X.grad).all()
    path = tmp_path / 'neighbors.pt'
    torch.save(dict(state_dict=model.state_dict(), architecture=model.architecture,
                    activation=model.activation, offsets=list(model.offsets)), path)
    torch.testing.assert_close(model(X), load_model(path)(X))
