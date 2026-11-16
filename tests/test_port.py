"""
Тесты-оракулы: перенос на PyTorch должен совпадать с эталоном.

Запуск:  pytest -q
Каждая функция проверяется отдельно — так сразу видно, ГДЕ ошибка
(как в докладе: изолированные модули, а не "вся модель целиком").
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from port import l96_torch as port  # noqa: E402

ORACLES = ROOT / "oracles" / "oracles.npz"
RTOL, ATOL = 1e-10, 1e-10  # допуск: перенос должен быть практически точным


@pytest.fixture(scope="module")
def o():
    if not ORACLES.exists():
        pytest.fail("Нет оракулов. Сначала запусти: python oracles/make_oracles.py")
    return dict(np.load(ORACLES))


def t(a):
    return torch.as_tensor(a, dtype=torch.float64)


def check(got, expected, name):
    assert isinstance(got, torch.Tensor), f"{name}: должна вернуть torch.Tensor"
    np.testing.assert_allclose(got.detach().numpy(), expected, rtol=RTOL, atol=ATOL,
                               err_msg=f"{name} расходится с оракулом")


def test_coupling(o):
    for Y, exp in zip(o["Y_in"], o["coupling"]):
        check(port.coupling(t(Y)), exp, "coupling")


def test_rhs_x(o):
    for X, Y, exp in zip(o["X_in"], o["Y_in"], o["rhs_x"]):
        check(port.rhs_x(t(X), t(Y)), exp, "rhs_x")


def test_rhs_y(o):
    for X, Y, exp in zip(o["X_in"], o["Y_in"], o["rhs_y"]):
        check(port.rhs_y(t(X), t(Y)), exp, "rhs_y")


def test_rk4_step(o):
    dt = float(o["dt"])
    for X, Y, eX, eY in zip(o["X_in"], o["Y_in"], o["rk4_X"], o["rk4_Y"]):
        Xn, Yn = port.rk4_step(t(X), t(Y), dt)
        check(Xn, eX, "rk4_step (X)")
        check(Yn, eY, "rk4_step (Y)")


def test_batched(o):
    """Все случаи разом одним батчем — результат тот же, что по одному."""
    dt = float(o["dt"])
    check(port.rhs_x(t(o["X_in"]), t(o["Y_in"])), o["rhs_x"], "rhs_x (батч)")
    check(port.rhs_y(t(o["X_in"]), t(o["Y_in"])), o["rhs_y"], "rhs_y (батч)")
    Xn, Yn = port.rk4_step(t(o["X_in"]), t(o["Y_in"]), dt)
    check(Xn, o["rk4_X"], "rk4_step (батч, X)")
    check(Yn, o["rk4_Y"], "rk4_step (батч, Y)")


def test_integrate(o):
    dt, n = float(o["dt"]), int(o["n_steps"])
    tX, tY = port.integrate(t(o["X_in"][0]), t(o["Y_in"][0]), dt, n)
    check(tX, o["traj_X"], "integrate (X)")
    check(tY, o["traj_Y"], "integrate (Y)")


def test_differentiable(o):
    """Главная причина переноса: через модель должен проходить градиент."""
    X = t(o["X_in"][0]).requires_grad_(True)
    Y = t(o["Y_in"][0])
    Xn, _ = port.rk4_step(X, Y, float(o["dt"]))
    Xn.sum().backward()
    assert X.grad is not None and torch.isfinite(X.grad).all(), "градиент не проходит"
