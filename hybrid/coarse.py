"""Огрублённая Lorenz-96: только X, с полиномиальной или нейросетевой поправкой.

Входы — torch.Tensor float64 формы (..., K); ведущие оси задают батч.
"""
import torch

from hybrid.polynomial import polynomial

F = 20.0


def rhs_x(X, F=F, coefficients=None, correction=None):
    """dX/dt; без поправки используется U = 0."""
    adv = -torch.roll(X, 1, dims=-1) * (
        torch.roll(X, 2, dims=-1) - torch.roll(X, -1, dims=-1)
    )
    if coefficients is not None and correction is not None:
        raise ValueError("Choose coefficients or correction, not both")
    result = adv - X + F
    if coefficients is not None:
        result = result + polynomial(X, coefficients)
    if correction is not None:
        result = result + correction(X)
    return result


def rk4_step(X, dt, F=F, coefficients=None, correction=None):
    """Один шаг Рунге–Кутты четвёртого порядка."""
    k1 = rhs_x(X, F, coefficients, correction)
    k2 = rhs_x(X + 0.5 * dt * k1, F, coefficients, correction)
    k3 = rhs_x(X + 0.5 * dt * k2, F, coefficients, correction)
    k4 = rhs_x(X + dt * k3, F, coefficients, correction)
    return X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(X0, dt, n_steps, **params):
    """Траектория формы (n_steps + 1, ..., K), включая начальное состояние."""
    Xs = [X0]
    X = X0
    for _ in range(n_steps):
        X = rk4_step(X, dt, **params)
        Xs.append(X)
    return torch.stack(Xs)
