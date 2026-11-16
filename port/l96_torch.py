"""
ПЕРЕНОС эталонной модели на PyTorch — это заполняет агент (Задание 1).

Правила:
- Сигнатуры и смысл функций те же, что в reference/l96_numpy.py.
- На вход и выход идут torch.Tensor (float64), а не numpy.
- Функции должны работать и для БАТЧА состояний: X формы (..., K), Y формы (..., K*J).
  (Это то, ради чего вообще переносят на PyTorch: считать тысячи вариантов разом на GPU
  и брать градиенты.)
- Тесты: tests/test_port.py. Пока они красные — работа не закончена.
"""
import torch

K = 8
J = 32
F = 20.0
H = 1.0
C = 10.0
B = 10.0


def coupling(Y, K=K, J=J, h=H, c=C, b=B):
    return -(h * c / b) * Y.reshape(*Y.shape[:-1], K, J).sum(dim=-1)


def rhs_x(X, Y, F=F, K=K, J=J, h=H, c=C, b=B):
    adv = -torch.roll(X, 1, dims=-1) * (torch.roll(X, 2, dims=-1) - torch.roll(X, -1, dims=-1))
    return adv - X + F + coupling(Y, K, J, h, c, b)


def rhs_y(X, Y, K=K, J=J, h=H, c=C, b=B):
    adv = -c * b * torch.roll(Y, -1, dims=-1) * (torch.roll(Y, -2, dims=-1) - torch.roll(Y, 1, dims=-1))
    return adv - c * Y + (h * c / b) * torch.repeat_interleave(X, J, dim=-1)


def rk4_step(X, Y, dt, F=F, K=K, J=J, h=H, c=C, b=B):
    """Один шаг метода Рунге–Кутты 4-го порядка для всей системы."""
    def f(X, Y):
        return rhs_x(X, Y, F, K, J, h, c, b), rhs_y(X, Y, K, J, h, c, b)

    k1x, k1y = f(X, Y)
    k2x, k2y = f(X + 0.5 * dt * k1x, Y + 0.5 * dt * k1y)
    k3x, k3y = f(X + 0.5 * dt * k2x, Y + 0.5 * dt * k2y)
    k4x, k4y = f(X + dt * k3x, Y + dt * k3y)
    X_new = X + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
    Y_new = Y + dt / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
    return X_new, Y_new


def integrate(X0, Y0, dt, n_steps, **params):
    """Траектории формы (n_steps + 1, ..., K) и (n_steps + 1, ..., K*J)."""
    Xs, Ys = [X0], [Y0]
    X, Y = X0, Y0
    for _ in range(n_steps):
        X, Y = rk4_step(X, Y, dt, **params)
        Xs.append(X)
        Ys.append(Y)
    return torch.stack(Xs), torch.stack(Ys)
