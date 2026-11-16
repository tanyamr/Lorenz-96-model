"""
ЭТАЛОННАЯ модель Lorenz-96 (двухуровневая) на NumPy.

Роль этого файла в проекте — как у старого кода на Fortran в докладе про WRF:
это "истина", которую мы НЕ меняем. Новый код (PyTorch) должен давать
те же числа, что и этот.

Модель (Lorenz 1996; параметры как в Arnold et al., 2013):
  X_k  — K "крупных" переменных (как погода на крупной сетке)
  Y_jk — J "мелких" переменных на каждую X_k (как мелкие вихри/облака)

  dX_k/dt  = -X_{k-1}(X_{k-2} - X_{k+1}) - X_k + F - (h c / b) * sum_j Y_{j,k}
  dY_jk/dt = -c b Y_{j+1,k}(Y_{j+2,k} - Y_{j-1,k}) - c Y_{j,k} + (h c / b) X_k

Индексы циклические (переменные стоят по кругу).
Y хранится как плоский массив длины K*J: сначала J штук для X_0, потом для X_1 и т.д.
"""
import numpy as np

# Параметры по умолчанию
K = 8      # число крупных переменных
J = 32     # мелких переменных на одну крупную
F = 20.0   # внешнее воздействие ("солнце")
H = 1.0    # сила связи между уровнями
C = 10.0   # во сколько раз мелкие переменные быстрее
B = 10.0   # во сколько раз мелкие переменные меньше по амплитуде


def coupling(Y, K=K, J=J, h=H, c=C, b=B):
    """Вклад мелких переменных в каждую X_k: -(h c / b) * sum_j Y_{j,k}.

    Именно этот член "не видит" огрублённая модель — его потом будет
    предсказывать нейросеть.
    """
    return -(h * c / b) * Y.reshape(K, J).sum(axis=1)


def rhs_x(X, Y, F=F, K=K, J=J, h=H, c=C, b=B):
    """Правая часть для крупных переменных X (форма (K,))."""
    adv = -np.roll(X, 1) * (np.roll(X, 2) - np.roll(X, -1))
    return adv - X + F + coupling(Y, K, J, h, c, b)


def rhs_y(X, Y, K=K, J=J, h=H, c=C, b=B):
    """Правая часть для мелких переменных Y (форма (K*J,))."""
    adv = -c * b * np.roll(Y, -1) * (np.roll(Y, -2) - np.roll(Y, 1))
    return adv - c * Y + (h * c / b) * np.repeat(X, J)


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
    """Прогон модели на n_steps шагов. Возвращает траектории X (n+1, K) и Y (n+1, K*J)."""
    Xs, Ys = [X0], [Y0]
    X, Y = X0, Y0
    for _ in range(n_steps):
        X, Y = rk4_step(X, Y, dt, **params)
        Xs.append(X)
        Ys.append(Y)
    return np.array(Xs), np.array(Ys)


def random_state(rng, K=K, J=J):
    """Случайное начальное состояние в типичном диапазоне значений."""
    X = rng.normal(0.0, 5.0, size=K)
    Y = rng.normal(0.0, 0.5, size=K * J)
    return X, Y
