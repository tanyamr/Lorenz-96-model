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
    raise NotImplementedError("Задание 1: перенеси coupling из reference/l96_numpy.py")


def rhs_x(X, Y, F=F, K=K, J=J, h=H, c=C, b=B):
    raise NotImplementedError("Задание 1: перенеси rhs_x")


def rhs_y(X, Y, K=K, J=J, h=H, c=C, b=B):
    raise NotImplementedError("Задание 1: перенеси rhs_y")


def rk4_step(X, Y, dt, F=F, K=K, J=J, h=H, c=C, b=B):
    raise NotImplementedError("Задание 1: перенеси rk4_step")


def integrate(X0, Y0, dt, n_steps, **params):
    raise NotImplementedError("Задание 1: перенеси integrate (возвращает тензоры (n+1, K) и (n+1, K*J))")
