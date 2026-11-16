"""Проверки здравого смысла для эталона (должны быть зелёными с самого начала)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference import l96_numpy as ref  # noqa: E402


def test_shapes():
    X, Y = ref.random_state(np.random.default_rng(0))
    assert ref.rhs_x(X, Y).shape == (ref.K,)
    assert ref.rhs_y(X, Y).shape == (ref.K * ref.J,)


def test_rest_state_without_coupling():
    """Без связи (h=0) состояние X_k = F, Y = 0 — равновесие: производные нулевые."""
    X = np.full(ref.K, ref.F)
    Y = np.zeros(ref.K * ref.J)
    assert np.allclose(ref.rhs_x(X, Y, h=0.0), 0.0)
    assert np.allclose(ref.rhs_y(X, Y, h=0.0), 0.0)


def test_stays_bounded():
    """Модель хаотична, но не "взрывается" на разумном шаге."""
    X, Y = ref.random_state(np.random.default_rng(1))
    Xs, Ys = ref.integrate(X, Y, 0.001, 2000)
    assert np.isfinite(Xs).all() and np.abs(Xs).max() < 100
