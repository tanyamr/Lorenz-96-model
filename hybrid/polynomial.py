"""Кубическая поправка и её подбор: python -m hybrid.polynomial."""
from pathlib import Path

import numpy as np


def polynomial(X, coefficients):
    """U(X) по схеме Горнера; коэффициенты идут от a0 до a3."""
    a0, a1, a2, a3 = coefficients
    return ((a3 * X + a2) * X + a1) * X + a0


def fit_polynomial(X, coupling):
    """Общий для всех k полином, МНК по всем парам (X_k, coupling_k)."""
    X, coupling = np.asarray(X), np.asarray(coupling)
    if X.shape != coupling.shape or X.size == 0:
        raise ValueError('X and coupling must have identical nonempty shapes')
    if not (np.isfinite(X).all() and np.isfinite(coupling).all()):
        raise ValueError('X and coupling must be finite')
    design = np.polynomial.polynomial.polyvander(X.ravel(), 3)
    coefficients, _, rank, _ = np.linalg.lstsq(design, coupling.ravel(), rcond=None)
    if rank != 4:
        raise ValueError('Insufficient independent data for a cubic polynomial')
    return coefficients


def main():
    directory = Path(__file__).resolve().parent
    with np.load(directory / 'data.npz') as data:
        X, target = data['X'], data['coupling']
    coefficients = fit_polynomial(X, target)
    rmse = float(np.sqrt(np.mean((polynomial(X, coefficients) - target) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(target ** 2)))
    np.savez(directory / 'polynomial.npz', coefficients=coefficients,
             rmse=rmse, zero_rmse=zero_rmse, n_samples=X.size)
    print('a0, a1, a2, a3:', coefficients.tolist())
    print(f'Samples: {X.size}; training RMSE: {rmse:.8f}; U=0 RMSE: {zero_rmse:.8f}')


if __name__ == '__main__':
    main()
