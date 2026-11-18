"""Generate full-model training data: python hybrid/generate_data.py."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference import l96_numpy as ref


def main():
    seed, n_steps, burn_in, dt = 42, 50_000, 5_000, 0.001
    X, Y = ref.random_state(np.random.default_rng(seed))
    X_data = np.empty((n_steps - burn_in, ref.K), dtype=np.float64)
    coupling_data = np.empty_like(X_data)
    for step in range(1, n_steps + 1):
        X, Y = ref.rk4_step(X, Y, dt)
        if step > burn_in:
            index = step - burn_in - 1
            X_data[index] = X
            coupling_data[index] = ref.coupling(Y)
    if not (np.isfinite(X_data).all() and np.isfinite(coupling_data).all()):
        raise ValueError('Non-finite values in generated trajectory')
    path = ROOT / 'hybrid' / 'data.npz'
    np.savez_compressed(
        path, X=X_data, coupling=coupling_data,
        time=np.arange(burn_in + 1, n_steps + 1) * dt,
        seed=seed, dt=dt, n_steps=n_steps, burn_in=burn_in,
        K=ref.K, J=ref.J, F=ref.F, h=ref.H, c=ref.C, b=ref.B,
    )
    print(f'Saved {path.relative_to(ROOT)}: X and coupling {X_data.shape}, float64')
    print(f'X range: [{X_data.min():.6f}, {X_data.max():.6f}]')
    print(f'coupling range: [{coupling_data.min():.6f}, {coupling_data.max():.6f}]')


if __name__ == '__main__':
    main()
