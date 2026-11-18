"""Compare forecasts: MPLCONFIGDIR=/tmp/l96-mpl python -m hybrid.forecast."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from hybrid import coarse
from hybrid.neural import load_model
from port import l96_torch as full


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", action="store_true", help="Include rollout-trained network")
    args = parser.parse_args()
    seed, batch, burn_in, n_steps, dt = 43, 20, 5000, 2000, 0.001
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    X = torch.from_numpy(rng.normal(0, 5, (batch, full.K)))
    Y = torch.from_numpy(rng.normal(0, 0.5, (batch, full.K * full.J)))
    for _ in range(burn_in):
        X, Y = full.rk4_step(X, Y, dt)
    X0, Y0 = X.clone(), Y.clone()
    plain, corrected = X.clone(), X.clone()
    root = Path(__file__).resolve().parents[1]
    with np.load(root / 'hybrid' / 'polynomial.npz') as data:
        coefficients = data['coefficients'].copy()
    network = load_model(root / "hybrid" / "neural.pt")
    rollout_network = load_model(root / "hybrid/neural_rollout.pt") if args.rollout else None
    rollout = X.clone()
    rmse_rollout = np.zeros(n_steps + 1)
    neural = X.clone()
    rmse_neural = np.zeros(n_steps + 1)
    rmse_plain, rmse_poly = np.zeros(n_steps + 1), np.zeros(n_steps + 1)
    for step in range(1, n_steps + 1):
        X, Y = full.rk4_step(X, Y, dt)
        plain = coarse.rk4_step(plain, dt)
        corrected = coarse.rk4_step(corrected, dt, coefficients=coefficients)
        neural = coarse.rk4_step(neural, dt, correction=network)
        if rollout_network is not None:
            rollout = coarse.rk4_step(rollout, dt, correction=rollout_network)
            rmse_rollout[step] = torch.sqrt(torch.mean((rollout - X) ** 2)).item()
        rmse_neural[step] = torch.sqrt(torch.mean((neural - X) ** 2)).item()
        rmse_plain[step] = torch.sqrt(torch.mean((plain - X) ** 2)).item()
        rmse_poly[step] = torch.sqrt(torch.mean((corrected - X) ** 2)).item()
    if not (np.isfinite(rmse_plain).all() and np.isfinite(rmse_poly).all() and np.isfinite(rmse_neural).all()):
        raise ValueError('Non-finite forecast errors')
    if not np.isfinite(rmse_rollout).all():
        raise ValueError("Non-finite rollout forecast errors")
    extra = {}
    if args.rollout:
        extra = dict(rmse_rollout=rmse_rollout, rollout_checkpoint="hybrid/neural_rollout.pt")
    time = np.arange(n_steps + 1) * dt
    results = root / 'results'
    results.mkdir(exist_ok=True)
    np.savez_compressed(results / 'forecast_rmse.npz', time=time,
                        rmse_full=np.zeros_like(time), rmse_zero=rmse_plain,
                        rmse_neural=rmse_neural, neural_checkpoint="hybrid/neural.pt",
                        rmse_polynomial=rmse_poly, X0=X0.numpy(), Y0=Y0.numpy(),
                        coefficients=coefficients, seed=seed, batch=batch,
                        burn_in=burn_in, n_steps=n_steps, dt=dt,
                        K=full.K, J=full.J, F=full.F, h=full.H, c=full.C, b=full.B, **extra)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time, np.zeros_like(time), label='Full model (reference)', color='black', ls=':')
    ax.plot(time, rmse_plain, label='Coarse, U = 0')
    ax.plot(time, rmse_poly, label='Coarse, cubic U(X)')
    ax.plot(time, rmse_neural, label="Coarse, neural U(X)")
    if args.rollout:
        ax.plot(time, rmse_rollout, label="Coarse, rollout-trained neural U(X)")
    ax.set(xlabel='Forecast lead time (model units)', ylabel='RMSE of X',
           title=f'Lorenz-96 forecasts: {batch} initial states, dt = {dt}')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results / 'forecast_rmse.png', dpi=180)
    plt.close(fig)
    print('RMSE averaged over 20 states and 8 X variables; independent seed=43')
    for lead in (0.1, 0.5, 1.0, 2.0):
        i = round(lead / dt)
        print(f't={lead:.1f}: U=0 {rmse_plain[i]:.6f}; polynomial {rmse_poly[i]:.6f}; neural {rmse_neural[i]:.6f}')
    if args.rollout:
        for lead in (0.1, 0.5, 1.0, 2.0):
            print(f't={lead:.1f}: rollout neural {rmse_rollout[round(lead / dt)]:.6f}')
    print('Saved results/forecast_rmse.png and results/forecast_rmse.npz')


if __name__ == '__main__':
    main()
