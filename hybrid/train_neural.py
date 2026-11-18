"""Reproduce supervised training: python -m hybrid.train_neural."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from hybrid.neural import NeuralCorrection
from hybrid.polynomial import fit_polynomial, polynomial


def main(config=None):
    from hybrid.experiment import ROOT, load_config
    if config is None:
        config = load_config(ROOT / 'hybrid/configs/supervised.yaml')
    seed, epochs, batch_size, lr = (config[k] for k in ('seed', 'epochs', 'batch_size', 'lr'))
    torch.manual_seed(seed)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parents[1]
    with np.load(root / config['data']) as data:
        X = torch.from_numpy(data['X'].copy())
        Y = torch.from_numpy(data['coupling'].copy())
    split = int(len(X) * config['train_fraction'])
    if split < 2 or len(X) - split < 2:
        raise ValueError("Both data splits need at least two states")
    # Chronological holdout; normalization uses training data only.
    x, y = X[:split].reshape(-1), Y[:split].reshape(-1)
    xv, yv = X[split:].reshape(-1), Y[split:].reshape(-1)
    model = NeuralCorrection(config["hidden_sizes"])
    with torch.no_grad():
        model.x_mean.copy_(x.mean())
        model.x_std.copy_(x.std())
        model.y_mean.copy_(y.mean())
        model.y_std.copy_(y.std())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_epoch, best_state = float('inf'), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(x.numel())
        for indices in order.split(batch_size):
            loss = ((model(x[indices]) - y[indices]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_mse = torch.cat([(model(a) - b) ** 2 for a, b in zip(x.split(batch_size), y.split(batch_size))]).mean().item()
            val_mse = torch.cat([(model(a) - b) ** 2 for a, b in zip(xv.split(batch_size), yv.split(batch_size))]).mean().item()
        if not np.isfinite([train_mse, val_mse]).all():
            raise ValueError('Non-finite training loss')
        history.append(dict(epoch=epoch, train_mse=train_mse, validation_mse=val_mse))
        if val_mse < best_loss:
            best_loss, best_epoch = val_mse, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: train RMSE={train_mse**0.5:.6f}, validation RMSE={val_mse**0.5:.6f}', flush=True)
    model.load_state_dict(best_state)
    # Fit comparison polynomial on the same training subset (no holdout leakage).
    coefficients = fit_polynomial(x.numpy(), y.numpy())
    with torch.no_grad():
        train_rmse = torch.mean((model(x) - y)**2).sqrt().item()
    metrics = dict(seed=seed, epochs=epochs, batch_size=batch_size, lr=lr,
                   train_states=split, validation_states=len(X)-split,
                   best_epoch=best_epoch, train_rmse=train_rmse,
                   validation_rmse=best_loss**0.5,
                   zero_validation_rmse=float(torch.mean(yv**2).sqrt()),
                   polynomial_validation_rmse=float(np.sqrt(np.mean((polynomial(xv.numpy(), coefficients)-yv.numpy())**2))),
                   dtype='float64', threads=config['threads'], torch_version=str(torch.__version__), history=history)
    metrics['config'] = config
    for key in ('checkpoint', 'metrics'):
        (root / config[key]).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=best_state, seed=seed, best_epoch=best_epoch,
                    architecture=model.architecture, activation='tanh'), root / config['checkpoint'])
    (root / config['metrics']).write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps({k: v for k, v in metrics.items() if k != 'history'}, indent=2))


if __name__ == '__main__':
    main()
