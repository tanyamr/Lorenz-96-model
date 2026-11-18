"""Fine-tune through RK4: python -m hybrid.train_rollout."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from hybrid import coarse
from hybrid.neural import load_model


def rollout_loss(model, windows, dt):
    """MSE over all forecast steps, excluding the shared initial state."""
    prediction = coarse.integrate(windows[:, 0], dt, windows.shape[1] - 1,
                                  correction=model)
    return (prediction[1:] - windows[:, 1:].transpose(0, 1)).square().mean()


def make_windows(X, horizon, stride):
    starts = torch.arange(0, len(X) - horizon, stride)
    return X[starts[:, None] + torch.arange(horizon + 1)]


def main(config=None):
    from hybrid.experiment import ROOT, load_config
    if config is None:
        config = load_config(ROOT / 'hybrid/configs/rollout.yaml')
        parser = argparse.ArgumentParser()
        parser.add_argument('--horizon', type=int)
        args = parser.parse_args()
        if args.horizon is not None:
            if args.horizon <= 0:
                parser.error('horizon must be positive')
            config['horizon'] = args.horizon
    seed, epochs, batch_size, lr = (config[k] for k in ('seed', 'epochs', 'batch_size', 'lr'))
    horizon, stride = config["horizon"], config["stride"]
    torch.manual_seed(seed)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parents[1]
    with np.load(root / config['data']) as data:
        X = torch.from_numpy(data['X'].copy())
        coupling = torch.from_numpy(data['coupling'].copy())
        dt = float(data['dt'])
    split = int(config['train_fraction'] * len(X))
    train = make_windows(X[:split], horizon, stride)
    validation = make_windows(X[split:], horizon, stride)
    model = load_model(root / config['initialization'])
    if model.architecture != [1, *config["hidden_sizes"], 1]:
        raise ValueError("Configured network size differs from initialization checkpoint")
    if not len(train) or not len(validation):
        raise ValueError("Not enough data for the configured horizon")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    @torch.no_grad()
    def evaluate(windows):
        return sum(rollout_loss(model, batch, dt).item() * len(batch)
                   for batch in windows.split(batch_size)) / len(windows)

    baseline = evaluate(validation)
    best_loss, best_epoch, best_state = baseline, 0, copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in torch.randperm(len(train)).split(batch_size):
            loss = rollout_loss(model, train[indices], dt)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite rollout loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        val = evaluate(validation)
        history.append(dict(epoch=epoch, validation_mse=val))
        if val < best_loss:
            best_loss, best_epoch, best_state = val, epoch, copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch}: validation trajectory RMSE={val**0.5:.8f}', flush=True)
    model.load_state_dict(best_state)
    with torch.no_grad():
        coupling_rmse = ((model(X[split:]) - coupling[split:])**2).mean().sqrt().item()
    metrics = dict(seed=seed, horizon=horizon, stride=stride, dt=dt, epochs=epochs,
                   batch_size=batch_size, lr=lr, grad_clip=config['grad_clip'],
                   train_windows=len(train), validation_windows=len(validation),
                   initialization=config['initialization'], best_epoch=best_epoch,
                   baseline_validation_rollout_rmse=baseline**0.5,
                   validation_rollout_rmse=best_loss**0.5,
                   validation_coupling_rmse=coupling_rmse,
                   history=history, dtype='float64', threads=config['threads'])
    metrics['config'] = config
    for key in ('checkpoint', 'metrics'):
        (root / config[key]).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=best_state, architecture=model.architecture,
                    activation='tanh', seed=seed, best_epoch=best_epoch, horizon=horizon),
               root / config['checkpoint'])
    (root / config['metrics']).write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps({k:v for k,v in metrics.items() if k != 'history'}, indent=2))


if __name__ == '__main__':
    main()
