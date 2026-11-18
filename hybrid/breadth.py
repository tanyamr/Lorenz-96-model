"""Quick controlled breadth study: python -m hybrid.breadth."""
import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

from hybrid.neural import NeuralCorrection, load_model
from hybrid import coarse
from hybrid.train_rollout import make_windows, rollout_loss
from port import l96_torch as full

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = ROOT / 'results/breadth'
    output.mkdir(parents=True, exist_ok=True)
    configs = ROOT / 'hybrid/configs/breadth'
    configs.mkdir(parents=True, exist_ok=True)
    common = dict(seed=42, epochs=10, batch_size=256, lr=0.001,
                  train_fraction=0.8, train_stride=5, dtype='float64', threads=1,
                  data='hybrid/data.npz', forecast_seed=45, forecast_batch=20,
                  burn_in=5000, forecast_steps=2000, dt=0.001)
    variants = [
        dict(name='control', hidden_sizes=[32,32], activation='tanh', offsets=[0]),
        dict(name='wide', hidden_sizes=[64,64], activation='tanh', offsets=[0]),
        dict(name='deep', hidden_sizes=[32,32,32], activation='tanh', offsets=[0]),
        dict(name='silu', hidden_sizes=[32,32], activation='silu', offsets=[0]),
        dict(name='neighbors', hidden_sizes=[32,32], activation='tanh', offsets=[-1,0,1]),
        dict(name='rollout', hidden_sizes=[32,32], activation='tanh', offsets=[0],
             horizon=100, window_stride=400, rollout_epochs=5, rollout_lr=0.0001,
             rollout_batch=32, grad_clip=1.0),
    ]
    with np.load(ROOT / common['data']) as data:
        X, Y = torch.from_numpy(data['X'].copy()), torch.from_numpy(data['coupling'].copy())
    split = int(len(X)*common['train_fraction'])
    # Validation is chronological, full resolution, excluded from training and normalization.
    x, y = X[:split:common['train_stride']], Y[:split:common['train_stride']]
    xv, yv = X[split:], Y[split:]
    rows, models = [], {}
    for variant in variants:
        config = dict(common, **variant)
        name = config['name']
        (configs / f'{name}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        torch.manual_seed(config['seed'])
        start = perf_counter()
        model = NeuralCorrection(config['hidden_sizes'], config['activation'], config['offsets'])
        with torch.no_grad():
            for key, value in [('x_mean', x.mean()), ('x_std', x.std()),
                               ('y_mean', y.mean()), ('y_std', y.std())]:
                getattr(model, key).copy_(value)
        history = []
        if name == 'rollout':
            model = load_model(output / 'control.pt')
            train_windows = make_windows(X[:split], config['horizon'], config['window_stride'])
            val_windows = make_windows(X[split:], config['horizon'], config['window_stride'])
            optimizer = torch.optim.Adam(model.parameters(), lr=config['rollout_lr'])
            @torch.no_grad()
            def validation_loss():
                return sum(rollout_loss(model,b,config['dt']).item()*len(b)
                           for b in val_windows.split(config['rollout_batch']))/len(val_windows)
            best = validation_loss()
            best_state, best_epoch = copy.deepcopy(model.state_dict()), 0
            for epoch in range(1, config['rollout_epochs']+1):
                for indices in torch.randperm(len(train_windows)).split(config['rollout_batch']):
                    optimizer.zero_grad()
                    loss = rollout_loss(model, train_windows[indices], config['dt'])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
                    optimizer.step()
                val = validation_loss()
                history.append(dict(epoch=epoch, validation_rollout_mse=val))
                if val < best:
                    best, best_state, best_epoch = val, copy.deepcopy(model.state_dict()), epoch
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
            best, best_epoch, best_state = float('inf'), 0, None
            for epoch in range(1, config['epochs']+1):
                for indices in torch.randperm(len(x)).split(config['batch_size']):
                    optimizer.zero_grad()
                    loss = (model(x[indices])-y[indices]).square().mean()
                    loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    val = sum((model(a)-b).square().sum().item() for a,b in zip(xv.split(512),yv.split(512)))/yv.numel()
                if not np.isfinite(val):
                    raise ValueError(f'{name}: non-finite validation loss')
                history.append(dict(epoch=epoch, validation_coupling_mse=val))
                if val < best:
                    best, best_state, best_epoch = val, copy.deepcopy(model.state_dict()), epoch
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            rmse = (sum((model(a)-b).square().sum().item() for a,b in zip(xv.split(512),yv.split(512)))/yv.numel())**0.5
        torch.save(dict(state_dict=best_state, architecture=model.architecture,
                        activation=model.activation, offsets=list(model.offsets)), output / f'{name}.pt')
        # Evaluate reloaded artifacts, not just in-memory models.
        models[name] = load_model(output / f'{name}.pt')
        row = dict(name=name, validation_coupling_rmse=rmse, best_epoch=best_epoch,
                   training_seconds=perf_counter()-start, config=config, history=history)
        rows.append(row)
        print(f'{name}: coupling RMSE={rmse:.6f}, epoch={best_epoch}', flush=True)
    models['previous_supervised'] = load_model(ROOT / 'hybrid/neural.pt')
    with torch.no_grad():
        rng = np.random.default_rng(common['forecast_seed'])
        Xf = torch.from_numpy(rng.normal(0,5,(common['forecast_batch'],full.K)))
        Yf = torch.from_numpy(rng.normal(0,.5,(common['forecast_batch'],full.K*full.J)))
        for _ in range(common['burn_in']):
            Xf,Yf = full.rk4_step(Xf,Yf,common['dt'])
        X0,Y0 = Xf.clone(),Yf.clone()
        predictions = {name:Xf.clone() for name in models}
        curves = {name:np.zeros(common['forecast_steps']+1) for name in models}
        for step in range(1, common['forecast_steps']+1):
            Xf,Yf = full.rk4_step(Xf,Yf,common['dt'])
            for name,model in models.items():
                predictions[name] = coarse.rk4_step(predictions[name],common['dt'],correction=model)
                curves[name][step] = (predictions[name]-Xf).square().mean().sqrt().item()
    for row in rows:
        curve = curves[row['name']]
        row['forecast_finite'] = bool(np.isfinite(curve).all())
        row['forecast_rmse'] = {str(t):float(curve[round(t/common['dt'])]) for t in (.1,.5,1.,2.)}
        row['mean_forecast_rmse_0_to_1'] = float(curve[1:1001].mean())
    payload = dict(protocol=common, selection_metric='mean_forecast_rmse_0_to_1',
                   experiments=rows, previous_supervised_rmse={str(t):float(curves['previous_supervised'][round(t/common['dt'])]) for t in (.1,.5,1.,2.)})
    (output / 'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    np.savez_compressed(output / 'curves.npz', time=np.arange(common['forecast_steps']+1)*common['dt'], X0=X0.numpy(),Y0=Y0.numpy(),**curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for name,curve in curves.items():
        ax.plot(np.arange(len(curve))*common['dt'],curve,label=name)
    ax.set(xlabel='Forecast lead time',ylabel='RMSE of X',title='Quick breadth study: 20 states, seed 45')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(output/'forecast_rmse.png',dpi=160); plt.close(fig)
    lines=['\n## 2026-11-18 — Задание 4: быстрый поиск вширь\n',
           '- Команда: `MPLCONFIGDIR=/tmp/l96-mpl .venv/bin/python -m hybrid.breadth`.\n',
           '- Пять гипотез плюс заново обученный контроль. Все supervised-варианты: 10 эпох, seed=42, Adam lr=0.001, batch=256 состояний; первые 80% траектории, каждое пятое состояние (7200 состояний). Валидация: последние 20%, 9000 состояний; нормализация только по обучающей части. float64, CPU, один поток.\n',
           '- Rollout: дообучение быстрого контроля, N=100, 5 эпох, lr=0.0001, batch=32 окон, stride=400, clip=1; лучший checkpoint по ошибке траектории, включая эпоху 0. Остальные выбираются по coupling MSE.\n',
           '- Прогноз: новый seed=45, 20 состояний, разгон 5000, dt=0.001, 2000 шагов. Все варианты имеют одинаковые начальные X/Y. Стандартные K=8,J=32,F=20,h=1,c=10,b=10. Основной критерий поиска — средний RMSE за t=(0,1]; t=2 приведён дополнительно. Это набор для выбора гипотез, не окончательный независимый тест.\n',
           '\n| Вариант | RMSE coupling (val) | RMSE t=0.5 | t=1 | t=2 | Средний RMSE (0,1] |\n|---|---:|---:|---:|---:|---:|\n']
    for row in rows:
        f=row['forecast_rmse']
        lines.append(f"| {row['name']} | {row['validation_coupling_rmse']:.6f} | {f['0.5']:.6f} | {f['1.0']:.6f} | {f['2.0']:.6f} | {row['mean_forecast_rmse_0_to_1']:.6f} |\n")
    finite=[r for r in rows if r['forecast_finite']]
    winner=min(finite,key=lambda r:r['mean_forecast_rmse_0_to_1'])
    lines.append(f"\n- Лучший по критерию: {winner['name']}. Сравнение одного seed и короткого бюджета — предварительное; ширина/глубина меняют вычислительную стоимость. Rollout требует уже обученного контроля. Предыдущая сеть 40 эпох оценена отдельно в summary.json, но не является контролем одинакового бюджета.\n")
    lines.append('- Конфиги: hybrid/configs/breadth/*.yaml; веса, полные метрики/истории, времена и кривые: results/breadth/. Все варианты, включая ухудшения, сохранены.\n')
    with (ROOT/'results/log.md').open('a') as stream:
        stream.writelines(lines)
    print(json.dumps([{k:v for k,v in r.items() if k not in ('config','history')} for r in rows],indent=2),flush=True)


if __name__ == '__main__':
    main()
