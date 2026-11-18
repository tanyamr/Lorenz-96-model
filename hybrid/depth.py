"""Bounded depth search (10 training runs): python -m hybrid.depth."""
import contextlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from hybrid import coarse
from hybrid.experiment import load_config
from hybrid.neural import load_model
from hybrid.train_rollout import main as train
from port import l96_torch as full

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def forecast(models, X0, Y0, steps=2000, dt=.001):
    X, Y = X0.clone(), Y0.clone()
    predicted = {name:X.clone() for name in models}
    curves = {name:np.zeros(steps+1) for name in models}
    for step in range(1, steps+1):
        X,Y = full.rk4_step(X,Y,dt)
        for name,model in models.items():
            predicted[name] = coarse.rk4_step(predicted[name],dt,correction=model)
            curves[name][step] = (predicted[name]-X).square().mean().sqrt().item()
    if not all(np.isfinite(c).all() for c in curves.values()):
        raise ValueError('Non-finite forecast')
    return curves


def score(curve):
    return float(curve[1:1001].mean())


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = ROOT/'results/depth'
    configs = ROOT/'hybrid/configs/depth'
    output.mkdir(parents=True,exist_ok=True)
    configs.mkdir(parents=True,exist_ok=True)
    base = load_config(ROOT/'hybrid/configs/rollout.yaml')
    base.update(initialization='results/breadth/control.pt', epochs=10,
                batch_size=32,stride=400)
    # Same selection metric and initial states as breadth; never select on final seed 46.
    with np.load(ROOT/'results/breadth/curves.npz') as data:
        X0,Y0=torch.from_numpy(data['X0'].copy()),torch.from_numpy(data['Y0'].copy())
    rows=[]
    for index in range(10):
        config=dict(base)
        if index < 8:
            config.update(horizon=[50,100,200,400][index//2],lr=[.0001,.001][index%2],seed=42)
        else:
            chosen=min(rows[:8],key=lambda row:row['selection_score'])['config']
            config.update(horizon=chosen['horizon'],lr=chosen['lr'],seed=43+index-8)
        name=f'run_{index+1:02d}'
        config.update(checkpoint=f'results/depth/{name}.pt',metrics=f'results/depth/{name}.json')
        config_path=configs/f'{name}.yaml'
        config_path.write_text(yaml.safe_dump(config,sort_keys=False))
        # Validate exactly the persisted config and retain the complete trainer output.
        with (output/f'{name}.txt').open('w') as stream, contextlib.redirect_stdout(stream):
            train(load_config(config_path))
        model=load_model(ROOT/config['checkpoint'])
        curve=forecast({name:model},X0,Y0,steps=1000)[name]
        metrics=json.loads((ROOT/config['metrics']).read_text())
        row=dict(name=name,config=config,selection_score=score(curve),
                 selected_epoch=metrics['best_epoch'],selection_rmse_t1=float(curve[-1]))
        rows.append(row)
        np.save(output/f'{name}_selection.npy',curve)
        (output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(f"{name}: N={config['horizon']}, lr={config['lr']}, seed={config['seed']}, score={row['selection_score']:.6f}",flush=True)
    # Choose hyperparameters by the grid, keep its seed-42 checkpoint as primary.
    winner=min(rows[:8],key=lambda row:row['selection_score'])
    repeated=[winner,*rows[8:]]
    models={'control':load_model(ROOT/'results/breadth/control.pt'),
            'breadth_rollout':load_model(ROOT/'results/breadth/rollout.pt'),
            'previous_supervised':load_model(ROOT/'hybrid/neural.pt')}
    models.update({row['name']:load_model(ROOT/row['config']['checkpoint']) for row in repeated})
    rng=np.random.default_rng(46)
    X=torch.from_numpy(rng.normal(0,5,(20,full.K)))
    Y=torch.from_numpy(rng.normal(0,.5,(20,full.K*full.J)))
    with torch.no_grad():
        for _ in range(5000):
            X,Y=full.rk4_step(X,Y,.001)
    curves=forecast(models,X,Y)
    final={name:dict(mean_rmse_0_to_1=score(curve),
                     rmse={str(t):float(curve[round(t/.001)]) for t in (.1,.5,1.,2.)})
           for name,curve in curves.items()}
    summary=dict(training_runs=10,selection_seed=45,final_seed=46,
                 metric='mean RMSE for t=(0,1]',winner=winner['name'],runs=rows,final=final,
                 protocol=dict(batch=20,burn_in=5000,steps=2000,dt=.001,K=8,J=32,F=20,h=1,c=10,b=10))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(output/'final_curves.npz',time=np.arange(2001)*.001,X0=X.numpy(),Y0=Y.numpy(),**curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for name,curve in curves.items():
        ax.plot(np.arange(2001)*.001,curve,label=name)
    ax.set(xlabel='Forecast lead time',ylabel='RMSE of X',title='Depth search: fresh evaluation seed 46')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'forecast_rmse.png',dpi=160);plt.close(fig)
    lines=['\n## 2026-11-18 — Задание 4: поиск вглубь, лимит 10 запусков\n',
           '- Команда: `MPLCONFIGDIR=/tmp/l96-mpl .venv/bin/python -m hybrid.depth`. Ровно 10 запусков обучения.\n',
           '- Начальные веса: быстрый контроль results/breadth/control.pt, архитектура 1→32→32→1 Tanh. Данные data.npz, первые 80% для обучения, последние 20% для выбора эпохи, окна не пересекают раздел; нормализация сохранена из контроля. Adam, float64, CPU, 1 поток, deterministic, 10 эпох, batch=32 окон, stride=400, clip=1.\n',
           '- Сетка: N=50/100/200/400 × lr=0.0001/0.001, seed=42. Затем два повтора лучшего сочетания с seed=43/44. Выбор эпохи по rollout MSE, включая исходные веса.\n',
           '- Выбор сочетания по среднему RMSE прогноза за t=(0,1] на прежних 20 состояниях seed=45, dt=0.001. Итоговый checkpoint — победитель сетки с seed=42, без отбора лучшего seed по тесту.\n',
           '\n| Запуск | N | lr | Seed | Эпоха | Средний RMSE выбора |\n|---|---:|---:|---:|---:|---:|\n']
    for row in rows:
        c=row['config'];lines.append(f"| {row['name']} | {c['horizon']} | {c['lr']} | {c['seed']} | {row['selected_epoch']} | {row['selection_score']:.6f} |\n")
    lines.append(f"\n- Победитель сетки: {winner['name']}. Все результаты, включая ухудшения, сохранены.\n")
    lines.append('- Финальная проверка: новые 20 начальных состояний seed=46, разгон 5000, прогноз 2000 шагов. Полная модель — эталон; стандартные параметры. Эти данные не использовались для выбора.\n\n| Вариант | Средний RMSE (0,1] | t=0.5 | t=1 | t=2 |\n|---|---:|---:|---:|---:|\n')
    for name,m in final.items():
        f=m['rmse'];lines.append(f"| {name} | {m['mean_rmse_0_to_1']:.6f} | {f['0.5']:.6f} | {f['1.0']:.6f} | {f['2.0']:.6f} |\n")
    lines.append('- Конфиги совместимы с hybrid.experiment: hybrid/configs/depth/*.yaml. Веса, истории, stdout, метрики и график: results/depth/. Проверка на одном новом ансамбле предварительная; повторы меняют порядок обучения, но не обучающую траекторию.\n')
    with (ROOT/'results/log.md').open('a') as stream:stream.writelines(lines)
    print(json.dumps(dict(winner=winner['name'],final=final),indent=2),flush=True)


if __name__=='__main__':
    main()
