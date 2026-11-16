"""
Записывает ОРАКУЛЫ: пары "вход -> выход" эталонной модели для каждой функции.

Это ровно тот приём из доклада про WRF: запускаем старый код как есть,
сохраняем, что он выдаёт, и потом новый код обязан выдать то же самое.

Запуск (из корня проекта):
    python oracles/make_oracles.py
Создаёт файл oracles/oracles.npz. Перезапускать нужно только если поменялся
эталон (а он меняться не должен).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference import l96_numpy as ref  # noqa: E402

N_CASES = 20      # сколько случайных входов на каждую функцию
DT = 0.001        # шаг по времени (мелкие переменные быстрые, шаг маленький)
N_STEPS = 200     # длина короткой траектории для теста integrate


def main():
    rng = np.random.default_rng(42)  # фиксированный seed -> воспроизводимость
    X_in, Y_in = [], []
    for _ in range(N_CASES):
        X, Y = ref.random_state(rng)
        X_in.append(X)
        Y_in.append(Y)
    X_in, Y_in = np.array(X_in), np.array(Y_in)

    out = {
        "X_in": X_in,
        "Y_in": Y_in,
        "dt": np.array(DT),
        "coupling": np.array([ref.coupling(Y) for Y in Y_in]),
        "rhs_x": np.array([ref.rhs_x(X, Y) for X, Y in zip(X_in, Y_in)]),
        "rhs_y": np.array([ref.rhs_y(X, Y) for X, Y in zip(X_in, Y_in)]),
    }
    steps = [ref.rk4_step(X, Y, DT) for X, Y in zip(X_in, Y_in)]
    out["rk4_X"] = np.array([s[0] for s in steps])
    out["rk4_Y"] = np.array([s[1] for s in steps])

    traj_X, traj_Y = ref.integrate(X_in[0], Y_in[0], DT, N_STEPS)
    out["traj_X"], out["traj_Y"] = traj_X, traj_Y
    out["n_steps"] = np.array(N_STEPS)

    path = ROOT / "oracles" / "oracles.npz"
    np.savez(path, **out)
    print(f"Оракулы записаны в {path.relative_to(ROOT)}: "
          f"{N_CASES} случаев на функцию + траектория из {N_STEPS} шагов")


if __name__ == "__main__":
    main()
