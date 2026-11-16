"""Run from the repository root: .venv/bin/python experiments/benchmark_port.py."""
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import l96_numpy as ref
from port import l96_torch as port


def main():
    seed, batch, repeats, dt = 42, 1000, 5, 0.001
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 5, (batch, ref.K))
    Y = rng.normal(0, 0.5, (batch, ref.K * ref.J))
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)

    def numpy_loop():
        pairs = [ref.rk4_step(x, y, dt) for x, y in zip(X, Y)]
        return tuple(np.stack(values) for values in zip(*pairs))

    def torch_batch():
        return port.rk4_step(Xt, Yt, dt)

    with torch.no_grad():
        expected, actual = numpy_loop(), torch_batch()  # warm up both paths
        errors = []
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got.numpy(), want, rtol=1e-10, atol=1e-10)
            errors.append(float(np.max(np.abs(got.numpy() - want))))
        timings = []
        for run in (numpy_loop, torch_batch):
            samples = []
            for _ in range(repeats):
                start = perf_counter()
                run()
                samples.append(perf_counter() - start)
            timings.append(median(samples))
    print(f'CPU: {platform.machine()}, {platform.platform()}')
    print(f'Python {platform.python_version()}, NumPy {np.__version__}, PyTorch {torch.__version__}')
    print(f'seed={seed}, batch={batch}, K={ref.K}, J={ref.J}, dt={dt}, steps=1, float64, threads=1')
    print(f'Median of {repeats} runs after warmup; tensor conversion excluded; no_grad')
    print(f'NumPy loop: {timings[0]:.6f} s; PyTorch batch: {timings[1]:.6f} s')
    print(f'Speedup: {timings[0] / timings[1]:.2f}x; max errors X/Y: {errors}')


if __name__ == '__main__':
    main()
