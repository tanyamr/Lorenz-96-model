"""Exercise both YAML entrypoint modes with a non-default network."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid.experiment import ROOT, load_config
from hybrid.neural import load_model


def test_yaml_training_and_rollout(tmp_path):
    import yaml
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 8))
    data = tmp_path / 'data.npz'
    np.savez(data, X=X, coupling=-X, dt=0.001)
    supervised = load_config(ROOT / 'hybrid/configs/supervised.yaml')
    supervised.update(hidden_sizes=[4, 6], epochs=1, batch_size=32,
                      data=str(data), checkpoint=str(tmp_path / 'supervised.pt'),
                      metrics=str(tmp_path / 'supervised.json'))
    rollout = load_config(ROOT / 'hybrid/configs/rollout.yaml')
    rollout.update(hidden_sizes=[4, 6], epochs=1, batch_size=4, horizon=2, stride=2,
                   data=str(data), initialization=supervised['checkpoint'],
                   checkpoint=str(tmp_path / 'rollout.pt'), metrics=str(tmp_path / 'rollout.json'))
    for config in (supervised, rollout):
        path = tmp_path / (config['mode'] + '.yaml')
        path.write_text(yaml.safe_dump(config))
        result = subprocess.run([sys.executable, '-m', 'hybrid.experiment', str(path)],
                                cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(Path(config['metrics']).read_text())['config'] == config
        model = load_model(config['checkpoint'])
        assert model.architecture == [1, 4, 6, 1]
        assert torch.isfinite(model(torch.tensor(X))).all()
    bad = dict(rollout, horizon=0)
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match='horizon'):
        load_config(path)
