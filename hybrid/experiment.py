"""One-command experiments: python -m hybrid.experiment hybrid/configs/supervised.yaml."""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path) as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get('mode') not in ('supervised', 'rollout'):
        raise ValueError('mode must be supervised or rollout')
    common = {'mode', 'hidden_sizes', 'seed', 'epochs', 'batch_size', 'lr',
              'train_fraction', 'threads', 'data', 'checkpoint', 'metrics'}
    required = common | ({'horizon', 'stride', 'grad_clip', 'initialization'}
                         if config['mode'] == 'rollout' else set())
    if set(config) != required:
        raise ValueError(f'Missing keys: {required-set(config)}; unknown keys: {set(config)-required}')
    for key in ('epochs', 'batch_size', 'threads', 'horizon', 'stride'):
        if key in config and (type(config[key]) is not int or config[key] <= 0):
            raise ValueError(f'{key} must be a positive integer')
    if type(config['seed']) is not int or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    sizes = config['hidden_sizes']
    if not isinstance(sizes, list) or not sizes or any(type(n) is not int or n <= 0 for n in sizes):
        raise ValueError('hidden_sizes must contain positive integers')
    for key in ('lr', 'grad_clip'):
        if key in config and (not isinstance(config[key], (int, float)) or not 0 < config[key] < float('inf')):
            raise ValueError(f'{key} must be positive and finite')
    if not isinstance(config['train_fraction'], (int, float)) or not 0 < config['train_fraction'] < 1:
        raise ValueError('train_fraction must be between 0 and 1')
    for key in ('data', 'checkpoint', 'metrics', 'initialization'):
        if key in config and (not isinstance(config[key], str) or not config[key]):
            raise ValueError(f'{key} must be a path')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args = parser.parse_args()
    config = load_config(args.config)
    if config['mode'] == 'supervised':
        from hybrid.train_neural import main as train
    else:
        from hybrid.train_rollout import main as train
    train(config)


if __name__ == '__main__':
    main()
