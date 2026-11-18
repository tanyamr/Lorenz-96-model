"""Pointwise neural correction U(X_k), preserving arbitrary input shapes."""
import torch
from torch import nn


class NeuralCorrection(nn.Module):
    def __init__(self, hidden_sizes=(32, 32), activation="tanh", offsets=(0,)):
        super().__init__()
        self.activation = activation
        self.offsets = tuple(offsets)
        self.architecture = [len(self.offsets), *hidden_sizes, 1]
        activation_cls = {"tanh": nn.Tanh, "silu": nn.SiLU}[activation]
        layers = []
        for inputs, outputs in zip(self.architecture[:-2], self.architecture[1:-1]):
            layers.extend([nn.Linear(inputs, outputs), activation_cls()])
        layers.append(nn.Linear(self.architecture[-2], 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer('x_mean', torch.tensor(0.0))
        self.register_buffer('x_std', torch.tensor(1.0))
        self.register_buffer('y_mean', torch.tensor(0.0))
        self.register_buffer('y_std', torch.tensor(1.0))
        self.double()

    def forward(self, X):
        normalized = (X - self.x_mean) / self.x_std
        z = torch.stack([torch.roll(normalized, -offset, dims=-1)
                         for offset in self.offsets], dim=-1) if self.offsets != (0,) else normalized.unsqueeze(-1)
        return self.net(z).squeeze(-1) * self.y_std + self.y_mean


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = NeuralCorrection(checkpoint.get("architecture", [1, 32, 32, 1])[1:-1],
                             checkpoint.get("activation", "tanh"), checkpoint.get("offsets", [0]))
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()
