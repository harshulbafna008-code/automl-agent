"""Optional small MLP models backed by PyTorch, exposed with a
scikit-learn-compatible `fit`/`predict`/`predict_proba` API so they slot
into the same Pipeline/cross-validation machinery as the sklearn models.

This module is imported lazily and is allowed to fail with ImportError if
`torch` isn't installed — PyTorch is an optional extra
(`pip install torch`), not a hard dependency of the base package.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2):
        super().__init__()
        layers = []
        prev = in_dim
        for _ in range(max(n_layers, 1)):
            layers += [nn.Linear(prev, hidden_dim), nn.ReLU()]
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _TorchMLPBase(BaseEstimator):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _make_loader(self, x: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.random_state)
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32 if self._is_regression else torch.long)
        ds = torch.utils.data.TensorDataset(x_t, y_t)
        return torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)


class TorchMLPClassifier(_TorchMLPBase, ClassifierMixin):
    _is_regression = False

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        self.model_ = _MLP(x.shape[1], self.hidden_dim, n_classes, self.n_layers)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        loader = self._make_loader(x, y_idx)
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, x):
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.tensor(np.asarray(x, dtype=np.float32)))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, x):
        proba = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]


class TorchMLPRegressor(_TorchMLPBase, RegressorMixin):
    _is_regression = True

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.model_ = _MLP(x.shape[1], self.hidden_dim, 1, self.n_layers)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        loader = self._make_loader(x, y)
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                pred = self.model_(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, x):
        self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.tensor(np.asarray(x, dtype=np.float32))).squeeze(-1).numpy()


def register_torch_models(registry: Dict) -> None:
    registry["torch_mlp_classifier"] = ("classification", TorchMLPClassifier, {})
    registry["torch_mlp_regressor"] = ("regression", TorchMLPRegressor, {})
