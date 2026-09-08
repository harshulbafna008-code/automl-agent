"""The whitelist of models the LLM planner is allowed to choose from.

This is the single safety boundary that keeps the agent from ever needing
to `eval()` or otherwise execute LLM-authored code: the LLM only ever
selects a *key* from this registry, never a class path or code string.

Each entry: (task, constructor_callable, default_param_overrides).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

RegistryEntry = Tuple[str, Callable, Dict]

MODEL_REGISTRY: Dict[str, RegistryEntry] = {
    # ---- classification ----
    "logistic_regression": (
        "classification",
        LogisticRegression,
        {"max_iter": 2000},
    ),
    "random_forest_classifier": (
        "classification",
        RandomForestClassifier,
        {"random_state": 42, "n_jobs": -1},
    ),
    "gradient_boosting_classifier": (
        "classification",
        GradientBoostingClassifier,
        {"random_state": 42},
    ),
    "decision_tree_classifier": (
        "classification",
        DecisionTreeClassifier,
        {"random_state": 42},
    ),
    "knn_classifier": (
        "classification",
        KNeighborsClassifier,
        {},
    ),
    "svc": (
        "classification",
        SVC,
        {"probability": True},
    ),
    # ---- regression ----
    "ridge_regression": (
        "regression",
        Ridge,
        {},
    ),
    "random_forest_regressor": (
        "regression",
        RandomForestRegressor,
        {"random_state": 42, "n_jobs": -1},
    ),
    "gradient_boosting_regressor": (
        "regression",
        GradientBoostingRegressor,
        {"random_state": 42},
    ),
    "decision_tree_regressor": (
        "regression",
        DecisionTreeRegressor,
        {"random_state": 42},
    ),
    "knn_regressor": (
        "regression",
        KNeighborsRegressor,
        {},
    ),
    "svr": (
        "regression",
        SVR,
        {},
    ),
}

# Optional PyTorch models are registered lazily so `torch` stays an optional
# dependency (see models/torch_models.py). If torch isn't installed, this
# is a silent no-op and those keys simply won't appear in the whitelist.
try:
    from .torch_models import register_torch_models

    register_torch_models(MODEL_REGISTRY)
except ImportError:
    pass


def model_whitelist_for_task(task: str) -> List[str]:
    return [name for name, (t, _, _) in MODEL_REGISTRY.items() if t == task]


def get_estimator(name: str):
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model key '{name}'. Must be one of: {sorted(MODEL_REGISTRY.keys())}"
        )
    _, ctor, defaults = MODEL_REGISTRY[name]
    return ctor(**defaults)
