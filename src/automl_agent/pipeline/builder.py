from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from ..data.preprocessing import build_preprocessor
from ..models.sklearn_models import get_estimator


def build_pipeline(x: pd.DataFrame, preprocessing_spec: dict, model_key: str) -> Pipeline:
    """Assemble a full Pipeline: [preprocessor] -> [model].

    The 'model' step name matters — `search_space.param_space_to_distributions`
    prefixes hyperparameters with "model__" to target this exact step.
    """
    preprocessor = build_preprocessor(x, preprocessing_spec)
    estimator = get_estimator(model_key)
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])
