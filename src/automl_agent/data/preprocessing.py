"""Build a scikit-learn ColumnTransformer from a validated preprocessing spec.

The LLM only ever picks from the small whitelisted vocab below (see
`pipeline/search_space.py` for validation) — it never supplies code.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "none": None,
}

_ENCODERS = {
    "onehot": lambda: OneHotEncoder(handle_unknown="ignore"),
    "ordinal": lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
}


def split_column_types(x: pd.DataFrame) -> tuple[List[str], List[str]]:
    numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in x.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(x: pd.DataFrame, spec: Dict) -> ColumnTransformer:
    """spec is the validated `preprocessing` dict from the LLM plan, e.g.:

    {
      "numeric_imputer": "median",
      "numeric_scaler": "standard",
      "categorical_imputer": "most_frequent",
      "categorical_encoder": "onehot",
      "handle_class_imbalance": false
    }
    """
    numeric_cols, categorical_cols = split_column_types(x)
    transformers = []

    if numeric_cols:
        steps = [("imputer", SimpleImputer(strategy=spec.get("numeric_imputer", "median")))]
        scaler_key = spec.get("numeric_scaler", "standard")
        scaler_cls = _SCALERS.get(scaler_key)
        if scaler_cls is not None:
            steps.append(("scaler", scaler_cls()))
        transformers.append(("num", Pipeline(steps), numeric_cols))

    if categorical_cols:
        cat_imputer_strategy = spec.get("categorical_imputer", "most_frequent")
        imputer_kwargs = {"strategy": cat_imputer_strategy}
        if cat_imputer_strategy == "constant":
            imputer_kwargs["fill_value"] = "missing"
        encoder_key = spec.get("categorical_encoder", "onehot")
        encoder_factory = _ENCODERS.get(encoder_key, _ENCODERS["onehot"])
        steps = [
            ("imputer", SimpleImputer(**imputer_kwargs)),
            ("encoder", encoder_factory()),
        ]
        transformers.append(("cat", Pipeline(steps), categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns found to build a preprocessor from.")

    return ColumnTransformer(transformers, remainder="drop")
