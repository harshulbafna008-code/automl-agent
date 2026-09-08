"""Turn a raw DataFrame into a compact, LLM-friendly profile.

The profile deliberately avoids sending raw data to the LLM (privacy +
token budget) — only shapes, types, and summary statistics.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

MAX_CATEGORIES_LISTED = 10


def _infer_task(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        return "regression"
    return "classification"


def profile_dataset(df: pd.DataFrame, target: str, task: str = "auto") -> Dict[str, Any]:
    """Compute a structured profile of `df` with respect to `target`.

    Returns a JSON-serializable dict: shape, per-column stats, target
    distribution, missingness, and an inferred (or user-specified) task type.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset columns: {list(df.columns)}")

    y = df[target]
    x = df.drop(columns=[target])

    if task == "auto":
        task = _infer_task(y)

    columns: Dict[str, Any] = {}
    for col in x.columns:
        series = x[col]
        n_missing = int(series.isna().sum())
        col_info: Dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing_count": n_missing,
            "missing_pct": round(100 * n_missing / max(len(series), 1), 2),
            "n_unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            col_info.update(
                {
                    "kind": "numeric",
                    "mean": _safe_float(desc.get("mean")),
                    "std": _safe_float(desc.get("std")),
                    "min": _safe_float(desc.get("min")),
                    "max": _safe_float(desc.get("max")),
                }
            )
        else:
            top_values = series.value_counts(dropna=True).head(MAX_CATEGORIES_LISTED)
            col_info.update(
                {
                    "kind": "categorical",
                    "top_values": {str(k): int(v) for k, v in top_values.items()},
                    "high_cardinality": bool(series.nunique(dropna=True) > 50),
                }
            )
        columns[col] = col_info

    if task == "classification":
        vc = y.value_counts(dropna=False)
        target_info = {
            "n_classes": int(vc.shape[0]),
            "class_distribution": {str(k): int(v) for k, v in vc.items()},
            "is_imbalanced": bool((vc.min() / vc.max()) < 0.3) if vc.shape[0] > 1 else False,
        }
    else:
        desc = y.describe()
        target_info = {
            "mean": _safe_float(desc.get("mean")),
            "std": _safe_float(desc.get("std")),
            "min": _safe_float(desc.get("min")),
            "max": _safe_float(desc.get("max")),
        }

    return {
        "task": task,
        "n_rows": int(df.shape[0]),
        "n_features": int(x.shape[1]),
        "target_column": target,
        "target_info": target_info,
        "columns": columns,
        "total_missing_pct": round(100 * float(x.isna().sum().sum()) / max(x.size, 1), 2),
    }


def _safe_float(v) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)
