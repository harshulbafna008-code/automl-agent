"""Validate LLM-produced JSON plans and convert param spaces into
scikit-learn-compatible distributions for RandomizedSearchCV.

This is the safety boundary between "text an LLM produced" and "code that
actually runs." Nothing here ever calls eval/exec on LLM output — every
field is checked against a fixed vocabulary or numeric type before use.
"""

from __future__ import annotations

from typing import Any, Dict, List

from scipy.stats import loguniform, randint, uniform

from ..models.sklearn_models import model_whitelist_for_task

_VALID_NUMERIC_IMPUTERS = {"mean", "median", "most_frequent"}
_VALID_SCALERS = {"standard", "minmax", "robust", "none"}
_VALID_CAT_IMPUTERS = {"most_frequent", "constant"}
_VALID_ENCODERS = {"onehot", "ordinal"}

DEFAULT_PREPROCESSING = {
    "numeric_imputer": "median",
    "numeric_scaler": "standard",
    "categorical_imputer": "most_frequent",
    "categorical_encoder": "onehot",
    "handle_class_imbalance": False,
}


class PlanValidationError(ValueError):
    pass


def default_plan(task: str) -> Dict[str, Any]:
    """A safe, always-valid fallback plan used when the LLM output can't be
    trusted (unreachable, malformed JSON, or fails validation)."""
    whitelist = model_whitelist_for_task(task)
    candidates = []
    # Pick up to 3 sensible defaults if present in the whitelist.
    preferred_order = [
        "random_forest_classifier",
        "gradient_boosting_classifier",
        "logistic_regression",
        "random_forest_regressor",
        "gradient_boosting_regressor",
        "ridge_regression",
    ]
    chosen = [m for m in preferred_order if m in whitelist][:3] or whitelist[:3]
    for model in chosen:
        candidates.append({"name": model.replace("_", " ").title(), "model": model, "param_space": {}})
    return {
        "reasoning": "Fallback default plan (LLM plan unavailable or invalid).",
        "preprocessing": dict(DEFAULT_PREPROCESSING),
        "candidates": candidates,
    }


def validate_plan(plan: Dict[str, Any], task: str) -> Dict[str, Any]:
    """Validate and sanitize a plan dict. Raises PlanValidationError on
    unrecoverable problems; silently drops/clamps minor issues (e.g. an
    unknown enum value falls back to the default for that field).
    """
    if not isinstance(plan, dict):
        raise PlanValidationError("Plan is not a JSON object.")

    whitelist = set(model_whitelist_for_task(task))
    if not whitelist:
        raise PlanValidationError(f"No registered models support task '{task}'.")

    raw_prep = plan.get("preprocessing", {}) or {}
    if not isinstance(raw_prep, dict):
        raw_prep = {}
    preprocessing = {
        "numeric_imputer": _clamp(raw_prep.get("numeric_imputer"), _VALID_NUMERIC_IMPUTERS, "median"),
        "numeric_scaler": _clamp(raw_prep.get("numeric_scaler"), _VALID_SCALERS, "standard"),
        "categorical_imputer": _clamp(raw_prep.get("categorical_imputer"), _VALID_CAT_IMPUTERS, "most_frequent"),
        "categorical_encoder": _clamp(raw_prep.get("categorical_encoder"), _VALID_ENCODERS, "onehot"),
        "handle_class_imbalance": bool(raw_prep.get("handle_class_imbalance", False)),
    }

    raw_candidates = plan.get("candidates", [])
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise PlanValidationError("Plan has no candidates list.")

    candidates: List[Dict[str, Any]] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        model = raw.get("model")
        if model not in whitelist:
            continue  # silently drop candidates the LLM hallucinated
        param_space = raw.get("param_space", {})
        if not isinstance(param_space, dict):
            param_space = {}
        clean_space = {}
        for pname, pspec in param_space.items():
            cleaned = _clean_param_spec(pspec)
            if cleaned is not None:
                clean_space[pname] = cleaned
        candidates.append(
            {
                "name": str(raw.get("name", model)),
                "model": model,
                "param_space": clean_space,
            }
        )

    if not candidates:
        raise PlanValidationError("No candidates survived whitelist validation.")

    return {
        "reasoning": str(plan.get("reasoning", "")),
        "preprocessing": preprocessing,
        "candidates": candidates,
    }


def _clamp(value: Any, valid: set, default: str) -> str:
    return value if isinstance(value, str) and value in valid else default


def _clean_param_spec(spec: Any):
    if not isinstance(spec, dict):
        return None
    ptype = spec.get("type")
    if ptype == "categorical":
        choices = spec.get("choices")
        if isinstance(choices, list) and choices:
            return {"type": "categorical", "choices": choices}
        return None
    if ptype in ("float", "int"):
        low, high = spec.get("low"), spec.get("high")
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)) or low >= high:
            return None
        return {"type": ptype, "low": low, "high": high, "log": bool(spec.get("log", False))}
    return None


def param_space_to_distributions(param_space: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a validated param_space dict into scipy/sklearn distributions
    suitable for `RandomizedSearchCV(param_distributions=...)`.
    """
    distributions: Dict[str, Any] = {}
    for name, spec in param_space.items():
        # Prefix with "model__" to target the estimator step inside our Pipeline.
        key = f"model__{name}"
        if spec["type"] == "categorical":
            distributions[key] = spec["choices"]
        elif spec["type"] == "int":
            distributions[key] = randint(int(spec["low"]), int(spec["high"]) + 1)
        elif spec["type"] == "float":
            if spec.get("log"):
                distributions[key] = loguniform(spec["low"], spec["high"])
            else:
                distributions[key] = uniform(spec["low"], spec["high"] - spec["low"])
    return distributions
