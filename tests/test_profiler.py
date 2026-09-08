import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from automl_agent.data.profiler import profile_dataset  # noqa: E402
from automl_agent.pipeline.search_space import (  # noqa: E402
    PlanValidationError,
    default_plan,
    validate_plan,
)


@pytest.fixture
def toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [22, 35, None, 40, 29],
            "city": ["NYC", "LA", "NYC", None, "SF"],
            "target": [0, 1, 0, 1, 1],
        }
    )


def test_profile_dataset_basic(toy_df):
    profile = profile_dataset(toy_df, target="target")
    assert profile["task"] == "classification"
    assert profile["n_rows"] == 5
    assert profile["n_features"] == 2
    assert "age" in profile["columns"]
    assert profile["columns"]["age"]["kind"] == "numeric"
    assert profile["columns"]["city"]["kind"] == "categorical"
    assert profile["columns"]["age"]["missing_count"] == 1


def test_profile_missing_target_raises(toy_df):
    with pytest.raises(ValueError):
        profile_dataset(toy_df, target="does_not_exist")


def test_default_plan_is_valid_for_classification():
    plan = default_plan("classification")
    validated = validate_plan(plan, "classification")
    assert len(validated["candidates"]) > 0
    for c in validated["candidates"]:
        assert c["model"]


def test_validate_plan_drops_unknown_model():
    plan = {
        "preprocessing": {},
        "candidates": [
            {"name": "bogus", "model": "definitely_not_a_real_model", "param_space": {}},
            {"name": "rf", "model": "random_forest_classifier", "param_space": {}},
        ],
    }
    validated = validate_plan(plan, "classification")
    models = [c["model"] for c in validated["candidates"]]
    assert "definitely_not_a_real_model" not in models
    assert "random_forest_classifier" in models


def test_validate_plan_raises_when_all_candidates_invalid():
    plan = {"preprocessing": {}, "candidates": [{"name": "bogus", "model": "nope", "param_space": {}}]}
    with pytest.raises(PlanValidationError):
        validate_plan(plan, "classification")


def test_validate_plan_clamps_bad_param_spec():
    plan = {
        "preprocessing": {"numeric_scaler": "not_a_real_scaler"},
        "candidates": [
            {
                "name": "rf",
                "model": "random_forest_classifier",
                "param_space": {
                    "n_estimators": {"type": "int", "low": 50, "high": 300},
                    "bad_param": {"type": "int", "low": 10, "high": 1},  # low >= high, invalid
                },
            }
        ],
    }
    validated = validate_plan(plan, "classification")
    assert validated["preprocessing"]["numeric_scaler"] == "standard"  # fell back to default
    space = validated["candidates"][0]["param_space"]
    assert "n_estimators" in space
    assert "bad_param" not in space
