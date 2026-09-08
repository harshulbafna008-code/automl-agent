"""Model-agnostic hyperparameter search using scikit-learn's
RandomizedSearchCV. Works for any candidate (sklearn or torch-wrapped
estimator) since the estimator only needs fit/predict/score.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from ..pipeline.builder import build_pipeline
from ..pipeline.search_space import param_space_to_distributions

logger = logging.getLogger("automl_agent.tuning")


@dataclass
class CandidateResult:
    name: str
    model: str
    best_score: float
    best_params: Dict[str, Any]
    scoring: str
    fit_seconds: float
    error: Optional[str] = None
    cv_results_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "best_score": None if self.best_score is None else round(float(self.best_score), 5),
            "best_params": self.best_params,
            "scoring": self.scoring,
            "fit_seconds": round(self.fit_seconds, 2),
            "error": self.error,
        }


def _default_scoring(task: str) -> str:
    return "accuracy" if task == "classification" else "neg_root_mean_squared_error"


def evaluate_candidate(
    x: pd.DataFrame,
    y: pd.Series,
    task: str,
    preprocessing_spec: Dict[str, Any],
    candidate: Dict[str, Any],
    cv_folds: int = 5,
    n_iter: int = 15,
    random_state: int = 42,
    scoring: Optional[str] = None,
) -> CandidateResult:
    """Build the pipeline for one candidate and run RandomizedSearchCV.

    Returns a CandidateResult even on failure (with `.error` populated) so
    the orchestrator can keep going and report partial results.
    """
    scoring = scoring or _default_scoring(task)
    name = candidate.get("name", candidate["model"])
    model_key = candidate["model"]
    param_space = candidate.get("param_space", {})

    t0 = time.time()
    try:
        pipeline: Pipeline = build_pipeline(x, preprocessing_spec, model_key)
        distributions = param_space_to_distributions(param_space)

        if distributions:
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=distributions,
                n_iter=min(n_iter, _search_space_size(distributions)),
                cv=cv_folds,
                scoring=scoring,
                random_state=random_state,
                n_jobs=-1,
                error_score=np.nan,
            )
            search.fit(x, y)
            best_score = search.best_score_
            best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
            cv_summary = {
                "mean_test_scores": [round(float(s), 5) for s in search.cv_results_["mean_test_score"]],
            }
        else:
            # No tunable params proposed -> just cross-validate the default estimator.
            from sklearn.model_selection import cross_val_score

            scores = cross_val_score(pipeline, x, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
            best_score = float(np.mean(scores))
            best_params = {}
            cv_summary = {"mean_test_scores": [round(float(s), 5) for s in scores]}

        return CandidateResult(
            name=name,
            model=model_key,
            best_score=best_score,
            best_params=best_params,
            scoring=scoring,
            fit_seconds=time.time() - t0,
            cv_results_summary=cv_summary,
        )
    except Exception as exc:  # noqa: BLE001 - we want to capture *any* estimator failure
        logger.exception("Candidate '%s' (%s) failed", name, model_key)
        return CandidateResult(
            name=name,
            model=model_key,
            best_score=float("-inf"),
            best_params={},
            scoring=scoring,
            fit_seconds=time.time() - t0,
            error=str(exc),
        )


def _search_space_size(distributions: Dict[str, Any]) -> int:
    """Rough cap so n_iter never wildly exceeds a purely-categorical space."""
    size = 1
    for dist in distributions.values():
        if isinstance(dist, list):
            size *= max(len(dist), 1)
            if size > 1000:
                return 1000
    return max(size, 15) if size != 1 else 15
