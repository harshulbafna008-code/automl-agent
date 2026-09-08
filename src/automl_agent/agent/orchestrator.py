"""The AutoML-Agent orchestrator: plan -> execute -> reflect -> (re-plan).

This is the piece that ties everything together:
  1. Profile the dataset (data/profiler.py)
  2. Ask the LLM to propose a plan (llm/prompts.py + llm/base.py)
  3. Validate the plan against a fixed whitelist (pipeline/search_space.py)
  4. Execute every candidate with cross-validated hyperparameter search
     (tuning/optimizer.py)
  5. Ask the LLM to reflect on the leaderboard and decide stop/continue
  6. Repeat up to `max_rounds` or until the time budget is exhausted
  7. Persist the best pipeline + a human-readable Markdown report

The LLM is *never* on the critical path for correctness: every step that
touches the LLM has a validated/whitelisted fallback, so a broken or
unreachable local model degrades the agent to "sensible default AutoML"
rather than crashing it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from ..config import RunConfig
from ..data.profiler import profile_dataset
from ..llm.base import LLMClient
from ..llm.prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_TEMPLATE,
    REFLECTOR_SYSTEM_PROMPT,
    REFLECTOR_USER_TEMPLATE,
)
from ..models.sklearn_models import model_whitelist_for_task
from ..pipeline.builder import build_pipeline
from ..pipeline.search_space import PlanValidationError, default_plan, validate_plan
from ..tuning.optimizer import CandidateResult, evaluate_candidate
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RunState:
    round_num: int = 0
    leaderboard: List[CandidateResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    plans_tried: List[Dict[str, Any]] = field(default_factory=list)
    llm_reasoning_log: List[str] = field(default_factory=list)


class AutoMLAgent:
    def __init__(self, config: RunConfig, llm: LLMClient):
        self.config = config
        self.llm = llm
        self.state = RunState()

  
    # Public entry point
    def run(self) -> Dict[str, Any]:
        start_time = time.time()
        out_dir = self.config.resolved_out_dir()

        df = pd.read_csv(self.config.data_path)
        profile = profile_dataset(df, self.config.target, self.config.task)
        task = profile["task"]
        logger.info("Loaded dataset: %d rows, %d features. Task: %s", profile["n_rows"], profile["n_features"], task)

        x = df.drop(columns=[self.config.target])
        y = df[self.config.target]

        llm_ready = self.llm.is_available()
        if not llm_ready:
            logger.warning("LLM backend unavailable — proceeding with default plans only.")

        plan = self._get_initial_plan(profile, task, llm_ready)

        while True:
            self.state.round_num += 1
            elapsed = time.time() - start_time
            remaining = self.config.time_budget_seconds - elapsed
            logger.info(
                "=== Round %d/%d (elapsed %.0fs, remaining %.0fs) ===",
                self.state.round_num,
                self.config.max_rounds,
                elapsed,
                remaining,
            )
            self.state.plans_tried.append(plan)
            self._execute_round(x, y, task, plan)

            if self.state.round_num >= self.config.max_rounds or remaining <= 0:
                logger.info("Stopping: reached round/time limit.")
                break

            decision = self._reflect(task, remaining, llm_ready)
            if decision["decision"] == "stop":
                logger.info("LLM decided to stop: %s", decision.get("reasoning", ""))
                break
            plan = decision["next_plan"]

        return self._finalize(x, y, task, profile, out_dir, time.time() - start_time)

   
    # Planning
    def _get_initial_plan(self, profile: Dict[str, Any], task: str, llm_ready: bool) -> Dict[str, Any]:
        if not llm_ready:
            return default_plan(task)

        whitelist = model_whitelist_for_task(task)
        system_prompt = PLANNER_SYSTEM_PROMPT.format(model_whitelist=", ".join(whitelist))
        user_prompt = PLANNER_USER_TEMPLATE.format(
            profile_json=json.dumps(profile, indent=2), task=task
        )
        raw = self._safe_llm_call(system_prompt, user_prompt)
        plan = self._parse_and_validate(raw, task)
        if plan.get("reasoning"):
            self.state.llm_reasoning_log.append(f"[Planner] {plan['reasoning']}")
        return plan

    def _reflect(self, task: str, remaining: float, llm_ready: bool) -> Dict[str, Any]:
        if not llm_ready:
            return {"decision": "stop", "reasoning": "LLM unavailable.", "next_plan": None}

        whitelist = model_whitelist_for_task(task)
        system_prompt = REFLECTOR_SYSTEM_PROMPT.format(model_whitelist=", ".join(whitelist))
        leaderboard_sorted = sorted(self.state.leaderboard, key=lambda r: r.best_score, reverse=True)
        user_prompt = REFLECTOR_USER_TEMPLATE.format(
            round_num=self.state.round_num,
            max_rounds=self.config.max_rounds,
            time_remaining=remaining,
            leaderboard_json=json.dumps([r.to_dict() for r in leaderboard_sorted[:10]], indent=2),
            errors_json=json.dumps(self.state.errors[-5:], indent=2),
        )
        raw = self._safe_llm_call(system_prompt, user_prompt)
        parsed = self._safe_json_parse(raw)
        if not parsed or parsed.get("decision") not in ("stop", "continue"):
            return {"decision": "stop", "reasoning": "Could not parse reflector output.", "next_plan": None}

        if parsed.get("reasoning"):
            self.state.llm_reasoning_log.append(f"[Reflector] {parsed['reasoning']}")

        if parsed["decision"] == "continue":
            try:
                plan = validate_plan(parsed.get("next_plan") or {}, task)
            except PlanValidationError as exc:
                logger.warning("Reflector's next_plan invalid (%s); stopping instead.", exc)
                return {"decision": "stop", "reasoning": "Invalid next_plan from reflector.", "next_plan": None}
            return {"decision": "continue", "reasoning": parsed.get("reasoning", ""), "next_plan": plan}

        return {"decision": "stop", "reasoning": parsed.get("reasoning", ""), "next_plan": None}

    def _parse_and_validate(self, raw: Optional[str], task: str) -> Dict[str, Any]:
        parsed = self._safe_json_parse(raw)
        if parsed is None:
            logger.warning("LLM returned unparseable JSON; using default plan.")
            return default_plan(task)
        try:
            return validate_plan(parsed, task)
        except PlanValidationError as exc:
            logger.warning("LLM plan failed validation (%s); using default plan.", exc)
            return default_plan(task)

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def _execute_round(self, x: pd.DataFrame, y: pd.Series, task: str, plan: Dict[str, Any]) -> None:
        preprocessing_spec = plan["preprocessing"]
        for candidate in plan["candidates"][: self.config.n_candidates_per_round]:
            logger.info("Evaluating candidate: %s (%s)", candidate["name"], candidate["model"])
            result = evaluate_candidate(
                x,
                y,
                task,
                preprocessing_spec,
                candidate,
                cv_folds=self.config.cv_folds,
                n_iter=self.config.random_search_iters,
                random_state=self.config.random_state,
            )
            if result.error:
                self.state.errors.append(f"{candidate['name']} ({candidate['model']}): {result.error}")
                logger.warning("Candidate failed: %s", result.error)
            else:
                logger.info("  -> score=%.5f (%s)", result.best_score, result.scoring)
            self.state.leaderboard.append(result)
            # Stash preprocessing spec alongside the result for final refit.
            result.cv_results_summary["preprocessing"] = preprocessing_spec

    
    # Finalization
    def _finalize(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        task: str,
        profile: Dict[str, Any],
        out_dir: Path,
        total_seconds: float,
    ) -> Dict[str, Any]:
        successful = [r for r in self.state.leaderboard if not r.error]
        if not successful:
            raise RuntimeError("All candidates failed. See logs / report for errors.")

        best = max(successful, key=lambda r: r.best_score)
        preprocessing_spec = best.cv_results_summary.get("preprocessing", {})
        best_pipeline = build_pipeline(x, preprocessing_spec, best.model)
        best_pipeline.set_params(**{f"model__{k}": v for k, v in best.best_params.items()})
        best_pipeline.fit(x, y)

        model_path = out_dir / "best_pipeline.joblib"
        joblib.dump(best_pipeline, model_path)

        report_path = out_dir / "report.md"
        report_path.write_text(self._build_report(profile, task, best, total_seconds))

        logger.info("Best candidate: %s (%s) score=%.5f", best.name, best.model, best.best_score)
        logger.info("Artifacts written to: %s", out_dir)

        return {
            "best": best.to_dict(),
            "model_path": str(model_path),
            "report_path": str(report_path),
            "leaderboard": [r.to_dict() for r in sorted(self.state.leaderboard, key=lambda r: r.best_score, reverse=True)],
        }

    def _build_report(self, profile: Dict[str, Any], task: str, best: CandidateResult, total_seconds: float) -> str:
        lines = [
            "# AutoML-Agent Run Report",
            "",
            f"- **Task**: {task}",
            f"- **Dataset**: {self.config.data_path} ({profile['n_rows']} rows, {profile['n_features']} features)",
            f"- **Target**: {self.config.target}",
            f"- **LLM model**: {self.llm.name or 'n/a'}",
            f"- **Rounds run**: {self.state.round_num}",
            f"- **Total time**: {total_seconds:.1f}s",
            "",
            "## Best result",
            "",
            f"- **Model**: {best.name} (`{best.model}`)",
            f"- **CV score** ({best.scoring}): `{best.best_score:.5f}`",
            f"- **Best hyperparameters**: `{json.dumps(best.best_params)}`",
            "",
            "## Leaderboard",
            "",
            "| Name | Model | Score | Scoring | Fit (s) | Error |",
            "|---|---|---|---|---|---|",
        ]
        for r in sorted(self.state.leaderboard, key=lambda r: r.best_score, reverse=True):
            score_str = "n/a" if r.error else f"{r.best_score:.5f}"
            lines.append(f"| {r.name} | {r.model} | {score_str} | {r.scoring} | {r.fit_seconds:.1f} | {r.error or ''} |")

        if self.state.llm_reasoning_log:
            lines += ["", "## LLM reasoning log", ""]
            lines += [f"- {entry}" for entry in self.state.llm_reasoning_log]

        if self.state.errors:
            lines += ["", "## Errors encountered", ""]
            lines += [f"- {e}" for e in self.state.errors]

        return "\n".join(lines) + "\n"

    # LLM call helpers
    def _safe_llm_call(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        try:
            return self.llm.complete(system_prompt, user_prompt, json_mode=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM call failed: %s", exc)
            return None

    @staticmethod
    def _safe_json_parse(raw: Optional[str]) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
            return None
