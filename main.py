#!/usr/bin/env python
"""CLI entry point for AutoML-Agent.

Example:
    python main.py --data data/train.csv --target label --model llama3.1
"""

from __future__ import annotations

import argparse
import json
import sys

from automl_agent.agent.orchestrator import AutoMLAgent
from automl_agent.config import RunConfig
from automl_agent.llm.ollama_client import OllamaClient
from automl_agent.utils.logging import get_logger

logger = get_logger("automl_agent.cli")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-Guided AutoML Experimentation Agent")
    p.add_argument("--data", required=True, help="Path to CSV dataset")
    p.add_argument("--target", required=True, help="Target column name")
    p.add_argument("--task", default="auto", choices=["auto", "classification", "regression"])
    p.add_argument("--model", default="llama3.1", help="Ollama model tag (default: llama3.1)")
    p.add_argument("--ollama-host", default="http://localhost:11434")
    p.add_argument("--rounds", type=int, default=3, help="Max plan/reflect rounds")
    p.add_argument("--time-budget", type=int, default=300, help="Soft wall-clock budget in seconds")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--search-iters", type=int, default=15, help="RandomizedSearchCV iterations per candidate")
    p.add_argument("--out", default=None, help="Output directory (default: runs/<timestamp>)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    config_kwargs = dict(
        data_path=args.data,
        target=args.target,
        task=args.task,
        llm_model=args.model,
        ollama_host=args.ollama_host,
        max_rounds=args.rounds,
        time_budget_seconds=args.time_budget,
        cv_folds=args.cv_folds,
        random_search_iters=args.search_iters,
    )
    if args.out:
        config_kwargs["out_dir"] = args.out

    config = RunConfig(**config_kwargs)
    llm = OllamaClient(model=config.llm_model, host=config.ollama_host, temperature=config.llm_temperature)

    agent = AutoMLAgent(config, llm)
    try:
        result = agent.run()
    except Exception as exc:  # noqa: BLE001
        logger.error("Run failed: %s", exc)
        return 1

    print("\n" + "=" * 60)
    print(f"Best model : {result['best']['name']} ({result['best']['model']})")
    print(f"Score      : {result['best']['best_score']} ({result['best']['scoring']})")
    print(f"Pipeline   : {result['model_path']}")
    print(f"Report     : {result['report_path']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
