#!/usr/bin/env python
"""End-to-end example: run the AutoML-Agent on a small synthetic
classification dataset (Titanic-like) so you can try the whole loop
without needing to source your own CSV first.

Requires a local Ollama server with a model pulled, e.g.:
    ollama pull llama3.1
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from automl_agent.agent.orchestrator import AutoMLAgent  # noqa: E402
from automl_agent.config import RunConfig  # noqa: E402
from automl_agent.llm.ollama_client import OllamaClient  # noqa: E402


def make_toy_titanic(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pclass = rng.choice([1, 2, 3], size=n, p=[0.2, 0.3, 0.5])
    sex = rng.choice(["male", "female"], size=n, p=[0.55, 0.45])
    age = np.clip(rng.normal(30, 12, size=n), 1, 80)
    age[rng.random(n) < 0.1] = np.nan  # inject some missingness
    fare = np.clip(rng.exponential(30, size=n) + (3 - pclass) * 10, 1, 300)
    embarked = rng.choice(["S", "C", "Q"], size=n, p=[0.7, 0.2, 0.1])
    sibsp = rng.poisson(0.5, size=n)

    # Survival probability loosely mimics real Titanic patterns.
    logit = (
        -0.8
        + (sex == "female") * 2.2
        + (pclass == 1) * 1.1
        + (pclass == 2) * 0.4
        - (pclass == 3) * 0.3
        + (age < 12) * 1.0
        - (fare < 10) * 0.3
    )
    prob = 1 / (1 + np.exp(-logit))
    survived = (rng.random(n) < prob).astype(int)

    return pd.DataFrame(
        {
            "pclass": pclass,
            "sex": sex,
            "age": age,
            "sibsp": sibsp,
            "fare": fare,
            "embarked": embarked,
            "survived": survived,
        }
    )


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "sample_data"
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "toy_titanic.csv"

    df = make_toy_titanic()
    df.to_csv(csv_path, index=False)
    print(f"Wrote synthetic dataset: {csv_path} ({len(df)} rows)")

    config = RunConfig(
        data_path=str(csv_path),
        target="survived",
        task="classification",
        llm_model="llama3.1",
        max_rounds=2,
        time_budget_seconds=180,
        random_search_iters=10,
    )
    llm = OllamaClient(model=config.llm_model, host=config.ollama_host)
    agent = AutoMLAgent(config, llm)
    result = agent.run()

    print("\nDone. Best model:", result["best"])
    print("Report written to:", result["report_path"])


if __name__ == "__main__":
    main()
