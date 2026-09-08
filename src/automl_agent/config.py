"""Central configuration for an AutoML-Agent run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime


@dataclass
class RunConfig:
    """All knobs for a single end-to-end agent run."""

    data_path: str
    target: str
    task: str = "auto"  # "classification" | "regression" | "auto"

    # LLM
    llm_model: str = "llama3.1"
    ollama_host: str = "http://localhost:11434"
    llm_temperature: float = 0.2
    llm_timeout: int = 120

    # Agent loop
    max_rounds: int = 3
    time_budget_seconds: int = 300
    n_candidates_per_round: int = 4
    cv_folds: int = 5
    random_search_iters: int = 15
    test_size: float = 0.2
    random_state: int = 42

    # Output
    out_dir: str = field(default_factory=lambda: f"runs/{datetime.now():%Y%m%d_%H%M%S}")

    def resolved_out_dir(self) -> Path:
        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
