# AutoML-Agent

**LLM-Guided AutoML Experimentation Agent** — a local-first agent that uses an
open-source LLM (via [Ollama](https://ollama.com)) to plan, build, tune, and
iterate on ML pipelines (preprocessing → model → hyperparameters) for a given
tabular dataset, then reports back results in plain English.

No cloud LLM API keys required — everything runs locally against an
Ollama-served model (e.g. `llama3.1`, `qwen2.5`, `mistral`).

## How it works

```
 dataset.csv
      │
      ▼
 ┌─────────────┐   dataset profile (schema, stats, target,
 │  Profiler   │   missingness, cardinality, class balance)
 └─────────────┘
      │
      ▼
 ┌─────────────┐   "Given this dataset profile, propose a JSON
 │  LLM Planner│    plan: preprocessing steps, 3-5 candidate
 │  (Ollama)   │    models, and hyperparameter search spaces."
 └─────────────┘
      │  structured JSON plan
      ▼
 ┌─────────────┐   builds sklearn Pipeline(s) per candidate,
 │  Pipeline   │   runs randomized hyperparameter search with
 │  Builder +  │   cross-validation
 │  Tuner      │
 └─────────────┘
      │  leaderboard of results
      ▼
 ┌─────────────┐   LLM reviews leaderboard + errors, decides:
 │  Reflector  │   stop, refine search space, or try a new
 │  (Ollama)   │   preprocessing/model idea → loop back
 └─────────────┘
      │
      ▼
  best_pipeline.joblib + report.md
```

The agent runs a bounded loop (default 3 rounds): **plan → execute → reflect
→ (re-plan)**. Every LLM call is schema-constrained (JSON) and validated;
if the LLM output is malformed or proposes something unsafe/unsupported,
the agent falls back to a safe default plan instead of crashing.

## Project layout

```
automl-agent/
├── src/automl_agent/
│   ├── config.py            # dataclasses for run configuration
│   ├── llm/
│   │   ├── base.py          # LLMClient interface
│   │   ├── ollama_client.py # Ollama HTTP client (local, no API key)
│   │   └── prompts.py       # prompt templates for planner/reflector
│   ├── data/
│   │   ├── profiler.py      # dataset -> structured profile dict
│   │   └── preprocessing.py # sklearn ColumnTransformer builders
│   ├── models/
│   │   ├── sklearn_models.py# registry: name -> estimator + param space
│   │   └── torch_models.py  # optional small MLP (PyTorch), same registry API
│   ├── pipeline/
│   │   ├── search_space.py  # JSON plan -> sklearn/torch param distributions
│   │   └── builder.py       # JSON plan -> runnable sklearn Pipeline
│   ├── tuning/
│   │   └── optimizer.py     # randomized search + CV, model-agnostic
│   ├── agent/
│   │   └── orchestrator.py  # the plan -> execute -> reflect loop
│   └── utils/
│       └── logging.py
├── examples/
│   └── run_titanic.py       # end-to-end example on a bundled toy dataset
├── tests/
│   └── test_profiler.py
├── main.py                  # CLI entry point
├── requirements.txt
└── .gitignore
```

## Setup

```bash
# 1. Install Ollama and pull a model (one-time)
#    https://ollama.com/download
ollama pull llama3.1

# 2. Python env
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # installs the automl_agent package

# 3. Run
python main.py --data path/to/your.csv --target target_column_name
```

## Quick start (bundled toy example)

```bash
python examples/run_titanic.py
```

This downloads/generates a small toy classification dataset, runs the full
plan → execute → reflect loop for 2 rounds against your local Ollama model,
and writes `runs/<timestamp>/report.md` + `best_pipeline.joblib`.

## CLI usage

```bash
python main.py \
  --data data/train.csv \
  --target label \
  --task classification \
  --model qwen2.5:7b \
  --rounds 3 \
  --time-budget 300 \
  --out runs/my_experiment
```

Key flags (see `python main.py --help` for the full list):

| Flag | Description | Default |
|---|---|---|
| `--data` | Path to CSV dataset | required |
| `--target` | Target column name | required |
| `--task` | `classification` \| `regression` \| `auto` | `auto` |
| `--model` | Ollama model tag | `llama3.1` |
| `--rounds` | Max plan/reflect rounds | `3` |
| `--time-budget` | Soft wall-clock budget (seconds) for tuning | `300` |
| `--out` | Output directory for report + artifacts | `runs/<timestamp>` |

## Design principles

- **LLM proposes, code disposes.** The LLM never executes arbitrary code —
  it only emits JSON that is validated against a strict schema
  (`pipeline/search_space.py`) and mapped to a whitelist of supported
  transformers/estimators (`models/sklearn_models.py`). This keeps the
  agent safe to run unattended.
- **Local-first.** Built against Ollama's `/api/chat` endpoint by default;
  the `LLMClient` interface in `llm/base.py` makes it easy to swap in any
  other backend later.
- **Graceful degradation.** If the LLM is unreachable or returns invalid
  JSON, the orchestrator falls back to a sensible default search space so
  the pipeline still produces a result.
- **Explainable output.** Every run produces a Markdown report summarizing
  what was tried, what worked, and the LLM's own reasoning for its choices.

## Roadmap ideas

- [ ] Optuna-based Bayesian search as an alternative to randomized search
- [ ] Multi-dataset benchmark harness
- [ ] Feature-engineering suggestions (not just model/hyperparameter choice)
- [ ] Web UI (Streamlit) on top of the same orchestrator
- [ ] Support for imbalanced classification metrics/strategies

## License

MIT — see `LICENSE`.
