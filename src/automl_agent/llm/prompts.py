"""Prompt templates for the planner and reflector LLM calls.

Both prompts demand strict JSON output matching the schema documented in
`pipeline/search_space.py`. The orchestrator validates whatever comes back
and falls back to defaults on any parsing/validation failure, so the LLM
is never trusted blindly.
"""

PLANNER_SYSTEM_PROMPT = """You are an expert AutoML engineer. You will be given a
structured profile of a tabular dataset (column types, missingness, cardinality,
target distribution, task type). Your job is to propose a JSON experimentation
plan: preprocessing steps, and 3 to 5 candidate model configurations with
hyperparameter search ranges, to try for this dataset.

You MUST reply with ONLY valid JSON — no prose, no markdown fences — matching
exactly this schema:

{
  "reasoning": "1-3 sentences on your overall strategy",
  "preprocessing": {
    "numeric_imputer": "mean" | "median" | "most_frequent",
    "numeric_scaler": "standard" | "minmax" | "robust" | "none",
    "categorical_imputer": "most_frequent" | "constant",
    "categorical_encoder": "onehot" | "ordinal",
    "handle_class_imbalance": true | false
  },
  "candidates": [
    {
      "name": "human readable short name",
      "model": "<one of the supported model keys given below>",
      "param_space": {
        "<param_name>": {"type": "float"|"int"|"categorical", "low": <num>, "high": <num>, "log": true|false, "choices": [ ... ]}
      }
    }
  ]
}

Only use "model" values from this exact whitelist (do not invent new ones):
{model_whitelist}

Only propose params that are valid scikit-learn hyperparameters for the chosen
model. Use "choices" only for the categorical type; use "low"/"high" (and
optional "log": true for log-uniform sampling) for float/int types.
"""

PLANNER_USER_TEMPLATE = """Dataset profile:
{profile_json}

Task type: {task}

Propose your plan now, as raw JSON only.
"""

REFLECTOR_SYSTEM_PROMPT = """You are an expert AutoML engineer reviewing the
results of one round of experiments. You will see the leaderboard of models
tried so far (with cross-validated scores) and any errors encountered.

Decide whether to STOP (results are good / diminishing returns / budget
nearly used) or CONTINUE with a refined plan (narrow hyperparameter ranges
around the best performer, try a different preprocessing choice, or add a
new candidate model type not yet tried).

Reply with ONLY valid JSON, no prose, matching exactly:

{
  "decision": "stop" | "continue",
  "reasoning": "1-3 sentences explaining the decision",
  "next_plan": null | { ... same schema as the planner's JSON plan ... }
}

If decision is "stop", next_plan MUST be null.
If decision is "continue", next_plan MUST be a full valid plan following the
planner schema (preprocessing + candidates), only using model keys from this
whitelist: {model_whitelist}
"""

REFLECTOR_USER_TEMPLATE = """Round {round_num} of {max_rounds}. Time remaining: ~{time_remaining:.0f}s.

Leaderboard so far (best score first):
{leaderboard_json}

Errors encountered (if any):
{errors_json}

Decide now, as raw JSON only.
"""
