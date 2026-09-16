"""Phase 5 — tune the two leading boosters and diagnose the champion.

The tuner adds a search, not a second harness. Every trial is
Pipeline(preprocessor, model) scored by `evaluate()` on the caller's folds, the
same path the sweep takes, so a tuned MAE sits on the same scale as the
`work_d012` leaderboard.

Design notes:

- **Trial 0 is the sweep.** `run_study` enqueues `default_params` first. Those
  values are the libraries' own defaults, measured in the dev container on
  2026-09-16, and a test proves they build a pipeline that predicts exactly like
  the registry's. A study whose best trial is trial 0 therefore says "tuning
  bought nothing", not "tuning made it worse".
- **Every default sits inside its search range**, so the sampler can return to
  it. `reg_lambda` is searched on a linear scale because its default is 0.
- **CatBoost keeps its automatic learning rate and 1,000 iterations.** Its
  default rate is derived from the data size, so no fixed number reproduces
  trial 0. Only `depth`, `random_strength` and `subsample` (the MVS bootstrap
  default) are searched.
- **`l2_leaf_reg` is not searched, measured 2026-09-16.** Passing it at all, even
  at its default of 3, changes the automatic learning rate CatBoost derives
  (0.029288 became 0.030000 on the test fixture) and so changes every
  prediction. The other three leave the rate unchanged. Searching it would make
  trial 0 a different model from the sweep's.
  `depth` stops at 8, because each extra level doubles the tree size and one
  default fit already costs 30.5 s on the work split.
- **The metric is MAE**, the owner's reading metric (D-013). RMSE, R² and the
  fold stds ride along as trial attributes, so the leaderboard row needs no
  second fit.
- The holdout and the temporal test set never reach this module. The caller
  passes the train split only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spark.ml.src.evaluate import RANDOM_STATE, evaluate
from spark.ml.src.sweep import ModelSpec, build_pipeline, make_catboost, make_lightgbm

TUNED_MODELS = ("lightgbm", "catboost")

_FACTORIES = {"lightgbm": make_lightgbm, "catboost": make_catboost}

# Library defaults, measured 2026-09-16: lightgbm 4.7.0 get_params(), catboost
# 1.2.10 get_all_params() after a fit.
_DEFAULTS = {
    "lightgbm": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_child_samples": 20,
        "colsample_bytree": 1.0,
        "reg_lambda": 0.0,
    },
    "catboost": {
        "depth": 6,
        "random_strength": 1.0,
        "subsample": 0.8,
    },
}


def _check(name: str) -> None:
    if name not in TUNED_MODELS:
        raise ValueError(f"Unknown tuned model '{name}' — expected one of {TUNED_MODELS}")


def default_params(name: str) -> dict:
    """The parameters trial 0 enqueues — the values the sweep ran with."""
    _check(name)
    return dict(_DEFAULTS[name])


def suggest_params(name: str, trial) -> dict:
    """Draw one parameter set. Each range contains the default above."""
    _check(name)
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        }
    return {
        "depth": trial.suggest_int("depth", 4, 8),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }


def build_tuned_pipeline(name: str, params: dict, feature_columns):
    """The registry's pipeline for `name`, with `params` set on the estimator.

    A fresh factory estimator every call — the sweep's rule against sharing a
    fitted instance between runs.
    """
    _check(name)
    estimator = _FACTORIES[name]().set_params(**params)
    return build_pipeline(ModelSpec(name, estimator, "tree", "boosting"), feature_columns)


def make_objective(name: str, X, y, cv):
    """Optuna objective: evaluate() MAE on `cv`, other metrics as attributes."""
    _check(name)

    def objective(trial) -> float:
        params = suggest_params(name, trial)
        row = evaluate(build_tuned_pipeline(name, params, X.columns), X, y, cv=cv, name=name)
        for key in ("mae_std", "rmse_mean", "rmse_std", "r2_mean", "r2_std", "fit_time_s"):
            trial.set_user_attr(key, float(row[key]))
        return float(row["mae_mean"])

    return objective


def run_study(name: str, X, y, cv, n_trials: int | None = None, timeout: float | None = None):
    """A seeded, minimising study whose trial 0 is the defaults."""
    import optuna

    _check(name)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name=f"tune_{name}",
    )
    study.enqueue_trial(default_params(name))
    study.optimize(make_objective(name, X, y, cv), n_trials=n_trials, timeout=timeout)
    return study


def hour_from_cyclic(hour_sin, hour_cos) -> np.ndarray:
    """Invert features.encode_cyclic_hour back to integer hours 0-23."""
    angle = np.arctan2(np.asarray(hour_sin, dtype=float), np.asarray(hour_cos, dtype=float))
    hours = np.rint(angle * 24.0 / (2 * np.pi)).astype(int) % 24
    return hours


def slice_mae(y_true, y_pred, groups) -> pd.DataFrame:
    """MAE and row count per group, ordered by group."""
    name = getattr(groups, "name", None) or "group"
    frame = pd.DataFrame(
        {
            name: np.asarray(groups),
            "abs_err": np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)),
        }
    )
    out = (
        frame.groupby(name, sort=True)["abs_err"]
        .agg(rows="size", mae="mean")
        .reset_index()
    )
    return out[[name, "rows", "mae"]]


def pick_champion(rows: dict) -> dict:
    """The owner's rule (2026-09-16) between exactly two tuned models.

    The lower tuned CV MAE wins. When the gap is inside the larger of the two
    fold stds, the difference is noise, and the faster fit wins instead.
    """
    if len(rows) != 2:
        raise ValueError(f"pick_champion compares two models, got {len(rows)}")
    (a, ra), (b, rb) = rows.items()
    lower, higher = (a, b) if ra["mae_mean"] <= rb["mae_mean"] else (b, a)
    gap = abs(ra["mae_mean"] - rb["mae_mean"])
    larger_std = max(ra["mae_std"], rb["mae_std"])
    inside = bool(gap <= larger_std)
    if inside:
        champion = a if ra["fit_time_s"] <= rb["fit_time_s"] else b
    else:
        champion = lower
    return {
        "champion": champion,
        "lower_mae": lower,
        "gap": gap,
        "larger_std": larger_std,
        "inside_std": inside,
    }
