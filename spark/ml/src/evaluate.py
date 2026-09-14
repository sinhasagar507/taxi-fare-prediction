"""Shared CV harness + leaderboard for the fare-prediction model sweep
(Phase 3 of the modeling plan).

One evaluate() that every model plugs into — identical KFold folds, identical
metrics — so the Phase-4 sweep is a fair comparison instead of anecdotes.
Models arrive as Pipeline(preprocessor, model) so all preprocessing is fit
inside each training fold (leakage-safe by construction).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split

RANDOM_STATE = 42

# Columns the prep stratified its sample on (service_type x temp_band); after
# build_features the band arrives as its ordinal, which is a 1:1 recode.
STRATIFY_COLUMNS = ("service_type", "temp_band_ord")

HOLDOUT_FRACTION = 0.2

# The temporal test set (plan §4a): every trip from this instant on — the last
# 2 of the 24 months — is sealed whole, before the random holdout is drawn.
TEMPORAL_CUTOFF = pd.Timestamp("2016-11-01")
TEMPORAL_COLUMN = "pickup_datetime"

# Metric functions — single source for both the CV scorers and any ad-hoc
# reporting (Phase-5 slice metrics reuse these).
mae = mean_absolute_error
rmse = root_mean_squared_error
r2 = r2_score


def compute_metrics(y_true, y_pred) -> dict:
    """All three sweep metrics for one prediction set."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def make_cv(n_splits: int = 5) -> KFold:
    """The one fold set every model is scored on (shuffled, fixed seed)."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def make_holdout(
    X,
    y,
    test_size: float = HOLDOUT_FRACTION,
    stratify_columns=STRATIFY_COLUMNS,
    random_state: int = RANDOM_STATE,
):
    """Carve the sealed test set off the front of the workflow (plan §4a).

    Returns (X_train, X_test, y_train, y_test). Everything downstream — the
    sweep, CV, hyperparameter search — sees `X_train` only. The test split is
    scored exactly once, in Phase 5, after the champion is chosen on CV
    evidence alone; scoring it repeatedly while tuning silently turns it into
    a validation set and the final number stops being an honest estimate.

    Stratified on `service_type` x `temp_band_ord`, the same key
    00_prep_spark.py sampled with, so the split cannot skew a small stratum
    (Green/Freezing is only ~1.5% of the work sample). Columns that aren't
    present are skipped; with none present the split is plain random, so the
    duration ablation and reduced smoke frames still work.

    Deterministic under `random_state`: the same sample file and seed always
    reproduce the same holdout, which is what keeps it stable across runs
    without persisting a second copy of the data.
    """
    present = [c for c in stratify_columns if c in X.columns]
    strata = None
    if present:
        strata = X[present[0]].astype(str)
        for col in present[1:]:
            strata = strata + "|" + X[col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=strata,
    )
    return X_train, X_test, y_train, y_test


def make_temporal_test(
    df: pd.DataFrame,
    cutoff: pd.Timestamp = TEMPORAL_CUTOFF,
    column: str = TEMPORAL_COLUMN,
):
    """Carve the temporal test set off a prep sample (plan §4a).

    Returns (before, temporal): trips that start before `cutoff`, and trips
    that start at or after it. The random 80/20 holdout lets 2016-06 inform a
    2016-03 prediction; the temporal set holds out the last two months whole,
    so Phase 5 can score the champion on trips later than anything it saw.
    It is sealed like the holdout — scored once, in Phase 5, never here.

    Runs on the **raw** frame, before `build_features`, because that drops
    `pickup_datetime`. `make_holdout` then draws only from `before`. Index
    labels are kept, so every row stays traceable to the sample file.

    A frame without the date column raises rather than returning an empty
    temporal set: the pre-D-012 samples carry no date, and an empty carve
    would look like a seal while sealing nothing.
    """
    if column not in df.columns:
        raise ValueError(
            f"no '{column}' column to carve the temporal test set on — the "
            "sample predates it; regenerate it with 00_prep_spark.py"
        )
    is_temporal = df[column] >= cutoff
    return df.loc[~is_temporal], df.loc[is_temporal]


def train_split_filename(
    sample: str,
    include_duration: bool = True,
    limit_rows: int | None = None,
) -> str:
    """Name the persisted train split from what determines its contents.

    Deliberately does **not** take the run's `--tag`. A tag is a presentation
    label chosen per invocation to keep leaderboard files apart; keying a data
    artifact off it means `--tag anything` silently produces a differently
    named split, and the file plan §5b points at stops existing. The three
    inputs here are the ones that actually change what is in the file: which
    prep sample, whether `trip_duration_min` is in the column set, and whether
    the rows were limited for a smoke run.

    A row limit lands in the name so a smoke run can never overwrite the real
    split — `limit_rows=0` is a limit, not a missing one.
    """
    parts = [f"sample_{sample}"]
    if not include_duration:
        parts.append("no_duration")
    if limit_rows is not None:
        parts.append(f"limit{limit_rows}")
    parts.append("train")
    return "_".join(parts) + ".parquet"


def write_train_split(X, y, path, target_name: str | None = None) -> Path:
    """Persist the train half of the sealed split as one parquet file.

    The MLlib baseline (plan §5b) has to train on the *same* rows the sklearn
    sweep used, but `make_holdout` is `sklearn.train_test_split` — seeds do not
    transfer across libraries, so Spark cannot re-derive the split. pandas
    writes it once and the Spark script reads that file. It stays a derived
    artifact: regenerable from sample + seed + fraction, so it is gitignored
    rather than versioned.

    Features and target go in **one** frame because MLlib needs `labelCol` in
    the same DataFrame. They are joined on index label, not position, since the
    input is `make_holdout`'s output whose index is shuffled and gappy.

    The pandas index is deliberately not written: `to_parquet` would emit it as
    `__index_level_0__` and Spark would read that back as an ordinary column.
    `mllib.split_column_groups` now allowlists its groups, so such a column is
    dropped and reported rather than trained on — but not writing it is still
    the right guard. It keeps the file to what it claims to hold, and stops
    every MLlib run printing an "unrecognised column" warning that is really
    just this function's own leftovers.
    """
    name = target_name if target_name is not None else y.name
    if name is None:
        raise ValueError(
            "target has no name and target_name was not given — writing it as "
            "a column called 'None' would fail later, inside Spark"
        )
    if name in X.columns:
        raise ValueError(
            f"target name '{name}' collides with a feature column — the "
            "duplicate would silently double that feature in Spark"
        )

    frame = pd.concat([X, y.rename(name)], axis=1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


_SCORING = {
    "mae": make_scorer(mae, greater_is_better=False),
    "rmse": make_scorer(rmse, greater_is_better=False),
    "r2": make_scorer(r2),
}


def evaluate(model, X, y, cv=None, name: str | None = None, n_jobs=None) -> dict:
    """Cross-validate one model and return its leaderboard row.

    `model` is any sklearn estimator — in the sweep, always a
    Pipeline(preprocessor, model). Error scorers come back negated by
    sklearn convention; they are flipped back to positive here.
    """
    cv = cv if cv is not None else make_cv()
    scores = cross_validate(model, X, y, cv=cv, scoring=_SCORING, n_jobs=n_jobs)
    return {
        "model": name if name is not None else type(model).__name__,
        "mae_mean": -scores["test_mae"].mean(),
        "mae_std": scores["test_mae"].std(),
        "rmse_mean": -scores["test_rmse"].mean(),
        "rmse_std": scores["test_rmse"].std(),
        "r2_mean": scores["test_r2"].mean(),
        "r2_std": scores["test_r2"].std(),
        "fit_time_s": scores["fit_time"].mean(),
    }


def leaderboard(rows: list[dict]) -> pd.DataFrame:
    """Assemble evaluate() rows into the running leaderboard, best RMSE first."""
    board = pd.DataFrame(rows)
    if board.empty:
        return board
    return board.sort_values("rmse_mean", ignore_index=True)
