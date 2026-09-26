"""
Phase 5 — the one score on the sealed sets (modeling plan §4a, §6 "HOLDOUT").

Fits the tuned LightGBM champion on the work split's train rows and scores it
**once** on both sealed sets of `sample_work`: the random holdout and the
temporal set (trips from 2016-11-01). The temporal set is also scored per
pickup month.

The once-only rule is enforced, not trusted: `holdout.guard_once` runs before
anything is loaded, and `holdout.write_score_once` refuses to overwrite. The
score file is gitignored, so the committed docs are the permanent record.

Why the work split and not the RESCALE refit: `sample_work` is drawn from
`sample_full` (00_prep_spark.py), so the refit has trained on most of this
holdout. §5c stays CV only.

Run once, in the dev container, from the repo root:

    python spark/ml/04_holdout_score.py

Wiring smoke — a synthetic sample only, never the real sealed rows:

    python spark/ml/04_holdout_score.py --smoke --input /tmp/fake.parquet \
        --score-path /tmp/fake_score.json

Prints metrics only — never a sealed row or a prediction.
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from spark.ml.src.evaluate import (  # noqa: E402
    HOLDOUT_FRACTION,
    RANDOM_STATE,
    TEMPORAL_CUTOFF,
    make_holdout,
    make_temporal_test,
)
from spark.ml.src.features import TARGET, build_features  # noqa: E402
from spark.ml.src.holdout import guard_once, score_sealed, write_score_once  # noqa: E402
from spark.ml.src.load import load_prep_sample  # noqa: E402
from spark.ml.src.tune import build_tuned_pipeline  # noqa: E402

# The single source of the champion's params: the §5c script's constant.
_scale = importlib.import_module("spark.ml.03_scale_champion")
CHAMPION_NAME = _scale.CHAMPION_NAME
CHAMPION_PARAMS = _scale.CHAMPION_PARAMS

DATA_DIR = REPO_ROOT / "spark" / "ml" / "data"
RESULTS_DIR = REPO_ROOT / "spark" / "ml" / "results"

# Measured when the HOLDOUT goal was written (2026-09-26).
EXPECTED_ROWS = {"sample": 1_828_181, "temporal": 133_629, "holdout": 338_911, "train": 1_355_641}
# The TUNE champion's CV score on the same train rows (§6, "The TUNE results").
CV_MAE = {"mean": 0.312976, "std": 0.001181}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATA_DIR / "sample_work.parquet"))
    ap.add_argument("--train-split", default=str(DATA_DIR / "sample_work_train.parquet"))
    ap.add_argument("--score-path", default=str(RESULTS_DIR / "holdout_d012.json"))
    ap.add_argument("--smoke", action="store_true",
                    help="synthetic input: skip the FACTS row counts and the "
                         "train-split comparison")
    args = ap.parse_args()

    guard_once(args.score_path)  # before any load, fit or predict

    df = load_prep_sample(args.input)
    n_sample = len(df)
    df, df_temporal = make_temporal_test(df)
    months = df_temporal["pickup_datetime"].dt.strftime("%Y-%m").to_numpy()
    X_temporal, y_temporal = build_features(df_temporal, include_duration=True)
    del df_temporal

    X, y = build_features(df, include_duration=True)
    del df
    X, X_holdout, y, y_holdout = make_holdout(X, y, test_size=HOLDOUT_FRACTION)

    counts = {"sample": n_sample, "temporal": len(X_temporal),
              "holdout": len(X_holdout), "train": len(X)}
    print(f"[split] {counts}")
    if not args.smoke:
        if counts != EXPECTED_ROWS:
            raise SystemExit(f"[stop] row counts differ from FACTS: {counts} != {EXPECTED_ROWS}")
        train_target = pd.read_parquet(args.train_split, columns=[TARGET])[TARGET]
        if not np.array_equal(y.to_numpy(), train_target.to_numpy()):
            raise SystemExit("[stop] the train target differs from sample_work_train.parquet")
        print("[split] counts equal FACTS; train target equals sample_work_train.parquet")

    pipeline = build_tuned_pipeline(CHAMPION_NAME, CHAMPION_PARAMS, X.columns)
    result = score_sealed(
        pipeline, X, y,
        {"holdout": (X_holdout, y_holdout), "temporal": (X_temporal, y_temporal)},
        groups={"temporal": months},
    )

    hold, temp = result["sets"]["holdout"], result["sets"]["temporal"]
    result.update({
        "model": CHAMPION_NAME,
        "params": CHAMPION_PARAMS,
        "counts": counts,
        "temporal_cutoff": str(TEMPORAL_CUTOFF.date()),
        "seed": RANDOM_STATE,
        "cv_mae_context": CV_MAE,
        "gap_mae_holdout_minus_cv": hold["mae"] - CV_MAE["mean"],
        "gap_mae_temporal_minus_holdout": temp["mae"] - hold["mae"],
        "smoke": args.smoke,
        "python": platform.python_version(),
        "platform": platform.platform(),
    })
    write_score_once(result, args.score_path)

    print(json.dumps({k: result[k] for k in (
        "fit_time_s", "sets", "gap_mae_holdout_minus_cv", "gap_mae_temporal_minus_holdout")},
        indent=2))
    print(f"[write] {args.score_path}")
    print("HOLDOUT SCORED ONCE")


if __name__ == "__main__":
    main()
