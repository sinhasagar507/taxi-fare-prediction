"""
Phase 5 / §5c at scale — refit the tuned LightGBM champion on `sample_full`,
CV only (plan §6, "The SCALE goal").

Carves `sample_full` exactly as `01_run_sweep.py` / `02_tune.py` carve
`sample_work` — temporal test set first, off the raw frame, then the 80/20
holdout — checks the carve's arithmetic with `evaluate.verify_carve_split`,
writes the train split (`--write-train`, uploaded separately by the caller),
then runs the champion's tuned pipeline through the one 5-fold CV harness.

This is deliberately **not** a second tuning run. The point of "refit
champion" is the champion's already-chosen hyperparameters at more rows, not
a new search — `01_run_sweep.py`'s registry has no path to a tuned model, so
this script calls `tune.build_tuned_pipeline` directly instead.

Run (on the scale VM, from the repo root):

    python spark/ml/03_scale_champion.py \
        --input  sample_full.parquet \
        --write-train sample_full_train.parquet

    # wiring smoke, e.g. against the small sample_work.parquet
    python spark/ml/03_scale_champion.py --input sample_work.parquet \
        --write-train /tmp/smoke_train.parquet --smoke --tag scale_smoke

The holdout and temporal set are carved, checked in count only, and deleted
in the same statement — never scored, never printed as anything but a row
count. `--smoke` skips the share-tolerance check, since a different sample
has no reason to reproduce the work split's shares exactly.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from spark.ml.src.evaluate import (  # noqa: E402
    HOLDOUT_FRACTION,
    RANDOM_STATE,
    evaluate,
    make_cv,
    make_holdout,
    make_temporal_test,
    verify_carve_split,
    write_train_split,
)
from spark.ml.src.features import build_features  # noqa: E402
from spark.ml.src.load import load_prep_sample  # noqa: E402
from spark.ml.src.tune import build_tuned_pipeline  # noqa: E402

RESULTS_DIR = REPO_ROOT / "spark" / "ml" / "results"

# Work-split shares, measured 2026-09-13 (modeling plan §4a) on sample_work's
# 1,828,181 rows. HOLDOUT_FRACTION fixes holdout_share near 0.2 regardless of
# sample size; temporal_share is data-dependent, so it is the real check.
REFERENCE_TEMPORAL_SHARE = 133_629 / 1_828_181
REFERENCE_HOLDOUT_SHARE = 338_911 / (1_828_181 - 133_629)

CHAMPION_NAME = "lightgbm"
# tune_d012b.json's best trial (#21), the TUNE goal, 2026-09-16/17.
CHAMPION_PARAMS = {
    "n_estimators": 617,
    "learning_rate": 0.03187483634849957,
    "num_leaves": 174,
    "min_child_samples": 13,
    "colsample_bytree": 0.9213481771321222,
    "reg_lambda": 4.121935086080686,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="local path to a prep sample")
    ap.add_argument("--write-train", required=True,
                    help="local path to write the carved train split parquet")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", default="scale_full_lightgbm")
    ap.add_argument("--tolerance", type=float, default=0.03)
    ap.add_argument("--smoke", action="store_true",
                    help="skip the share-tolerance check (a different sample "
                         "has no reason to match the work split's shares)")
    args = ap.parse_args()

    # Lean load (plan §6, "RESCALE"): only the columns the feature path uses,
    # decimals already float64. pd.read_parquet here OOM-killed a 32 GiB VM.
    df = load_prep_sample(args.input)
    n_sample = len(df)
    print(f"[data] {args.input}: {n_sample:,} rows")

    # Temporal test set first, on the raw frame, then the random holdout —
    # the same order §4a fixes for every sweep run.
    df, df_temporal = make_temporal_test(df)
    n_temporal = len(df_temporal)
    del df_temporal

    X, y = build_features(df, include_duration=True)
    del df  # the raw frame is dead weight from here on
    X, X_holdout, y, _ = make_holdout(X, y, test_size=HOLDOUT_FRACTION)
    n_holdout = len(X_holdout)
    del X_holdout
    n_train = len(X)

    check = verify_carve_split(
        n_sample, n_temporal, n_holdout, n_train,
        reference_temporal_share=None if args.smoke else REFERENCE_TEMPORAL_SHARE,
        reference_holdout_share=None if args.smoke else REFERENCE_HOLDOUT_SHARE,
        tolerance=args.tolerance,
    )
    print(f"[split] sample={n_sample:,} temporal={n_temporal:,} "
          f"holdout={n_holdout:,} train={n_train:,} "
          "— temporal and holdout SEALED, deleted")
    print(f"[split] shares: {check}")
    if not check["shares_match"]:
        raise SystemExit(f"[stop] carve shares differ from the work split "
                          f"beyond tolerance {args.tolerance}: {check}")

    # Chunked, so the write never holds a second copy of X.
    write_train_split(X, y, args.write_train)
    print(f"[write] {args.write_train} ({n_train:,} rows)")

    cv = make_cv(n_splits=args.folds)
    pipeline = build_tuned_pipeline(CHAMPION_NAME, CHAMPION_PARAMS, X.columns)
    started = time.time()
    row = evaluate(pipeline, X, y, cv=cv, name=f"{CHAMPION_NAME}@scale_full")
    elapsed = time.time() - started

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    board_path = RESULTS_DIR / f"leaderboard_{args.tag}.csv"
    pd.DataFrame([row]).to_csv(board_path, index=False)

    meta = {
        "tag": args.tag,
        "model": CHAMPION_NAME,
        "params": CHAMPION_PARAMS,
        "rows": n_train,
        "sample_rows": n_sample,
        "temporal_rows": n_temporal,
        "holdout_rows": n_holdout,
        "carve_check": check,
        "folds": args.folds,
        "seed": RANDOM_STATE,
        "elapsed_s": round(elapsed, 1),
        "holdout_scored": False,
        "temporal_scored": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    meta_path = RESULTS_DIR / f"{args.tag}.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(pd.DataFrame([row]).to_string(index=False))
    print(f"[write] {board_path}")
    print(f"[write] {meta_path}")
    print(f"[time] {elapsed:.1f}s")
    print("SCALE OK")


if __name__ == "__main__":
    main()
