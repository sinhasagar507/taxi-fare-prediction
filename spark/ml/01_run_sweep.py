"""
Phase 4 — run the classical model sweep and write the leaderboard.

Loads a prep sample, builds the leakage-safe feature frame, and scores every model
in the plan §5 registry through the shared Phase-3 CV harness. All the logic lives
in unit-tested modules (`src/features.py`, `src/preprocess.py`, `src/evaluate.py`,
`src/sweep.py`); this script is the thin CLI around them, following the same
convention as 00_prep_spark.py.

Run (from repo root). The dev container is the real target — xgboost/lightgbm/catboost
are container-only, so a host run silently sweeps 11 models instead of 14:

    # smoke: wiring check, ~100K rows, 3 folds
    docker compose -f docker/dev/docker-compose.yml run --rm dev \
        python spark/ml/01_run_sweep.py --limit-rows 100000 --folds 3

    # full sweep on the work sample
    docker compose -f docker/dev/docker-compose.yml run --rm dev \
        python spark/ml/01_run_sweep.py

    # duration ablation (plan §0 locked decision 1)
    ... python spark/ml/01_run_sweep.py --no-duration --tag no_duration

    # persist the exact training rows for the Spark MLlib baseline (§5b)
    ... python spark/ml/01_run_sweep.py --write-train

Outputs (per run, keyed by --tag):
    spark/ml/results/leaderboard_<tag>.csv    the ranked table
    spark/ml/results/sweep_<tag>.json         run metadata — rows, folds, models,
                                              env, wall time — so a leaderboard is
                                              never a number without provenance
    spark/ml/data/sample_<sample>_train.parquet
                                              --write-train only: the train half of
                                              the sealed split, for Spark. Named from
                                              the sample/ablation/limit — NOT --tag,
                                              which only keys the two files above.

Design notes:
  - The temporal test set — every trip from 2016-11-01 on — is sealed first, on
    the raw frame, and is never scored here either. The holdout below is drawn
    from the earlier months only.
  - A stratified 20% holdout is sealed off before the sweep starts and is never
    scored here. The leaderboard below is CV-on-train only; the sealed rows are
    scored once in Phase 5, after the champion is chosen. `--no-holdout` exists
    for wiring smoke runs, not for runs you intend to draw conclusions from.
  - Models are NOT fitted here beyond cross-validation; persisting a champion is
    Phase 5's job after tuning.
  - The mean baseline (`--baseline`) is off by default. It is a reading aid: any
    model near it has learned nothing.
  - `--limit-rows` samples with the project seed, so a smoke run is reproducible
    and its leaderboard is comparable across invocations.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd

# Import the unit-tested modules by their package path (repo root on sys.path).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spark.ml.src.evaluate import (  # noqa: E402
    HOLDOUT_FRACTION,
    RANDOM_STATE,
    STRATIFY_COLUMNS,
    TEMPORAL_CUTOFF,
    make_cv,
    make_holdout,
    make_temporal_test,
    train_split_filename,
    write_train_split,
)
from spark.ml.src.features import build_features  # noqa: E402
from spark.ml.src.sweep import build_model_specs, run_sweep  # noqa: E402

DATA_DIR = REPO_ROOT / "spark" / "ml" / "data"
RESULTS_DIR = REPO_ROOT / "spark" / "ml" / "results"

SAMPLES = {
    "work": DATA_DIR / "sample_work.parquet",
    "full": DATA_DIR / "sample_full.parquet",
}


def load_sample(sample: str, limit_rows: int | None) -> pd.DataFrame:
    """Read a prep sample, optionally down to a seeded subset for smoke runs."""
    path = SAMPLES[sample]
    if not path.exists():
        raise SystemExit(
            f"[error] {path} not found — run spark/ml/00_prep_spark.py first."
        )
    df = pd.read_parquet(path)
    if limit_rows is not None and limit_rows < len(df):
        df = df.sample(n=limit_rows, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=sorted(SAMPLES), default="work",
                    help="which prep sample to sweep (default: work)")
    ap.add_argument("--limit-rows", type=int, default=None,
                    help="seeded row subset for a fast smoke run")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-duration", action="store_true",
                    help="drop trip_duration_min — the plan §0 ablation arm")
    ap.add_argument("--baseline", action="store_true",
                    help="include the predict-the-mean floor in the leaderboard")
    ap.add_argument("--only", default=None,
                    help="comma-separated model names to sweep (default: all)")
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="parallel CV jobs passed through to cross_validate")
    ap.add_argument("--holdout-frac", type=float, default=HOLDOUT_FRACTION,
                    help="sealed test fraction held back from the sweep "
                         f"(default: {HOLDOUT_FRACTION})")
    ap.add_argument("--no-holdout", action="store_true",
                    help="sweep the whole sample, with neither the holdout nor "
                         "the temporal set sealed — smoke/wiring runs only, "
                         "never a run whose leaderboard you intend to act on")
    ap.add_argument("--write-train", action="store_true",
                    help="persist the train half of the sealed split to "
                         "spark/ml/data/ for the Spark MLlib baseline (§5b), "
                         "which cannot re-derive it")
    ap.add_argument("--tag", default=None,
                    help="output filename suffix (default: derived from options)")
    args = ap.parse_args()

    # A "train split" only means something opposite a sealed holdout; without
    # one the file would be the whole sample under a name promising otherwise.
    if args.write_train and args.no_holdout:
        raise SystemExit(
            "[error] --write-train needs a sealed holdout to be the train half "
            "OF something; --no-holdout makes the file the entire sample."
        )

    tag = args.tag or f"{args.sample}{'' if not args.no_duration else '_no_duration'}"

    df = load_sample(args.sample, args.limit_rows)

    # Seal the temporal test set first (plan §4a): every trip from the cutoff
    # on. It runs on the raw frame because build_features drops
    # pickup_datetime, and before make_holdout so the random 80/20 draws only
    # from the earlier months. Like the holdout, it is never scored here.
    n_temporal = 0
    if not args.no_holdout:
        n_sample = len(df)
        df, df_temporal = make_temporal_test(df)
        n_temporal = len(df_temporal)
        del df_temporal
        print(f"[split] temporal={n_temporal} rows, pickup >= "
              f"{TEMPORAL_CUTOFF:%Y-%m-%d} ({n_temporal / n_sample:.2%} of "
              f"{n_sample}) — SEALED")

    X, y = build_features(df, include_duration=not args.no_duration)
    print(f"[data] {args.sample} sample -> X={X.shape}, duration="
          f"{'ON' if not args.no_duration else 'OFF'}")

    # Seal the test set before anything else touches the data. The sweep, CV
    # and any later tuning see X_train only; the holdout is scored once in
    # Phase 5 after the champion is picked. It is deliberately NOT scored or
    # printed here — a test number visible on every sweep run is one you end
    # up selecting on.
    n_holdout = 0
    if not args.no_holdout:
        X, X_holdout, y, _ = make_holdout(X, y, test_size=args.holdout_frac)
        n_holdout = len(X_holdout)
        del X_holdout
        print(f"[split] train={len(X)} rows | holdout={n_holdout} rows "
              f"({args.holdout_frac:.0%}, stratified, seed={RANDOM_STATE}) — SEALED")
    else:
        print("[split] NO HOLDOUT — sweeping the full sample (smoke run)")

    # Hand the exact training rows to Spark. The filename comes from what
    # determines the contents, NOT from --tag — see train_split_filename.
    train_split_path = None
    if args.write_train:
        train_split_path = write_train_split(
            X,
            y,
            DATA_DIR / train_split_filename(
                args.sample,
                include_duration=not args.no_duration,
                limit_rows=args.limit_rows,
            ),
        )
        print(f"[write] train split -> {train_split_path} ({len(X)} rows)")

    specs = build_model_specs(include_baseline=args.baseline)
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        unknown = wanted - {s.name for s in specs}
        if unknown:
            raise SystemExit(f"[error] unknown model name(s): {sorted(unknown)}")
        specs = [s for s in specs if s.name in wanted]
    print(f"[sweep] {len(specs)} models x {args.folds} folds: "
          f"{', '.join(s.name for s in specs)}")

    started = time.time()
    board = run_sweep(specs, X, y, cv=make_cv(n_splits=args.folds), n_jobs=args.n_jobs)
    elapsed = time.time() - started

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    board_path = RESULTS_DIR / f"leaderboard_{tag}.csv"
    board.to_csv(board_path, index=False)

    meta = {
        "tag": tag,
        "sample": args.sample,
        "rows": int(len(X)),
        "features": list(X.columns),
        "include_duration": not args.no_duration,
        "folds": args.folds,
        "seed": RANDOM_STATE,
        # The split is reproducible from these three values + the sample file,
        # so Phase 5 can re-derive the identical sealed rows without a second
        # copy of the data on disk.
        "holdout_rows": n_holdout,
        "holdout_frac": 0.0 if args.no_holdout else args.holdout_frac,
        "holdout_stratified_on": None if args.no_holdout else list(STRATIFY_COLUMNS),
        # The temporal set is re-derivable from the cutoff + the sample file.
        # The random holdout above is drawn from the rows before the cutoff.
        "temporal_cutoff": None if args.no_holdout else TEMPORAL_CUTOFF.isoformat(),
        "temporal_rows": n_temporal,
        # Which file, if any, the Spark baseline should read to train on these
        # exact rows — so an MLlib leaderboard row is traceable to a split.
        "train_split_path": (
            str(train_split_path.relative_to(REPO_ROOT)) if train_split_path else None
        ),
        "models": [s.name for s in specs],
        "elapsed_s": round(elapsed, 1),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    meta_path = RESULTS_DIR / f"sweep_{tag}.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(board.to_string(index=False))
    print(f"[write] {board_path}")
    print(f"[write] {meta_path}")
    print(f"[time] {elapsed:.1f}s")
    print("SWEEP OK")


if __name__ == "__main__":
    main()
