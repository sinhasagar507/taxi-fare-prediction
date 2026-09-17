"""
Phase 5 — tune the two leading boosters on the work train split, then diagnose
the champion (plan §6).

Two phases, two invocations, so a failure in the diagnostics never costs the
tuning run:

    # tune lightgbm and catboost; pick the champion by the owner's rule
    docker compose -f docker/dev/docker-compose.yml run --rm dev \
        python spark/ml/02_tune.py --phase tune

    # out-of-fold slices, SHAP and the duration ablation for that champion
    docker compose -f docker/dev/docker-compose.yml run --rm dev \
        python spark/ml/02_tune.py --phase diagnose

    # re-tune one model and reuse the other's result from an earlier run
    ... python spark/ml/02_tune.py --phase tune --models catboost \
        --reuse tune_d012 --tag tune_d012b

    # wiring smoke: a small row subset, 3 folds, 2 trials, nothing checked
    ... python spark/ml/02_tune.py --phase tune --smoke --tag tune_smoke

Outputs, all under spark/ml/results/ (gitignored), keyed by --tag:
    leaderboard_<tag>.csv          default and tuned rows for both models
    <tag>.json                     trial 0, best params, trials, elapsed, champion
    diagnose_<tag>.json            OOF slices, SHAP ranking, duration gap
    diagnose_<tag>_slice_<col>.csv one table per slice column

Design notes:
  - The carve is 01_run_sweep.py's, step for step: temporal set first on the
    raw frame, then build_features, then make_holdout. The counts are checked
    against the 2026-09-13 baseline, and the train target is checked value by
    value against sample_work_train.parquet, before any model is fitted.
  - The holdout and the temporal test set are deleted as soon as they are
    carved. Nothing here scores them; Phase 5 scores them once, later.
  - Tuning selects on the same 5 folds it reports, so a tuned CV MAE is mildly
    optimistic. The sealed holdout is the unbiased number.
  - Diagnostics use out-of-fold predictions from the same make_cv(5) folds, so
    every slice MAE is a train-split CV number, never a test number.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spark.ml.src.evaluate import (  # noqa: E402
    HOLDOUT_FRACTION,
    RANDOM_STATE,
    compute_metrics,
    evaluate,
    make_cv,
    make_holdout,
    make_temporal_test,
)
from spark.ml.src.features import TARGET, build_features  # noqa: E402
from spark.ml.src.tune import (  # noqa: E402
    TUNED_MODELS,
    build_tuned_pipeline,
    default_params,
    hour_from_cyclic,
    pick_champion,
    run_study,
    slice_mae,
)

DATA_DIR = REPO_ROOT / "spark" / "ml" / "data"
RESULTS_DIR = REPO_ROOT / "spark" / "ml" / "results"
SAMPLE_PATH = DATA_DIR / "sample_work.parquet"
TRAIN_SPLIT_PATH = DATA_DIR / "sample_work_train.parquet"
SWEEP_BOARD = RESULTS_DIR / "leaderboard_work_d012.csv"

# The 2026-09-13 baseline, re-measured when the TUNE goal was written.
EXPECTED_ROWS = {"sample": 1_828_181, "temporal": 133_629, "holdout": 338_911, "train": 1_355_641}
METRIC_KEYS = ("mae_mean", "mae_std", "rmse_mean", "rmse_std", "r2_mean", "r2_std", "fit_time_s")
SHAP_ROWS = 20_000


def carve(include_duration: bool, limit_rows: int | None, check: bool):
    """Return the train split (X, y) exactly as 01_run_sweep.py carves it."""
    df = pd.read_parquet(SAMPLE_PATH)
    if limit_rows is not None:
        df = df.sample(n=limit_rows, random_state=RANDOM_STATE).reset_index(drop=True)
    n_sample = len(df)
    df, df_temporal = make_temporal_test(df)
    n_temporal = len(df_temporal)
    del df_temporal
    X, y = build_features(df, include_duration=include_duration)
    X, X_holdout, y, _ = make_holdout(X, y, test_size=HOLDOUT_FRACTION)
    n_holdout = len(X_holdout)
    del X_holdout
    counts = {"sample": n_sample, "temporal": n_temporal, "holdout": n_holdout, "train": len(X)}
    print(f"[split] {counts} — temporal and holdout SEALED, deleted")
    if check:
        if counts != EXPECTED_ROWS:
            raise SystemExit(f"[stop] row counts differ from FACTS: {counts} != {EXPECTED_ROWS}")
        train_target = pd.read_parquet(TRAIN_SPLIT_PATH, columns=[TARGET])[TARGET]
        if not np.array_equal(y.to_numpy(), train_target.to_numpy()):
            raise SystemExit("[stop] the carved train target differs from sample_work_train.parquet")
        print("[split] counts equal FACTS; train target equals sample_work_train.parquet")
    return X, y


def trial_row(trial) -> dict:
    row = {"mae_mean": float(trial.value)}
    row.update({k: float(trial.user_attrs[k]) for k in METRIC_KEYS if k != "mae_mean"})
    return row


def phase_tune(args) -> None:
    X, y = carve(True, args.limit_rows, check=not args.smoke)
    cv = make_cv(n_splits=args.folds)
    sweep = pd.read_csv(SWEEP_BOARD).set_index("model") if SWEEP_BOARD.exists() else None

    report, board, best_rows = {}, [], {}
    for name in args.models:
        print(f"[tune] {name}: timeout {args.timeout_min} min, trial 0 = defaults "
              f"{default_params(name)}")
        started = time.time()
        study = run_study(name, X, y, cv, n_trials=args.n_trials,
                          timeout=args.timeout_min * 60)
        elapsed = time.time() - started

        trial0, best = study.trials[0], study.best_trial
        row0, best_row = trial_row(trial0), trial_row(best)
        best_rows[name] = best_row
        entry = {
            "trials": len(study.trials),
            "elapsed_s": round(elapsed, 1),
            "timeout_s": args.timeout_min * 60,
            "trial0": {"params": trial0.params, **row0},
            "best": {"number": best.number, "params": best.params, **best_row},
            "trial_mae": [None if t.value is None else float(t.value) for t in study.trials],
        }
        if sweep is not None and name in sweep.index and not args.smoke:
            ref = sweep.loc[name]
            diff = row0["mae_mean"] - float(ref["mae_mean"])
            entry["trial0_vs_work_d012"] = {
                "work_d012_mae_mean": float(ref["mae_mean"]),
                "work_d012_mae_std": float(ref["mae_std"]),
                "difference": diff,
                "inside_std": bool(abs(diff) <= float(ref["mae_std"])),
            }
            print(f"[tune] {name} trial 0 mae {row0['mae_mean']:.6f} vs work_d012 "
                  f"{float(ref['mae_mean']):.6f}: diff {diff:+.3e}, std {float(ref['mae_std']):.6f}")
        report[name] = entry
        board += [{"model": f"{name}@default", **row0}, {"model": f"{name}@tuned", **best_row}]
        print(f"[tune] {name}: {len(study.trials)} trials in {elapsed:.1f}s; best #{best.number} "
              f"mae {best_row['mae_mean']:.6f} ± {best_row['mae_std']:.6f} {best.params}")

    # A model left out of --models keeps its record from the --reuse run, so
    # the champion is still picked between both. The record says so.
    for name in TUNED_MODELS:
        if name in args.models:
            continue
        if args.reuse is None:
            raise SystemExit(f"[error] {name} was not tuned here; pass --reuse <tag>")
        entry = dict(json.loads((RESULTS_DIR / f"{args.reuse}.json").read_text())["models"][name])
        entry["reused_from"] = args.reuse
        report[name] = entry
        best_rows[name] = {k: entry["best"][k] for k in METRIC_KEYS}
        board += [{"model": f"{name}@default", **{k: entry["trial0"][k] for k in METRIC_KEYS}},
                  {"model": f"{name}@tuned", **best_rows[name]}]
        print(f"[tune] {name}: reused from {args.reuse} — {entry['trials']} trials, "
              f"best mae {best_rows[name]['mae_mean']:.6f}")

    champion = pick_champion(best_rows)
    print(f"[champion] {champion}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(board).sort_values("mae_mean").to_csv(
        RESULTS_DIR / f"leaderboard_{args.tag}.csv", index=False)
    meta = {
        "tag": args.tag, "rows": int(len(X)), "folds": args.folds, "seed": RANDOM_STATE,
        "metric": "mae", "sampler": "TPESampler(seed=42)", "models": report,
        "champion": champion, "features": list(X.columns),
        "holdout_scored": False, "temporal_scored": False,
        "python": platform.python_version(), "platform": platform.platform(),
    }
    (RESULTS_DIR / f"{args.tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"[write] {RESULTS_DIR / f'leaderboard_{args.tag}.csv'}")
    print(f"[write] {RESULTS_DIR / f'{args.tag}.json'}")
    print("TUNE OK")


def phase_diagnose(args) -> None:
    from sklearn.model_selection import cross_val_predict

    meta = json.loads((RESULTS_DIR / f"{args.tag}.json").read_text())
    name = meta["champion"]["champion"]
    params = meta["models"][name]["best"]["params"]
    print(f"[diagnose] champion {name} {params}")

    X, y = carve(True, args.limit_rows, check=not args.smoke)
    cv = make_cv(n_splits=args.folds)
    out = {"tag": args.tag, "champion": name, "params": params, "rows": int(len(X))}

    started = time.time()
    oof = cross_val_predict(build_tuned_pipeline(name, params, X.columns), X, y, cv=cv)
    out["oof_overall"] = compute_metrics(y, oof)
    out["oof_elapsed_s"] = round(time.time() - started, 1)
    print(f"[oof] {out['oof_overall']} in {out['oof_elapsed_s']}s")

    slices = {
        "pickup_borough": X["pickup_borough"].rename("pickup_borough"),
        "temp_band_ord": X["temp_band_ord"].rename("temp_band_ord"),
        "hour": pd.Series(hour_from_cyclic(X["pickup_hour_sin"], X["pickup_hour_cos"]),
                          index=X.index, name="hour"),
    }
    out["slices"] = {}
    for col, groups in slices.items():
        table = slice_mae(y, oof, groups)
        table.to_csv(RESULTS_DIR / f"diagnose_{args.tag}_slice_{col}.csv", index=False)
        out["slices"][col] = table.to_dict(orient="records")
        print(f"[slice] {col}\n{table.to_string(index=False)}")

    import shap

    started = time.time()
    fitted = build_tuned_pipeline(name, params, X.columns).fit(X, y)
    sample = X.sample(n=min(SHAP_ROWS, len(X)), random_state=RANDOM_STATE)
    matrix = fitted.named_steps["pre"].transform(sample)
    names = list(fitted.named_steps["pre"].get_feature_names_out())
    values = shap.TreeExplainer(fitted.named_steps["model"]).shap_values(matrix)
    ranking = (pd.Series(np.abs(np.asarray(values)).mean(axis=0), index=names)
               .sort_values(ascending=False))
    out["shap"] = {"rows": int(len(sample)), "mean_abs": ranking.round(6).to_dict(),
                   "elapsed_s": round(time.time() - started, 1)}
    print(f"[shap] mean |SHAP| on {len(sample)} rows\n{ranking.to_string()}")

    X_nd, y_nd = carve(False, args.limit_rows, check=False)
    if not (X_nd.index.equals(X.index) and np.array_equal(y_nd.to_numpy(), y.to_numpy())):
        raise SystemExit("[stop] the duration-off carve selected different train rows")
    with_row = {k: float(v) for k, v in evaluate(
        build_tuned_pipeline(name, params, X.columns), X, y, cv=cv, name=name).items() if k != "model"}
    off_row = {k: float(v) for k, v in evaluate(
        build_tuned_pipeline(name, params, X_nd.columns), X_nd, y_nd, cv=cv, name=name).items() if k != "model"}
    out["duration_ablation"] = {
        "on": with_row, "off": off_row,
        "mae_gap_off_minus_on": off_row["mae_mean"] - with_row["mae_mean"],
        "note": "same rows, folds and tuned params; params were tuned with duration on",
    }
    print(f"[ablation] on {with_row['mae_mean']:.6f} ± {with_row['mae_std']:.6f}; "
          f"off {off_row['mae_mean']:.6f} ± {off_row['mae_std']:.6f}; "
          f"gap {out['duration_ablation']['mae_gap_off_minus_on']:+.6f}")

    out["holdout_scored"] = False
    out["temporal_scored"] = False
    (RESULTS_DIR / f"diagnose_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[write] {RESULTS_DIR / f'diagnose_{args.tag}.json'}")
    print("DIAGNOSE OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("tune", "diagnose"), required=True)
    ap.add_argument("--tag", default="tune_d012")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--timeout-min", type=float, default=45.0)
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--limit-rows", type=int, default=None)
    ap.add_argument("--models", default=",".join(TUNED_MODELS),
                    help="comma-separated subset of the tuned models to search")
    ap.add_argument("--reuse", default=None,
                    help="tag of an earlier run that supplies the models not searched")
    ap.add_argument("--smoke", action="store_true",
                    help="wiring check: skips the FACTS checks; pair with --limit-rows")
    args = ap.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(args.models) - set(TUNED_MODELS)
    if unknown:
        raise SystemExit(f"[error] unknown model(s): {sorted(unknown)}")
    if args.reuse == args.tag:
        raise SystemExit("[error] --reuse and --tag must differ, or the earlier record is overwritten")
    if args.smoke and args.tag == "tune_d012":
        raise SystemExit("[error] a smoke run must not write the tune_d012 outputs; pass --tag")
    (phase_tune if args.phase == "tune" else phase_diagnose)(args)


if __name__ == "__main__":
    main()
