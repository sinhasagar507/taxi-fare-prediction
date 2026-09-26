"""
Phase 4b — the Spark MLlib GBT baseline (plan §5b).

The DE-portfolio counterpart to the sklearn sweep: one Spark-native model,
trained on **the same rows the sweep trained on**, so the only difference
between its leaderboard row and sklearn's is the stack itself.

Reads `spark/ml/data/sample_work_train.parquet` — written by
`01_run_sweep.py --write-train`. That file is not a convenience: `make_holdout`
is `sklearn.train_test_split` and seeds do not transfer across libraries, so
Spark cannot re-derive those 612,608 rows at any seed. Reading the written
split is the only way the two stacks share a train set, and it keeps the sealed
153,153-row holdout untouched by every model in the project.

The features arrive already derived (the split was written from
`features.build_features` output), so this script does **no** feature
engineering — no second implementation of the cyclic-hour or temp-band
transforms to drift against the pandas ones. That changes if a future run reads
a raw prep sample instead; then the transforms come back and want a parity test.

Run (from repo root):

    # full baseline on the sweep's train split
    .venv/bin/python spark/ml/01_mllib_baseline.py

    # smoke: wiring check, small subset, 2 folds
    .venv/bin/python spark/ml/01_mllib_baseline.py --limit-rows 50000 --folds 2 \
        --max-iter 5 --tag mllib_smoke

    # row 3 of §5b: the corridor ablation, re-runnable
    .venv/bin/python spark/ml/01_mllib_baseline.py --drop-corridor \
        --tag mllib_gbt_nocorr

    # §5.4: the corridor encoded out-of-fold, outside MLlib (src/oof_encode.py)
    .venv/bin/python spark/ml/01_mllib_baseline.py --oof --tag mllib_gbt_oof

Run (Dataproc Serverless, migration plan M4):

    gcloud dataproc batches submit pyspark spark/ml/01_mllib_baseline.py -- \
        --cluster \
        --input  gs://<bucket>/ml/samples/sample_work_train.parquet \
        --output gs://<bucket>/ml/results

`--input` and `--output` take a local path or a bucket URI, and `--cluster`
swaps the local master for the Serverless one, `dataproc`. Those
three flags are the whole difference between the two invocations; nothing about
the model, the folds or the metrics changes with them.

Outputs (per run, keyed by --tag, under --output):
    leaderboard_<tag>.csv    one row, same shape as sklearn rows
    sweep_<tag>.json         run metadata + provenance,
                             same convention as 01_run_sweep.py

Design notes:
  - `od_corridor` is **target-encoded**, not dropped. The 2026-08-08 run of this
    script dropped it on the premise that "MLlib has no TargetEncoder", which is
    false here: `pyspark.ml.feature.TargetEncoder` arrived in Spark 4.0.0 and we
    run 4.0.1. That run stands as the corridor ablation (row 3 of §5b), still
    reproducible with `--drop-corridor`. The parity rule governs the baseline:
    whatever the sklearn sweep encodes, this encodes.
  - Spark's `TargetEncoder` does **not** cross-fit, so a training row's own fare
    enters its own feature. `--smoothing` is the only lever against that; the
    default and the arithmetic behind it live in `src/mllib.py`.
  - `--oof` replaces that encoder for migration plan §5.4. Each outer fold
    encodes its train half out-of-fold over 5 inner folds and its test fold
    from the whole train half (`src/oof_encode.encode_outer_fold`). The
    pipeline then has no StringIndexer or TargetEncoder on `od_corridor` and
    assembles `od_corridor_te` as a number. Same rows, folds and GBT settings.
  - Folds are k random groups at a fixed seed. They are NOT the sweep's folds:
    sklearn's `KFold(seed=42)` and anything Spark does at `seed=42` partition
    differently, because seeds do not cross library boundaries. Same k, same
    train pool, different membership — noise at the observed ~$0.002 fold SE,
    but not the equivalence an earlier draft of §5b claimed.
  - The holdout is never read here, let alone scored. It is scored once, in
    Phase 5, after the champion is chosen (§4a).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

# Import the unit-tested modules by their package path (repo root on sys.path).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.evaluation import RegressionEvaluator  # noqa: E402
from pyspark.ml.feature import (  # noqa: E402
    OneHotEncoder,
    SQLTransformer,
    StringIndexer,
    TargetEncoder,
    VectorAssembler,
)
from pyspark.ml.regression import GBTRegressor  # noqa: E402
from pyspark.sql import SparkSession, functions as F  # noqa: E402

from spark.ml.src.evaluate import RANDOM_STATE  # noqa: E402
from spark.ml.src.features import TARGET  # noqa: E402
from spark.ml.src.mllib import (  # noqa: E402
    DEFAULT_SMOOTHING,
    METRIC_KEYS,
    MLLIB_EXCLUDED_COLUMNS,
    fold_metrics_to_row,
    is_local_master,
    is_remote_uri,
    join_uri,
    model_name,
    self_leakage_weight,
    spark_master,
    split_column_groups,
    write_text,
)
from spark.ml.src.oof_encode import (  # noqa: E402
    DEFAULT_K as OOF_INNER_FOLDS,
    ENCODED_COL as OOF_ENCODED_COL,
    OUTER_FOLD_COL,
    encode_outer_fold,
)

DATA_DIR = REPO_ROOT / "spark" / "ml" / "data"
RESULTS_DIR = REPO_ROOT / "spark" / "ml" / "results"
# Strings, not Paths. `Path("gs://b/x")` collapses the duplicate separator into
# `gs:/b/x`, so a bucket URI cannot survive a round trip through pathlib — see
# src/mllib.is_remote_uri. The local defaults are stringified here so both
# invocations travel the same code path instead of only the laptop one.
DEFAULT_INPUT = str(DATA_DIR / "sample_work_train.parquet")
DEFAULT_OUTPUT = str(RESULTS_DIR)

# `oof_encode` reads the same column, so there is one name for it.
FOLD_COL = OUTER_FOLD_COL


def build_pipeline(
    categorical,
    target_encoded,
    numeric,
    max_iter: int,
    max_depth: int,
    smoothing: float,
) -> Pipeline:
    """The §5b pipeline: index, one-hot or target-encode, assemble, GBT.

    Numerics are assembled raw — GBT splits rather than scales, so a scaler
    buys nothing and would only make the model harder to reason about.

    Both encoders need a numeric index, so every categorical column is indexed
    first; only the *second* stage differs. `od_corridor` gets `TargetEncoder`
    because 19,953 levels is far past what one-hot can carry; boroughs and
    `service_type` get `OneHotEncoder` because they are closed three- and
    two-level sets.

    `targetType="continuous"` is not optional and not the default. Spark
    defaults to `"binary"`, which encodes a category as the conditional
    probability of the target — meaningless for a dollar fare, and it fails
    loudly rather than silently only because the label is not 0/1.

    `handleInvalid="keep"` sends an unseen category to its own bucket in the
    indexer, and to the dataset overall statistics in `TargetEncoder`. That
    second behaviour matches sklearn's fallback to the global target mean, so
    the corridor path needs no adjustment for parity — and it is a live path,
    not a theoretical one: ~1% of every CV test fold carries a corridor absent
    from that fold's train half. The one-hot mismatch does remain (sklearn's
    `handle_unknown="ignore"` gives all zeros), but boroughs and service_type
    are closed sets, so it should never fire.

    **The SQLTransformer is not cosmetic.** Spark's `TargetEncoderModel` copies
    the *indexer's* nominal `ml_attr` metadata onto its own output —
    listing the encoded fare means where the category labels used to be. The
    column holds continuous dollars, but `VectorAssembler` reads that metadata
    and marks the feature categorical, so the tree tries to bin it and dies:

        requirement failed: DecisionTree requires maxBins (= 32) to be at least
        as large as the number of values in each categorical feature, but
        categorical feature 26 has 1695 values

    Multiplying by 1.0 produces a fresh column with empty metadata, which is
    the cheapest way to say "this is a number". Raising `maxBins` instead would
    have made it *run* — and one-hot split a target encoding into thousands of
    buckets, which is not the feature this pipeline is supposed to have.
    """
    indexed = [*categorical, *target_encoded]
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in indexed
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe", dropLast=False)
        for c in categorical
    ]
    target_encoders = [
        TargetEncoder(
            inputCol=f"{c}_idx",
            outputCol=f"{c}_te_nominal",
            labelCol=TARGET,
            targetType="continuous",
            handleInvalid="keep",
            smoothing=smoothing,
        )
        for c in target_encoded
    ]
    # Strip the inherited nominal metadata — see the docstring. No stage at all
    # when nothing is target-encoded, so the ablation path stays identical to
    # the pipeline that produced the 2026-08-08 row.
    #
    # `SELECT *` deliberately keeps the intermediate `_idx` and `_te_nominal`
    # columns, whose schema carries ~19,000 category labels of `ml_attr`
    # metadata into every task binary. Projecting them away was measured on
    # 200,000 rows at maxIter=20 and changed nothing: 65.3s against 65.9s, the
    # same metrics to four decimal places. The cost of this pipeline is
    # target-encoding and boosting work, not metadata serialisation, so the
    # simpler statement stays.
    demote = (
        [
            SQLTransformer(
                statement="SELECT *, "
                + ", ".join(
                    f"{c}_te_nominal * 1.0 AS {c}_te" for c in target_encoded
                )
                + " FROM __THIS__"
            )
        ]
        if target_encoded
        else []
    )
    assembler = VectorAssembler(
        inputCols=[
            *numeric,
            *[f"{c}_ohe" for c in categorical],
            *[f"{c}_te" for c in target_encoded],
        ],
        outputCol="features",
    )
    gbt = GBTRegressor(
        labelCol=TARGET,
        featuresCol="features",
        maxIter=max_iter,
        maxDepth=max_depth,
        seed=RANDOM_STATE,
    )
    return Pipeline(
        stages=[*indexers, *encoders, *target_encoders, *demote, assembler, gbt]
    )


def input_label(input_uri: str) -> str:
    """What the run metadata records as its input.

    A repo-relative path when the file came from the working tree, so the
    record survives a clone at a different absolute path; the URI verbatim
    otherwise, because a bucket location has no repository to be relative to.
    """
    if is_remote_uri(input_uri):
        return input_uri
    try:
        return str(Path(input_uri).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return input_uri


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help="train split written by 01_run_sweep.py --write-train; "
                         "a local path or a gs:// URI")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="directory for leaderboard_<tag>.csv and "
                         "sweep_<tag>.json; a local path or a gs:// URI")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--max-iter", type=int, default=100,
                    help="GBT boosting rounds")
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--smoothing", type=float, default=DEFAULT_SMOOTHING,
                    help="TargetEncoder shrinkage toward the global mean; "
                         "bounds a single-trip corridor's self-leakage at "
                         "1/(1+smoothing). See src/mllib.self_leakage_weight.")
    ap.add_argument("--drop-corridor", action="store_true",
                    help="ablate od_corridor entirely — row 3 of plan §5b")
    ap.add_argument("--oof", action="store_true",
                    help="encode od_corridor out-of-fold outside MLlib, in "
                         "place of StringIndexer + TargetEncoder — migration "
                         "plan §5.4")
    ap.add_argument("--limit-rows", type=int, default=None,
                    help="seeded row subset for a fast wiring check")
    # Measured peak heap on the full 612,608-row split is 0.82 GiB with the
    # cached frame at 53 MB and no disk spill, so 2g is generous. A larger
    # ceiling only gives the JVM permission to grow into swap.
    ap.add_argument("--driver-memory", default="2g")
    # Not local[*]: leaving cores free keeps the machine usable for the ~10
    # minutes this takes, and the job is not CPU-starved at 8.
    ap.add_argument("--cores", type=int, default=8,
                    help="local Spark threads (default 8; 0 = all cores)")
    # Not a variant of --cores 0. A local master on Dataproc Serverless runs the
    # whole job inside the driver while every allocated executor sits idle — a
    # wrong answer that looks like a slow one. Leaving the master unset does
    # the same: the runtime's own default is `local` (mllib.spark_master).
    ap.add_argument("--cluster", action="store_true",
                    help="use the Dataproc Serverless master instead of "
                         "local[n] (plan M4)")
    ap.add_argument("--tag", default="mllib_gbt")
    args = ap.parse_args()

    if not is_remote_uri(args.input) and not Path(args.input).exists():
        raise SystemExit(
            f"[error] {args.input} not found — write it first with:\n"
            "        python spark/ml/01_run_sweep.py --write-train "
            "--only ridge --folds 2 --tag worksplit"
        )

    master = spark_master(None if args.cluster else args.cores)
    builder = SparkSession.builder.appName("mllib-gbt-baseline").master(master)
    if is_local_master(master):
        # Driver memory is a launch-time setting too, so it belongs with the
        # local master. On Serverless the runtime sizes the driver and this
        # value would be an inert but misleading claim in the code.
        builder = builder.config("spark.driver.memory", args.driver_memory)
    spark = (
        builder
        # 200 shuffle partitions is the cluster default and pure overhead at
        # this row count on any executor count we run here.
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        df = spark.read.parquet(args.input)
        if args.limit_rows is not None:
            # Plain head, not a re-sample: the file was written in
            # train_test_split's shuffled order, so the first n rows are
            # already an unbiased subset and an orderBy(rand()) would cost a
            # full shuffle to buy nothing.
            df = df.limit(args.limit_rows)

        feature_columns = [c for c in df.columns if c != TARGET]
        if args.oof:
            # The encoder adds this column inside the fold loop, per outer fold.
            feature_columns.append(OOF_ENCODED_COL)
        categorical, target_encoded, numeric, unrecognised = split_column_groups(
            feature_columns,
            drop_target_encoded=args.drop_corridor,
            oof_encoded=args.oof,
        )

        # Say what was dropped. A silent drop and a silent include are both
        # ways of not telling you; the deliberate choices are reported apart
        # from the anomalies so a documented finding is not buried in noise.
        print(f"[cols] categorical    ({len(categorical)}): {categorical}")
        print(f"[cols] target-encoded ({len(target_encoded)}): {target_encoded}")
        print(f"[cols] numeric        ({len(numeric)}): {numeric}")
        print(f"[cols] excluded by design: {list(MLLIB_EXCLUDED_COLUMNS)}")
        if args.drop_corridor:
            print("[cols] od_corridor ABLATED by --drop-corridor (plan §5b row 3)")
        if args.oof:
            print(f"[cols] od_corridor OUT-OF-FOLD encoded -> {OOF_ENCODED_COL}, "
                  f"{OOF_INNER_FOLDS} inner folds per outer train half "
                  "(migration plan §5.4)")
        if unrecognised:
            print(f"[cols] !! UNRECOGNISED, not used as features: {unrecognised}")

        # Assign folds once and pin them. rand() is only stable if the frame is
        # not recomputed, and every fold filter below would otherwise re-evaluate
        # it — silently reshuffling the split between folds.
        df = df.withColumn(
            FOLD_COL, (F.rand(seed=args.seed) * args.folds).cast("int")
        ).cache()
        n_rows = df.count()  # materialises the cache, fixing the fold assignment

        name = model_name(
            args.input, n_rows, drop_corridor=args.drop_corridor, oof=args.oof
        )
        print(f"[data] {args.input}: {n_rows:,} rows -> {name}")
        print(f"[spark] master={master or spark.sparkContext.master} "
              f"defaultParallelism={spark.sparkContext.defaultParallelism}")
        print(f"[cv]   {args.folds} folds, seed={args.seed}, "
              f"GBT maxIter={args.max_iter} maxDepth={args.max_depth}")
        if args.oof:
            print(f"[te]   out-of-fold, smoothing={args.smoothing}: a training "
                  "row never reads back its own fare")
        if target_encoded:
            # State the leakage bound the smoothing buys, at the point of use.
            # Spark's TargetEncoder does not cross-fit, so this number is the
            # honest residual difference against sklearn, not a formality.
            print(f"[te]   smoothing={args.smoothing}, a single-trip corridor "
                  f"reads back {self_leakage_weight(1, args.smoothing):.1%} of "
                  "its own fare")

        pipeline = build_pipeline(
            categorical,
            target_encoded,
            numeric,
            args.max_iter,
            args.max_depth,
            args.smoothing,
        )
        # The stage list is the evidence that an OOF run fitted no TargetEncoder.
        print("[cols] stages: "
              f"{[type(stage).__name__ for stage in pipeline.getStages()]}")
        evaluators = {
            key: RegressionEvaluator(
                labelCol=TARGET, predictionCol="prediction", metricName=key
            )
            for key in METRIC_KEYS
        }

        fold_metrics, fit_times, encode_times = [], [], []
        started = time.time()
        for fold in range(args.folds):
            encode_note = ""
            if args.oof:
                # Both halves are encoded from this fold's train half only, off
                # the cached `_fold` above. Cached and counted here so the
                # encoding runs once, not once per GBT pass and per evaluator,
                # and so its time is kept apart from the fit's.
                t0 = time.time()
                train, test = encode_outer_fold(
                    df, fold, fold_col=FOLD_COL, k=OOF_INNER_FOLDS,
                    smoothing=args.smoothing,
                )
                train, test = train.cache(), test.cache()
                train.count()
                test.count()
                encode_times.append(time.time() - t0)
                encode_note = f" encode={encode_times[-1]:.1f}s"
            else:
                train = df.filter(F.col(FOLD_COL) != fold).drop(FOLD_COL)
                test = df.filter(F.col(FOLD_COL) == fold).drop(FOLD_COL)

            t0 = time.time()
            model = pipeline.fit(train)
            fit_times.append(time.time() - t0)

            predictions = model.transform(test)
            metrics = {k: ev.evaluate(predictions) for k, ev in evaluators.items()}
            fold_metrics.append(metrics)
            print(f"[fold {fold}] mae={metrics['mae']:.4f} "
                  f"rmse={metrics['rmse']:.4f} r2={metrics['r2']:.4f} "
                  f"fit={fit_times[-1]:.1f}s{encode_note}")
            if args.oof:
                train.unpersist()
                test.unpersist()
        elapsed = time.time() - started

        row = fold_metrics_to_row(name, fold_metrics, fit_times)

        board_uri = join_uri(args.output, f"leaderboard_{args.tag}.csv")
        import pandas as pd  # local: keeps the Spark path pandas-free until now

        write_text(spark, board_uri, pd.DataFrame([row]).to_csv(index=False))

        meta = {
            "tag": args.tag,
            "model": name,
            "stack": "spark-mllib",
            "input": input_label(args.input),
            "output": args.output,
            "rows": n_rows,
            "features": {
                "categorical": categorical,
                "target_encoded": target_encoded,
                "numeric": numeric,
            },
            "target_encoding": {
                "smoothing": args.smoothing,
                "cross_fitted": False,
                # sklearn cross-fits inside fit_transform; Spark does not. This
                # is the residual difference §5b reports as a Tier-2 finding.
                "singleton_self_weight": self_leakage_weight(1, args.smoothing),
            } if target_encoded else {
                "smoothing": args.smoothing,
                "cross_fitted": True,
                "method": "out-of-fold, spark/ml/src/oof_encode.py",
                "inner_folds": OOF_INNER_FOLDS,
                "inner_fold_key": "crc32 over the feature columns",
                "encoded_column": OOF_ENCODED_COL,
                "singleton_self_weight": 0.0,
                "mean_encode_s": sum(encode_times) / len(encode_times),
            } if args.oof else None,
            "corridor_ablated": args.drop_corridor,
            "oof_encoded": args.oof,
            "excluded_by_design": list(MLLIB_EXCLUDED_COLUMNS),
            "unrecognised_columns": unrecognised,
            "folds": args.folds,
            "seed": args.seed,
            # Same k and same train pool as the sklearn sweep, but NOT the same
            # fold membership — seeds do not transfer across libraries.
            "folds_match_sklearn_membership": False,
            "gbt": {"maxIter": args.max_iter, "maxDepth": args.max_depth},
            "elapsed_s": round(elapsed, 1),
            "spark": spark.version,
            # Recorded because the cloud run and the laptop run differ here and
            # nowhere else that matters: defaultParallelism fixes the read
            # partitioning, which fixes fold membership. Two runs of this script
            # agree to full precision only when this number agrees.
            "master": spark.sparkContext.master,
            "default_parallelism": spark.sparkContext.defaultParallelism,
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        meta_uri = join_uri(args.output, f"sweep_{args.tag}.json")
        write_text(spark, meta_uri, json.dumps(meta, indent=2))

        print(pd.DataFrame([row]).to_string(index=False))
        print(f"[write] {board_uri}")
        print(f"[write] {meta_uri}")
        print(f"[time] {elapsed:.1f}s")
        print("MLLIB BASELINE OK")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
