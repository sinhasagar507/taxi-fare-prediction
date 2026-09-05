"""Pure helpers for the Phase-4b Spark MLlib GBT baseline (plan §5b).

Deliberately free of any `pyspark` import. The fit itself is a script run
(01_mllib_baseline.py) like Phase 1, but the decisions *around* the fit — which
columns get one-hot encoded, which get target-encoded, which pass through raw,
how hard the encoder is smoothed, and how a Spark result becomes a leaderboard
row — are ordinary data transformations, so they live
here and are unit-tested on the host venv without a SparkSession.

Three contracts matter:

  - `od_corridor` is **target-encoded**, in its own column group. An earlier
    version of this module dropped it and justified the drop with "MLlib has no
    TargetEncoder". That was false for this project: `TargetEncoder` arrived in
    Spark 4.0.0 and we run 4.0.1, so the 2026-08-08 baseline shipped without a
    feature it could have had. The parity rule of §5b now governs — whatever the
    sklearn sweep encodes, MLlib encodes, because a gap that comes from a
    missing feature measures nothing anyone wants to know. The corridor-dropped
    run survives as the deliberate ablation behind `drop_target_encoded=True`.
  - Column groups are **allowlists**, mirroring
    `preprocess.build_preprocessor`'s `remainder="drop"`. Both stacks must
    discard the same columns or they are not training on the same feature set,
    and the comparison §5b exists to make stops meaning anything. An earlier
    catch-all ("numeric is everything not categorical") sent any unexpected
    column straight into the VectorAssembler — a stray pandas index would have
    become a model feature, and a re-added `fare_amount` would have been
    trained on while sklearn silently dropped it.
  - Dropped columns are **named, not swallowed**. A silent drop and a silent
    include are both ways of not telling you, so the unrecognised names come
    back for the script to print. Deliberate choices are reported separately
    from anomalies: an ablated corridor is a result the project publishes, and
    listing it among the surprises would bury a real surprise next to it.

Rows produced here are the same shape as `evaluate.evaluate()` rows, so the
Spark baseline sorts into the one leaderboard next to the sklearn sweep instead
of living in a parallel table.
"""

from pathlib import Path

import numpy as np

from .features import CATEGORICAL_COLUMNS
from .preprocess import BINARY_COLUMNS, NUMERIC_COLUMNS, TARGET_ENCODED_COLUMNS

# Nothing is dropped by design any more — the 2026-08-09 correction emptied
# this. The constant stays so the run metadata keeps its field and a future
# exclusion has one obvious home instead of being an omission in the script.
MLLIB_EXCLUDED_COLUMNS: tuple[str, ...] = ()

# Target-encoded rather than one-hot: 19,953 levels. Derived from the sklearn
# list so the two stacks cannot drift into encoding different columns — that is
# the §5b parity rule, and it is the whole basis of the comparison.
MLLIB_TARGET_ENCODED_COLUMNS = tuple(TARGET_ENCODED_COLUMNS)

# The low-cardinality categoricals MLlib can afford to one-hot: the Phase-2
# categorical list minus the target-encoded corridor. Derived from
# features.CATEGORICAL_COLUMNS so the two cannot drift apart.
MLLIB_CATEGORICAL_COLUMNS = tuple(
    c
    for c in CATEGORICAL_COLUMNS
    if c not in MLLIB_TARGET_ENCODED_COLUMNS and c not in MLLIB_EXCLUDED_COLUMNS
)

# Spark's TargetEncoder does not cross-fit, so a training row's own target
# enters its own feature. `smoothing` is the only lever against that, and
# `self_leakage_weight` below is the arithmetic.
#
# This was 20 first, from the bound: 20 holds a single-trip corridor's
# self-weight to 1/21 = 4.8% while leaving a 1,000-trip corridor at 98%. The
# 2026-09-01 sweep overruled it. Seven arms on identical rows (200,000 rows,
# 3 folds, maxIter=20), mean MAE:
#
#     dropped 0.6329 | s=5 0.6650 | s=1 0.6713 | s=0.067 0.6908
#             | s=100 0.7094 | s=20 0.7102 | s=500 0.7236
#
# s=5 is the best encoded setting, so it is the default — even though it lets a
# singleton read back 16.7% of its own fare. Note what the first column says:
# **no smoothing beat dropping the corridor**, so this is the best of a losing
# set. Plan §5b carries the reading.
DEFAULT_SMOOTHING = 5.0

# Everything that goes into the VectorAssembler unscaled. sklearn keeps numerics
# and binaries apart because StandardScaler should not touch a 0/1 flag; GBT
# splits rather than scales, so the distinction is meaningless here and the two
# lists are unioned. Derived, not retyped, for the same reason as above —
# and the union is stable if a column ever moves between the two sklearn lists.
MLLIB_NUMERIC_COLUMNS = tuple(dict.fromkeys([*NUMERIC_COLUMNS, *BINARY_COLUMNS]))

# The three metrics every model in this project reports (evaluate.compute_metrics).
METRIC_KEYS = ("mae", "rmse", "r2")

# Schemes Spark opens through a Hadoop FileSystem rather than the local disk.
# An explicit tuple, not a regex over "anything://": an unlisted or misspelled
# scheme should read as a local path and fail with a missing file, which names
# the problem, rather than reach Spark as a filesystem nobody configured.
REMOTE_URI_SCHEMES: tuple[str, ...] = (
    "gs://",
    "s3://",
    "s3a://",
    "hdfs://",
    "abfss://",
    "wasbs://",
)


def is_remote_uri(path) -> bool:
    """True when Spark must open `path` through a bucket filesystem.

    The script needs this because `pathlib` cannot carry a bucket URI:
    `Path("gs://b/x")` collapses the duplicate separator and stringifies back
    to `gs:/b/x`, which Spark reads as a relative path. So the input and output
    locations stay strings, and the three jobs `Path` used to do — classify,
    stem, join — are the three functions below.
    """
    return str(path).startswith(REMOTE_URI_SCHEMES)


def uri_stem(path) -> str:
    """The filename stem of a local path or a bucket URI, matching `Path.stem`.

    `model_name()` builds the leaderboard label from this, so a cloud run and a
    laptop run of the same file have to produce the same label. Otherwise one
    input lands in the leaderboard twice under two names, and the table stops
    meaning what it says.

    A trailing slash is ignored, because a partitioned parquet *directory* is an
    ordinary Spark input and must not stem to the empty string.
    """
    name = str(path).rstrip("/").rsplit("/", 1)[-1]
    base, dot, _suffix = name.rpartition(".")
    # `base` is empty for a dotfile (".bashrc"), where the dot is not a suffix
    # separator. `Path.stem` keeps the whole name there; so does this.
    return base if (dot and base) else name


def join_uri(base, name: str) -> str:
    """Join an output directory to a filename, for a local path or a bucket URI.

    `os.path.join` would serve for the local half only; this keeps one code
    path. `rstrip` handles a caller who supplied the trailing slash without
    producing the doubled separator that a bucket treats as a real empty path
    segment.
    """
    return f"{str(base).rstrip('/')}/{name}"


def spark_master(cores):
    """The `.master()` argument, or None meaning "do not set one".

    None is the cluster case: on Dataproc Serverless the master arrives in the
    environment, and pinning `local[n]` there would run the whole job inside
    the driver and silently ignore every executor allocated to it.

    For a local run `cores` is not a performance knob. It fixes
    `defaultParallelism`, which fixes the read partitioning, which fixes fold
    membership — the reason the 2026-09-02 parity run reproduced to full float64
    precision instead of only to fold noise. 0 means every core.
    """
    if cores is None:
        return None
    if cores < 0:
        raise ValueError(
            f"cores must be non-negative or None, got {cores} — Spark accepts "
            "local[-1] and then fails deep inside the scheduler"
        )
    return "local[*]" if cores == 0 else f"local[{cores}]"


def write_text(spark, uri, text: str) -> None:
    """Write one small text artifact to a local path or a bucket URI.

    Used by both M4 scripts — `01_mllib_baseline.py` for the leaderboard row and
    the run metadata, `00_prep_spark.py` for `prep_stats.json`.

    `spark` is a parameter rather than an import, so this module keeps the
    no-pyspark contract in its docstring and the local branch needs no
    SparkSession at all: pass None for a local write.

    Local goes through pathlib. Remote goes through the JVM's Hadoop
    FileSystem, which every Spark deployment already has configured, for two
    reasons. Adding `gcsfs` so pandas could write `gs://` would put a Python
    dependency on the managed runtime that we have not verified is there. And
    `spark.write` would turn one named file into a directory of part files,
    changing the artifact shape `evaluate.leaderboard()` reads — the cloud run
    has to produce the same files as the laptop run, or the comparison M4
    exists to make is comparing two different things.
    """
    if not is_remote_uri(uri):
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return

    jvm = spark.sparkContext._jvm
    hadoop_path = jvm.org.apache.hadoop.fs.Path(uri)
    fs = hadoop_path.getFileSystem(spark.sparkContext._jsc.hadoopConfiguration())
    stream = fs.create(hadoop_path, True)  # True = overwrite
    try:
        stream.write(bytearray(text.encode("utf-8")))
    finally:
        stream.close()


def split_column_groups(
    columns, drop_target_encoded: bool = False
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split a feature frame's columns into (categorical, target_encoded, numeric, unrecognised).

    Categoricals go through StringIndexer + OneHotEncoder. The target-encoded
    group goes through StringIndexer + TargetEncoder — 19,953 corridors are far
    past what one-hot can carry, and dropping the column instead is what the
    2026-08-09 correction reversed. Numerics pass straight into the
    VectorAssembler, unscaled — GBT is a tree, so scaling buys nothing.
    **Unrecognised columns go nowhere near the model**; they are returned so the
    caller can say what it ignored.

    `drop_target_encoded=True` reproduces row 3 of §5b, the corridor ablation.
    It is a flag rather than an old commit because the ablation is a result the
    project reports, and a reported result has to stay re-runnable.

    All four lists come back **sorted**, not in DataFrame order.
    VectorAssembler input order is what fixes feature-importance indices, so
    deriving it from however the caller's columns happen to be arranged would
    silently relabel importances when an upstream frame is rebuilt in a
    different order. `unrecognised` is sorted so a warning reads the same twice.

    Absent is not the same as unknown. Columns that simply aren't there are
    absent from the result and reported nowhere, so reduced frames (smoke runs,
    the duration ablation) need no special-casing and produce no warning — a
    warning that fires on every ordinary run is one people stop reading.
    Likewise a deliberately ablated corridor is not called unrecognised: that
    drop is a documented §5b result, not a surprise.
    """
    kept = [c for c in columns if c not in MLLIB_EXCLUDED_COLUMNS]
    categorical = sorted(c for c in kept if c in MLLIB_CATEGORICAL_COLUMNS)
    target_encoded = (
        []
        if drop_target_encoded
        else sorted(c for c in kept if c in MLLIB_TARGET_ENCODED_COLUMNS)
    )
    numeric = sorted(c for c in kept if c in MLLIB_NUMERIC_COLUMNS)
    known = (
        set(MLLIB_CATEGORICAL_COLUMNS)
        | set(MLLIB_TARGET_ENCODED_COLUMNS)
        | set(MLLIB_NUMERIC_COLUMNS)
    )
    unrecognised = sorted(c for c in kept if c not in known)
    return categorical, target_encoded, numeric, unrecognised


def self_leakage_weight(category_size: int, smoothing: float) -> float:
    """How much of its own target one row reads back through the encoding.

    Spark encodes a category of size `n` as
    `(n*category_mean + smoothing*global_mean) / (n + smoothing)`. One row
    contributes `1/n` of `category_mean`, so its own target carries weight
    `1 / (n + smoothing)`.

    The formula is here, rather than in a comment on the constant, because it is
    the argument for `DEFAULT_SMOOTHING`: at `smoothing=0` a single-trip
    corridor reads back 100% of its own fare, and 5,373 of the 18,668 corridors
    in the train split are single-trip. A number chosen this way can be checked;
    a number chosen by taste cannot.

    This bounds *training-half* leakage only. Both stacks are already safe
    against test-fold leakage, because the pipeline is fitted on each fold's
    train half alone. The residual biases the Spark score **down**, not up.
    """
    if category_size < 1:
        raise ValueError(
            f"category_size must be at least 1, got {category_size} — an empty "
            "category has no mean to encode"
        )
    if smoothing < 0:
        raise ValueError(
            f"smoothing must be non-negative, got {smoothing} — Spark accepts a "
            "negative value and produces nonsense"
        )
    return 1.0 / (category_size + smoothing)


def fold_metrics_to_row(name: str, fold_metrics, fit_times) -> dict:
    """Fold-wise RegressionEvaluator results -> one leaderboard row.

    `fold_metrics` is one dict per fold with mae/rmse/r2 keys (the
    `evaluate.compute_metrics` shape); `fit_times` is the matching wall time
    per fold.

    Spread is population std (ddof=0) because `evaluate()` takes
    `ndarray.std()`, whose default is ddof=0. Using a sample std here would
    make the Spark row look systematically tighter or looser than every
    sklearn row sharing the table, for no reason a reader could see.
    """
    fold_metrics = list(fold_metrics)
    fit_times = list(fit_times)

    if not fold_metrics:
        raise ValueError("fold_metrics is empty — nothing to summarise")
    if len(fold_metrics) != len(fit_times):
        raise ValueError(
            f"got {len(fold_metrics)} fold metrics but {len(fit_times)} fit "
            "times — zipping these would silently drop a fold"
        )

    row = {"model": name}
    for key in METRIC_KEYS:
        values = np.array([fold[key] for fold in fold_metrics], dtype=float)
        row[f"{key}_mean"] = float(values.mean())
        row[f"{key}_std"] = float(values.std())
    row["fit_time_s"] = float(np.mean(fit_times))
    return row
