"""The out-of-fold target encoder for the MLlib GBT baseline (migration plan §5.3).

Why this module exists. Spark's `TargetEncoder.fit` takes the plain
per-category mean, so a training row's own fare enters its own feature.
sklearn's `TargetEncoder` cross-fits inside `fit_transform`, so it does not.
On the D-012 train split (1,355,641 rows, measured 2026-09-15) 5,617 of the
21,332 corridors hold exactly one trip, and for those rows an uncross-fitted
encoding hands the label back, shrunk only by the smoothing. This module
computes the encoding outside MLlib, cross-fitted, and returns it as a plain
double column, so the §5.4 acceptance run can say whether the encoder was the
whole story behind the corridor's net-negative effect.

Four functions:

  - `add_oof_fold` assigns each row an inner fold from a crc32 over its own
    feature values. `monotonically_increasing_id()` and `rand()` both depend on
    partitioning, so a frame read back a second time could land in different
    folds, and the out-of-fold guarantee would stop holding without an error.
  - `oof_target_encode` is the training-half encoder. For each inner fold it
    fits on the other folds and applies to this one, then unions the k frames.
  - `fit_target_encoding` and `apply_target_encoding` are the transform-time
    pair. A test fold, the holdout or a prediction frame is encoded with the
    whole training half's means and no fold exclusion, which is what sklearn's
    `transform()` does.

The smoothed encoding is `(n*mean + smoothing*global_mean) / (n + smoothing)`,
the formula `mllib.self_leakage_weight` documents. `global_mean` is taken from
the rows the encoding was fitted on, never from the whole frame: the whole
frame's mean would carry the held-out rows' own fares back in.

The fold key, and why it is not `tripid`
----------------------------------------
§5.3 item 1 asked for a crc32 over `tripid`. The D-012 train split carries no
`tripid` (the prep drops it), and adding it means a new split, which would void
row 2 (`mllib_gbt@work1355k`) and the M4 smoke that reproduced it. The owner
chose a crc32 over the feature columns instead (decision 1, 2026-09-15). It is
as stable as `tripid` across re-materialisation and sessions, which is the one
reason §5.3 gave for wanting `tripid`.

It is not a unique row key. On the D-012 split 11 pairs of rows share every
feature value (measured 2026-09-15; the largest group is 2). That is safe: rows
that share a key share a fold, so each is still encoded from the other folds
only, and a row never reads back its own fare.

The key leaves out the fare, the outer `_fold`, the encoder's output and any
earlier `oof_fold`. A key over the fare would move rows between inner folds when
a fare changed; a key over `_fold` would give each outer training half its own
inner split. The key casts each value to a string, and the train split holds no
nulls (measured 2026-09-15), so `concat_ws` skipping a null never fires on it.
Whether Spark 4.0.1 and 4.1.2 cast a double to the same string is UNVERIFIED;
every §5.4 run uses 4.0.1 so the question stays out of the comparison.
"""

from dataclasses import dataclass
from functools import reduce

from pyspark.sql import DataFrame, functions as F

from .features import TARGET
from .mllib import DEFAULT_SMOOTHING

KEY_COL = "od_corridor"
ENCODED_COL = f"{KEY_COL}_te"
OOF_FOLD_COL = "oof_fold"
# The outer CV fold column `01_mllib_baseline.py` assigns.
OUTER_FOLD_COL = "_fold"
DEFAULT_K = 5

# A separator no feature value contains, so ("a", "bc") and ("ab", "c") hash
# differently.
_KEY_SEPARATOR = "\x1f"


@dataclass(frozen=True)
class TargetEncoding:
    """A fitted encoding: one row per key, plus the fallback for unseen keys."""

    table: DataFrame
    global_mean: float
    key_col: str
    out_col: str


def fold_key_columns(columns, target_col: str = TARGET) -> list[str]:
    """The columns the inner fold key hashes: the features, in frame order."""
    excluded = {target_col, OUTER_FOLD_COL, ENCODED_COL, OOF_FOLD_COL}
    return [c for c in columns if c not in excluded]


def add_oof_fold(
    df: DataFrame,
    k: int = DEFAULT_K,
    fold_col: str = OOF_FOLD_COL,
    target_col: str = TARGET,
) -> DataFrame:
    """Add `fold_col` in [0, k), from a crc32 over the row's feature values.

    `pmod`, not `%`: Spark's `%` keeps the sign of a negative input, and a
    negative fold index names a fold that does not exist.
    """
    key = F.concat_ws(
        _KEY_SEPARATOR,
        *[F.col(c).cast("string") for c in fold_key_columns(df.columns, target_col)],
    )
    return df.withColumn(fold_col, F.pmod(F.crc32(key), F.lit(k)).cast("int"))


def fit_target_encoding(
    df: DataFrame,
    key_col: str = KEY_COL,
    target_col: str = TARGET,
    smoothing: float = DEFAULT_SMOOTHING,
    out_col: str | None = None,
) -> TargetEncoding:
    """Fit the smoothed per-key means on `df`, and its global mean."""
    global_mean = float(df.agg(F.avg(target_col)).first()[0])
    table = (
        df.groupBy(key_col)
        .agg(F.sum(target_col).alias("_sum"), F.count(F.lit(1)).alias("_n"))
        .select(
            key_col,
            (
                (F.col("_sum") + F.lit(smoothing * global_mean))
                / (F.col("_n") + F.lit(smoothing))
            ).alias(out_col or f"{key_col}_te"),
        )
    )
    return TargetEncoding(table, global_mean, key_col, out_col or f"{key_col}_te")


def apply_target_encoding(df: DataFrame, encoding: TargetEncoding) -> DataFrame:
    """Add the encoded column to `df`. An unseen key gets the global mean.

    The table holds one row per key, so the left join cannot fan rows out. The
    `coalesce` turns the join's null for an unseen key into the global mean: a
    null reaching `VectorAssembler` fails the fit with an opaque message.
    """
    joined = df.join(F.broadcast(encoding.table), on=encoding.key_col, how="left")
    return joined.withColumn(
        encoding.out_col,
        F.coalesce(F.col(encoding.out_col), F.lit(encoding.global_mean)).cast("double"),
    )


def oof_target_encode(
    df: DataFrame,
    key_col: str = KEY_COL,
    target_col: str = TARGET,
    fold_col: str = OOF_FOLD_COL,
    k: int = DEFAULT_K,
    smoothing: float = DEFAULT_SMOOTHING,
) -> DataFrame:
    """Encode every row of `df` from the rows in the other `fold_col` folds.

    `df` must already carry `fold_col` (see `add_oof_fold`). Each of the k
    passes fits on the complement of one fold, including the complement's own
    global mean, and applies to that fold; the k frames are unioned back.
    """
    parts = []
    for fold in range(k):
        complement = df.filter(F.col(fold_col) != fold)
        held_out = df.filter(F.col(fold_col) == fold)
        encoding = fit_target_encoding(complement, key_col, target_col, smoothing)
        parts.append(apply_target_encoding(held_out, encoding))
    return reduce(DataFrame.unionByName, parts)
