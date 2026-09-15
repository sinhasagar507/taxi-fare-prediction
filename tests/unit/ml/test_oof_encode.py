"""Unit tests for spark/ml/src/oof_encode.py — the out-of-fold target encoder
specified in migration plan §5.3.

Why this module exists. Plan §5b established an *effect* and left the
*mechanism* open. sklearn's TargetEncoder cross-fits inside `fit_transform`;
Spark's `TargetEncoder.fit` takes the plain per-category mean, so a training
row's own fare enters its own feature. On the D-012 train split (1,355,641
rows) 5,617 of 21,332 corridors hold exactly one trip (measured 2026-09-15), so
the encoding for those rows is the label, shrunk only by the smoothing. §5.3
asks for a cross-fitted encoding computed outside MLlib and fed in as a plain
column, so the §5.4 acceptance run can say whether the encoder was the whole
story.

Unlike tests/unit/ml/test_mllib.py, this module DOES build a SparkSession. The
encoder is k group-by + join passes over DataFrames, not scalar arithmetic, and
the thing most worth testing — that no row reads back its own target — only
exists at the DataFrame level. Plan §5.3 accepts the cost and confines it here:
these tests take seconds, not milliseconds.

Plan §10 names this the one place where a bug is *silent* leakage that inflates
R². Every test below is a defence against that, so each asserts an exact
expected number rather than a bound.

Row key, and a deviation from §5.3 item 1 recorded on purpose
-------------------------------------------------------------
§5.3 item 1 asks for `F.pmod(F.crc32(F.col(row_key)), k)` over `tripid`. The
D-012 train split, `sample_work_train.parquet`, carries no `tripid`: the prep
drops it. Adding it means a new split.

The owner chose the crc32 over the feature columns instead (migration plan
§5.4, "Re-based to D-012", decision 1, 2026-09-15). A new split would break
§5.4's own requirement, "same rows, same folds as 01_mllib_baseline.py", and
void row 2 (`mllib_gbt@work1355k`, MAE 0.466188) and the M4 smoke that
reproduced it. The single stated reason for wanting `tripid` is that
`monotonically_increasing_id()` is unstable across re-materialisation, so a
fold bug would be silent. A crc32 over the row's own frozen feature columns is
equally stable, and meets that reason. The stability tests below hold it to the
claim, and the key-column test pins which columns the key reads.

The feature key is not unique: 11 pairs of rows on the D-012 split share every
feature value (measured 2026-09-15; the largest group is 2). Rows that share a
key share a fold, so each is still encoded from the other folds only.
"""

import pytest

pyspark = pytest.importorskip("pyspark", reason="oof_encode is a PySpark module")

from pyspark.sql import SparkSession

from spark.ml.src import oof_encode as oof


SMOOTHING = 5.0


@pytest.fixture(scope="module")
def spark():
    """One local SparkSession for the module.

    `local[1]` and 2 shuffle partitions on purpose: the frames here are a
    handful of rows, and the default 200 partitions would spend the whole
    runtime on empty tasks.
    """
    session = (
        SparkSession.builder.master("local[1]")
        .appName("test_oof_encode")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def encodings_by_key(df, key_col="od_corridor", out_col="od_corridor_te"):
    """Collect {key: encoded value} so a test can assert exact arithmetic."""
    return {r[key_col]: r[out_col] for r in df.select(key_col, out_col).collect()}


# --------------------------------------------------------------------------
# The §5b 12-row worst case: every corridor unique, targets 10..21, mean 15.5.
# This is the shape that broke the Spark encoder — a frame of singletons, where
# an uncross-fitted encoding hands every row its own label back.
# --------------------------------------------------------------------------

@pytest.fixture
def worst_case(spark):
    rows = [(f"Z{i:02d}->Z{i:02d}", float(10 + i), i % 4) for i in range(12)]
    return spark.createDataFrame(rows, ["od_corridor", "fare_capped", "oof_fold"])


def test_worst_case_no_row_reads_back_its_own_target(worst_case):
    """The whole point of §5.3: no encoded value may equal its row's own fare.

    Under Spark's TargetEncoder every one of these 12 rows encodes to a value
    containing 1/(1+smoothing) of its own target. Out-of-fold, none does.
    """
    out = oof.oof_target_encode(worst_case, k=4, smoothing=SMOOTHING).collect()
    offenders = [r for r in out if r["od_corridor_te"] == pytest.approx(r["fare_capped"])]
    assert offenders == []


def test_worst_case_each_row_gets_its_complement_global_mean(worst_case):
    """Every corridor is a singleton, so no key survives into its complement.

    §5.3 item 2: a key absent from the complement encodes to the complement's
    global mean — not to the whole frame's 15.5, which would leak the row's own
    target through the mean.
    """
    targets = {i: float(10 + i) for i in range(12)}
    folds = {i: i % 4 for i in range(12)}
    expected = {}
    for i in range(12):
        complement = [targets[j] for j in range(12) if folds[j] != folds[i]]
        expected[f"Z{i:02d}->Z{i:02d}"] = sum(complement) / len(complement)

    out = oof.oof_target_encode(worst_case, k=4, smoothing=SMOOTHING)
    assert encodings_by_key(out) == pytest.approx(expected)


def test_worst_case_complement_mean_is_not_the_global_mean(worst_case):
    """A guard against the bug that would pass the test above by accident.

    If the implementation took the *whole frame's* mean (15.5) instead of the
    complement's, every row would still avoid its own target and the first test
    would pass. This one fails on that implementation.
    """
    out = oof.oof_target_encode(worst_case, k=4, smoothing=SMOOTHING)
    values = set(encodings_by_key(out).values())
    assert values != {15.5}
    assert len(values) == 4  # one distinct complement mean per fold


# --------------------------------------------------------------------------
# A repeated corridor: the encoding must come from the other folds only.
# --------------------------------------------------------------------------

@pytest.fixture
def repeated_corridor(spark):
    rows = [
        ("A", 10.0, 0),
        ("B", 12.0, 0),
        ("A", 20.0, 1),
        ("C", 30.0, 1),
    ]
    return spark.createDataFrame(rows, ["od_corridor", "fare_capped", "oof_fold"])


def test_repeated_corridor_encodes_from_the_other_fold_only(repeated_corridor):
    """Corridor A sits in both folds. Each A row must see only the other's fare.

    Fold 0's A row: complement is fold 1 = {A:20, C:30}, so global mean 25 and
    category mean 20 over n=1 -> (1*20 + 5*25) / (1+5) = 145/6.
    Fold 1's A row: complement is fold 0 = {A:10, B:12}, so global mean 11 and
    category mean 10 over n=1 -> (1*10 + 5*11) / (1+5) = 65/6.
    """
    out = oof.oof_target_encode(repeated_corridor, k=2, smoothing=SMOOTHING)
    got = {(r["od_corridor"], r["fare_capped"]): r["od_corridor_te"] for r in out.collect()}
    assert got[("A", 10.0)] == pytest.approx(145.0 / 6.0)
    assert got[("A", 20.0)] == pytest.approx(65.0 / 6.0)


def test_singleton_corridor_falls_back_to_its_complement_global_mean(repeated_corridor):
    """B and C appear in one fold each, so each is absent from its complement."""
    out = oof.oof_target_encode(repeated_corridor, k=2, smoothing=SMOOTHING)
    got = {(r["od_corridor"], r["fare_capped"]): r["od_corridor_te"] for r in out.collect()}
    assert got[("B", 12.0)] == pytest.approx(25.0)  # mean of fold 1
    assert got[("C", 30.0)] == pytest.approx(11.0)  # mean of fold 0


# --------------------------------------------------------------------------
# smoothing = 0 reduces to the plain complement mean.
# --------------------------------------------------------------------------

def test_smoothing_zero_returns_the_plain_complement_mean(spark):
    """§5.3's stated check: with every key present in every complement and
    smoothing 0, the formula collapses to the other fold's category mean.

    `self_leakage_weight(1, 0)` is 1.0 — at smoothing 0 an uncross-fitted
    singleton reads back 100% of its own fare. Out-of-fold, smoothing 0 is
    simply unsmoothed, and carries no leakage at all.
    """
    rows = [
        ("A", 10.0, 0),
        ("B", 12.0, 0),
        ("A", 20.0, 1),
        ("B", 40.0, 1),
    ]
    df = spark.createDataFrame(rows, ["od_corridor", "fare_capped", "oof_fold"])
    out = oof.oof_target_encode(df, k=2, smoothing=0.0)
    got = {(r["od_corridor"], r["fare_capped"]): r["od_corridor_te"] for r in out.collect()}
    assert got[("A", 10.0)] == pytest.approx(20.0)
    assert got[("B", 12.0)] == pytest.approx(40.0)
    assert got[("A", 20.0)] == pytest.approx(10.0)
    assert got[("B", 40.0)] == pytest.approx(12.0)


# --------------------------------------------------------------------------
# Transform time: full training means, no fold exclusion. §5.3 item 4.
# --------------------------------------------------------------------------

@pytest.fixture
def train_frame(spark):
    rows = [("A", 10.0), ("A", 20.0), ("B", 30.0)]
    return spark.createDataFrame(rows, ["od_corridor", "fare_capped"])


def test_transform_uses_full_training_means_without_fold_exclusion(spark, train_frame):
    """Test folds and the holdout encode the way sklearn's `transform()` does.

    Training global mean is 20. A has n=2 and mean 15 -> (2*15 + 5*20)/7 = 130/7.
    B has n=1 and mean 30 -> (1*30 + 5*20)/6 = 130/6. No fold is excluded here:
    fold exclusion is a training-half device, and applying it at transform time
    would make the feature disagree with the one the model was fitted on.
    """
    mapping = oof.fit_target_encoding(train_frame, smoothing=SMOOTHING)
    target = spark.createDataFrame([("A",), ("B",)], ["od_corridor"])
    got = encodings_by_key(oof.apply_target_encoding(target, mapping))
    assert got["A"] == pytest.approx(130.0 / 7.0)
    assert got["B"] == pytest.approx(130.0 / 6.0)


def test_unseen_key_at_transform_time_returns_the_global_mean(spark, train_frame):
    """§5.3 item 4, and §5b's "unknown categories agree" — a corridor the
    training half never saw encodes to the training global mean, which is 20."""
    mapping = oof.fit_target_encoding(train_frame, smoothing=SMOOTHING)
    target = spark.createDataFrame([("NEVER->SEEN",)], ["od_corridor"])
    got = encodings_by_key(oof.apply_target_encoding(target, mapping))
    assert got["NEVER->SEEN"] == pytest.approx(20.0)


def test_transform_never_leaves_a_null_encoding(spark, train_frame):
    """A null reaching VectorAssembler fails the fit with an opaque message.

    An unseen key must become the global mean, never a null from the left join.
    """
    mapping = oof.fit_target_encoding(train_frame, smoothing=SMOOTHING)
    target = spark.createDataFrame([("A",), ("X",), ("Y",)], ["od_corridor"])
    out = oof.apply_target_encoding(target, mapping)
    assert out.filter(out["od_corridor_te"].isNull()).count() == 0


# --------------------------------------------------------------------------
# Fold assignment: stable across re-materialisation. §5.3 item 1's real reason.
# --------------------------------------------------------------------------

@pytest.fixture
def unfolded(spark):
    rows = [(f"C{i % 5}", float(i), i % 3) for i in range(40)]
    return spark.createDataFrame(rows, ["od_corridor", "fare_capped", "passenger_count"])


def test_fold_assignment_is_stable_across_rematerialisation(spark, unfolded):
    """The bug §5.3 item 1 guards against, stated as a test.

    `monotonically_increasing_id()` depends on partitioning, so the same rows
    read back a second time would land in different folds and the OOF guarantee
    would quietly stop holding. A crc32 over the row's own values does not move.
    Row order and partition count are both changed here to prove it.
    """
    first = oof.add_oof_fold(unfolded, k=5)
    shuffled = unfolded.orderBy(unfolded["fare_capped"].desc()).repartition(7)
    second = oof.add_oof_fold(shuffled, k=5)

    def as_map(df):
        return {r["fare_capped"]: r["oof_fold"] for r in df.select("fare_capped", "oof_fold").collect()}

    assert as_map(first) == as_map(second)


def test_fold_assignment_is_stable_across_sessions(spark, unfolded):
    """Same rows, a second SparkSession, same folds.

    §5.3 asks for stability across sessions. A fresh session on the same JVM is
    the strongest form available in one test process, and it is enough to catch
    any assignment that depends on session-local state.
    """
    first = {r["fare_capped"]: r["oof_fold"] for r in oof.add_oof_fold(unfolded, k=5).collect()}
    other = spark.newSession()
    rows = [(f"C{i % 5}", float(i), i % 3) for i in range(40)]
    rebuilt = other.createDataFrame(rows, ["od_corridor", "fare_capped", "passenger_count"])
    second = {r["fare_capped"]: r["oof_fold"] for r in oof.add_oof_fold(rebuilt, k=5).collect()}
    assert first == second


def test_fold_assignment_uses_every_fold(unfolded):
    """A key that collapses to one fold would make the encoder a no-op and
    still pass every arithmetic test above, because there would be no
    complement to disagree with."""
    out = oof.add_oof_fold(unfolded, k=5)
    folds = {r["oof_fold"] for r in out.select("oof_fold").collect()}
    assert folds == {0, 1, 2, 3, 4}


def test_fold_values_stay_inside_the_range(unfolded):
    """`pmod`, not `%` — crc32 can exceed the signed int range, and Spark's `%`
    returns a negative remainder for a negative input, which would index a fold
    that does not exist."""
    out = oof.add_oof_fold(unfolded, k=5)
    assert out.filter((out["oof_fold"] < 0) | (out["oof_fold"] >= 5)).count() == 0


def test_add_oof_fold_preserves_the_row_count(unfolded):
    """Fold assignment is a `withColumn`, never a join that could fan rows out."""
    assert oof.add_oof_fold(unfolded, k=5).count() == unfolded.count()


def test_fold_key_uses_the_feature_columns_only(spark):
    """Break 6 of the D-012 re-base: the crc32 key's column list is pinned.

    The key hashes the feature columns only. The fare, the outer `_fold`, the
    encoder's own output and any earlier `oof_fold` stay out of it. Here two
    frames share every feature value and disagree on all four of those
    columns, so every row must land in the same fold in both.

    A key that took the fare would move rows between inner folds whenever a
    fare changed. A key that took the outer `_fold` would give each outer
    training half a different inner split, and a key that took the output
    would change the moment the column was written.
    """
    rows = [(f"C{i % 5}", float(i), i) for i in range(40)]
    features = spark.createDataFrame(rows, ["od_corridor", "fare_capped", "passenger_count"])
    noisy = spark.createDataFrame(
        [(c, fare * 7.0 + 1.0, p, p % 5, fare * 3.0, (p + 1) % 5) for c, fare, p in rows],
        ["od_corridor", "fare_capped", "passenger_count", "_fold", "od_corridor_te", "oof_fold"],
    )

    def as_map(df):
        return {
            (r["od_corridor"], r["passenger_count"]): r["oof_fold"]
            for r in oof.add_oof_fold(df, k=5).select("od_corridor", "passenger_count", "oof_fold").collect()
        }

    assert as_map(features) == as_map(noisy)


# --------------------------------------------------------------------------
# Contract with the MLlib pipeline. §5.3 item 5.
# --------------------------------------------------------------------------

def test_encoded_column_is_a_plain_double(worst_case):
    """§5.3 item 5 feeds this straight to VectorAssembler in place of the
    StringIndexer + TargetEncoder stages, so it has to be a double."""
    out = oof.oof_target_encode(worst_case, k=4, smoothing=SMOOTHING)
    assert dict(out.dtypes)["od_corridor_te"] == "double"


def test_oof_encode_preserves_the_row_count(worst_case):
    """k group-by + join passes unioned back together must return the input
    rows exactly once. A duplicated row would silently reweight the training
    half; a dropped one would shrink it."""
    out = oof.oof_target_encode(worst_case, k=4, smoothing=SMOOTHING)
    assert out.count() == worst_case.count()


def test_oof_encode_keeps_the_original_columns(worst_case):
    """The encoder adds a column. It must not take the raw corridor away — the
    --drop-corridor ablation and the audit path both still read it."""
    out = oof.oof_target_encode(worst_case, k=4, smoothing=SMOOTHING)
    assert set(worst_case.columns).issubset(set(out.columns))
