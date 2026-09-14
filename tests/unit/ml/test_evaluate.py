"""Unit tests for spark/ml/src/evaluate.py — the shared CV harness that makes
the Phase-4 model sweep a fair comparison (plan §4): every model goes through
the same folds and the same metrics, producing rows for one leaderboard.

Covers: metric functions against hand-computed values, deterministic shared
KFold, the evaluate() row contract, composability with Phase-2 pipelines,
and leaderboard assembly/sorting.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from spark.ml.src import evaluate as ev


# ---------------------------------------------------------------------------
# Metric functions — hand-computed expectations
# ---------------------------------------------------------------------------

class TestMetricFunctions:
    def test_perfect_prediction_gives_zero_error_full_r2(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert ev.mae(y, y) == 0.0
        assert ev.rmse(y, y) == 0.0
        assert ev.r2(y, y) == 1.0

    def test_constant_offset_mae_equals_rmse_equals_offset(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = y_true + 2.0
        assert ev.mae(y_true, y_pred) == pytest.approx(2.0)
        assert ev.rmse(y_true, y_pred) == pytest.approx(2.0)

    def test_rmse_penalizes_outliers_more_than_mae(self):
        y_true = np.zeros(4)
        y_pred = np.array([0.0, 0.0, 0.0, 4.0])  # one big miss
        assert ev.mae(y_true, y_pred) == pytest.approx(1.0)
        assert ev.rmse(y_true, y_pred) == pytest.approx(2.0)
        assert ev.rmse(y_true, y_pred) > ev.mae(y_true, y_pred)

    def test_mean_prediction_gives_zero_r2(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full(4, y_true.mean())
        assert ev.r2(y_true, y_pred) == pytest.approx(0.0)

    def test_compute_metrics_returns_all_three(self):
        y = np.array([1.0, 2.0, 3.0])
        m = ev.compute_metrics(y, y + 1.0)
        assert set(m) == {"mae", "rmse", "r2"}
        assert m["mae"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# make_cv — one deterministic fold set for every model
# ---------------------------------------------------------------------------

class TestMakeCv:
    def test_returns_kfold_with_requested_splits(self):
        cv = ev.make_cv(n_splits=7)
        assert isinstance(cv, KFold)
        assert cv.get_n_splits() == 7

    def test_default_is_five_shuffled_seeded_folds(self):
        cv = ev.make_cv()
        assert cv.get_n_splits() == 5
        assert cv.shuffle is True
        assert cv.random_state == 42

    def test_two_instances_yield_identical_folds(self):
        X = np.arange(40).reshape(-1, 1)
        folds_a = [test for _, test in ev.make_cv().split(X)]
        folds_b = [test for _, test in ev.make_cv().split(X)]
        for a, b in zip(folds_a, folds_b):
            assert np.array_equal(a, b), "make_cv folds are not reproducible"


# ---------------------------------------------------------------------------
# evaluate — the row contract
# ---------------------------------------------------------------------------

REQUIRED_ROW_KEYS = {
    "model",
    "mae_mean", "mae_std",
    "rmse_mean", "rmse_std",
    "r2_mean", "r2_std",
    "fit_time_s",
}


@pytest.fixture
def linear_data():
    """y is an exact linear function of X — LinearRegression must ace it."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"a": rng.uniform(0, 10, 60), "b": rng.uniform(0, 5, 60)})
    y = pd.Series(3.0 * X["a"] - 2.0 * X["b"] + 7.0, name="target")
    return X, y


class TestEvaluate:
    def test_returns_row_with_required_keys(self, linear_data):
        X, y = linear_data
        row = ev.evaluate(LinearRegression(), X, y, cv=ev.make_cv(n_splits=3))
        assert REQUIRED_ROW_KEYS <= set(row)

    def test_model_name_defaults_to_class_name(self, linear_data):
        X, y = linear_data
        row = ev.evaluate(LinearRegression(), X, y, cv=ev.make_cv(n_splits=3))
        assert row["model"] == "LinearRegression"

    def test_explicit_name_overrides_default(self, linear_data):
        X, y = linear_data
        row = ev.evaluate(
            LinearRegression(), X, y, cv=ev.make_cv(n_splits=3), name="ols_baseline"
        )
        assert row["model"] == "ols_baseline"

    def test_exact_linear_relation_scores_near_perfect(self, linear_data):
        X, y = linear_data
        row = ev.evaluate(LinearRegression(), X, y, cv=ev.make_cv(n_splits=3))
        assert row["mae_mean"] == pytest.approx(0.0, abs=1e-8)
        assert row["r2_mean"] == pytest.approx(1.0, abs=1e-8)

    def test_dummy_mean_regressor_scores_near_zero_r2(self, linear_data):
        X, y = linear_data
        row = ev.evaluate(DummyRegressor(), X, y, cv=ev.make_cv(n_splits=3))
        assert row["r2_mean"] < 0.1
        assert row["rmse_mean"] > 0.0

    def test_composes_with_phase2_pipeline(self):
        """The harness must accept Pipeline(preprocessor, model) unchanged —
        that is how every Phase-4 sweep entry will arrive."""
        from spark.ml.src.preprocess import build_preprocessor

        rng = np.random.default_rng(42)
        n = 60
        X = pd.DataFrame(
            {
                "distance_capped": rng.uniform(0.5, 18.0, n),
                "service_type": rng.choice(["Yellow", "Green"], n),
                "od_corridor": rng.choice(["A→B", "B→C", "C→D"], n),
            }
        )
        y = pd.Series(4.0 * X["distance_capped"] + rng.normal(0, 0.5, n))
        pipe = Pipeline(
            [
                ("pre", build_preprocessor("scaled", X.columns)),
                ("model", LinearRegression()),
            ]
        )
        row = ev.evaluate(pipe, X, y, cv=ev.make_cv(n_splits=3), name="ridge_pipe")
        assert row["r2_mean"] > 0.9


# ---------------------------------------------------------------------------
# leaderboard — assembly and ranking
# ---------------------------------------------------------------------------

class TestLeaderboard:
    def test_rows_become_dataframe_sorted_by_rmse(self):
        rows = [
            {"model": "worse", "mae_mean": 3.0, "mae_std": 0.1,
             "rmse_mean": 5.0, "rmse_std": 0.2, "r2_mean": 0.5, "r2_std": 0.02,
             "fit_time_s": 1.0},
            {"model": "better", "mae_mean": 1.0, "mae_std": 0.1,
             "rmse_mean": 2.0, "rmse_std": 0.2, "r2_mean": 0.9, "r2_std": 0.01,
             "fit_time_s": 2.0},
        ]
        board = ev.leaderboard(rows)
        assert isinstance(board, pd.DataFrame)
        assert board.iloc[0]["model"] == "better"
        assert board["rmse_mean"].is_monotonic_increasing

    def test_empty_rows_give_empty_frame(self):
        board = ev.leaderboard([])
        assert isinstance(board, pd.DataFrame)
        assert board.empty


# ---------------------------------------------------------------------------
# make_holdout — the sealed test set (plan §4a)
# ---------------------------------------------------------------------------

@pytest.fixture
def strata_frame():
    """Shaped like build_features() output: carries the two columns the prep
    stratified its sample on (service_type x temp_band_ord), with deliberately
    uneven strata so a proportional split is actually testable."""
    rng = np.random.default_rng(42)
    n = 2000
    service = np.where(rng.random(n) < 0.75, "Yellow", "Green")
    band = rng.choice([0, 1, 2, 3, 4], n, p=[0.05, 0.2, 0.35, 0.3, 0.1])
    X = pd.DataFrame(
        {
            "distance_capped": rng.uniform(0.5, 18.0, n),
            "service_type": service,
            "temp_band_ord": band,
        }
    )
    y = pd.Series(4.0 * X["distance_capped"] + rng.normal(0, 0.5, n), name="fare_capped")
    return X, y


class TestMakeHoldout:
    def test_default_split_is_eighty_twenty(self, strata_frame):
        X, y = strata_frame
        X_tr, X_te, y_tr, y_te = ev.make_holdout(X, y)
        assert len(X_te) == pytest.approx(0.2 * len(X), abs=1)
        assert len(X_tr) + len(X_te) == len(X)
        assert len(y_tr) == len(X_tr) and len(y_te) == len(X_te)

    def test_train_and_test_rows_are_disjoint(self, strata_frame):
        X, y = strata_frame
        X_tr, X_te, _, _ = ev.make_holdout(X, y)
        assert not set(X_tr.index) & set(X_te.index), "row appears in both splits"

    def test_features_and_target_stay_aligned(self, strata_frame):
        X, y = strata_frame
        X_tr, X_te, y_tr, y_te = ev.make_holdout(X, y)
        assert (X_tr.index == y_tr.index).all()
        assert (X_te.index == y_te.index).all()

    def test_same_seed_gives_identical_split(self, strata_frame):
        X, y = strata_frame
        first = ev.make_holdout(X, y)[1].index.tolist()
        second = ev.make_holdout(X, y)[1].index.tolist()
        assert first == second, "holdout must be reproducible across calls"

    def test_different_seed_gives_different_split(self, strata_frame):
        X, y = strata_frame
        a = ev.make_holdout(X, y, random_state=1)[1].index.tolist()
        b = ev.make_holdout(X, y, random_state=2)[1].index.tolist()
        assert a != b

    def test_strata_proportions_are_preserved(self, strata_frame):
        X, y = strata_frame
        X_tr, X_te, _, _ = ev.make_holdout(X, y)
        key = lambda d: (d["service_type"] + "|" + d["temp_band_ord"].astype(str))
        full = key(X).value_counts(normalize=True)
        for part in (X_tr, X_te):
            got = key(part).value_counts(normalize=True)
            for stratum, share in full.items():
                assert got[stratum] == pytest.approx(share, abs=0.01), (
                    f"stratum {stratum} drifted"
                )

    def test_custom_test_size_is_honoured(self, strata_frame):
        X, y = strata_frame
        _, X_te, _, _ = ev.make_holdout(X, y, test_size=0.1)
        assert len(X_te) == pytest.approx(0.1 * len(X), abs=1)

    def test_missing_strata_columns_fall_back_to_unstratified(self):
        """The duration ablation and smoke frames may not carry both keys;
        a missing key must degrade to a plain random split, not crash."""
        X = pd.DataFrame({"distance_capped": np.arange(100, dtype=float)})
        y = pd.Series(np.arange(100, dtype=float), name="fare_capped")
        X_tr, X_te, _, _ = ev.make_holdout(X, y)
        assert len(X_te) == 20 and len(X_tr) == 80


# ---------------------------------------------------------------------------
# make_temporal_test — the second sealed test set, by date (plan §4a)
# ---------------------------------------------------------------------------

@pytest.fixture
def dated_frame():
    """Shaped like a prep sample, before build_features: one trip per day over
    the 24-month window, so the last two months are 61 of 731 rows."""
    days = pd.date_range("2015-01-01 08:30", "2016-12-31 08:30", freq="D")
    return pd.DataFrame(
        {
            "pickup_datetime": days,
            "fare_capped": np.arange(len(days), dtype=float),
            "service_type": ["Yellow", "Green"] * (len(days) // 2) + ["Yellow"],
        }
    )


class TestMakeTemporalTest:
    """The random 80/20 lets 2016-06 inform a 2016-03 prediction. The temporal
    set holds out the last two months whole, so Phase 5 can score the champion
    on trips later than anything it trained on. It carves the raw frame,
    because build_features drops `pickup_datetime`."""

    def test_default_cutoff_is_the_first_of_november_2016(self):
        assert ev.TEMPORAL_CUTOFF == pd.Timestamp("2016-11-01")

    def test_rows_from_the_cutoff_on_form_the_temporal_set(self, dated_frame):
        _, temporal = ev.make_temporal_test(dated_frame)
        assert (temporal["pickup_datetime"] >= pd.Timestamp("2016-11-01")).all()
        assert len(temporal) == 61

    def test_rows_before_the_cutoff_stay_for_the_holdout_and_sweep(self, dated_frame):
        before, _ = ev.make_temporal_test(dated_frame)
        assert (before["pickup_datetime"] < pd.Timestamp("2016-11-01")).all()
        assert len(before) == 670

    def test_a_trip_exactly_at_the_cutoff_is_temporal(self):
        df = pd.DataFrame(
            {"pickup_datetime": pd.to_datetime(["2016-10-31 23:59:59", "2016-11-01 00:00:00"])}
        )
        before, temporal = ev.make_temporal_test(df)
        assert temporal["pickup_datetime"].tolist() == [pd.Timestamp("2016-11-01")]
        assert len(before) == 1

    def test_no_row_is_lost_or_duplicated(self, dated_frame):
        before, temporal = ev.make_temporal_test(dated_frame)
        assert not set(before.index) & set(temporal.index)
        assert sorted(before.index.tolist() + temporal.index.tolist()) == dated_frame.index.tolist()

    def test_index_labels_are_kept_so_rows_stay_traceable(self, dated_frame):
        before, temporal = ev.make_temporal_test(dated_frame)
        assert (dated_frame.loc[temporal.index, "fare_capped"] == temporal["fare_capped"]).all()
        assert (dated_frame.loc[before.index, "fare_capped"] == before["fare_capped"]).all()

    def test_custom_cutoff_is_honoured(self, dated_frame):
        _, temporal = ev.make_temporal_test(dated_frame, cutoff=pd.Timestamp("2016-12-01"))
        assert len(temporal) == 31

    def test_a_frame_without_the_date_column_raises(self):
        """A pre-D-012 sample carries no `pickup_datetime`. Returning an empty
        temporal set would look like a carve and silently seal nothing."""
        with pytest.raises(ValueError, match="pickup_datetime"):
            ev.make_temporal_test(pd.DataFrame({"fare_capped": [1.0, 2.0]}))

    def test_the_random_holdout_then_draws_only_from_before_the_cutoff(self, dated_frame):
        """The order the sweep uses: carve by date first, then the random 80/20
        on what is left. No sealed or training row may come from the last two
        months."""
        before, temporal = ev.make_temporal_test(dated_frame)
        X = before.drop(columns=["fare_capped"])
        y = before["fare_capped"]
        X_tr, X_te, _, _ = ev.make_holdout(X, y, stratify_columns=())
        for part in (X_tr, X_te):
            assert (part["pickup_datetime"] < ev.TEMPORAL_CUTOFF).all()
            assert set(part.index).isdisjoint(temporal.index)


# ---------------------------------------------------------------------------
# write_train_split — handing the sealed split's TRAIN half to Spark
# ---------------------------------------------------------------------------

class TestWriteTrainSplit:
    """The MLlib baseline (plan §5b) must train on the *same* rows the sklearn
    sweep used, but `make_holdout` is `sklearn.train_test_split` and seeds do
    not transfer across libraries — Spark cannot re-derive the split. So pandas
    writes the train half once and Spark reads that file.
    """

    @pytest.fixture
    def frame(self):
        X = pd.DataFrame(
            {
                "distance_capped": np.arange(100, dtype=float),
                "service_type": ["Yellow", "Green"] * 50,
            }
        )
        y = pd.Series(np.arange(100, dtype=float) * 2, name="fare_capped")
        return X, y

    def test_writes_a_file_at_the_requested_path(self, frame, tmp_path):
        X, y = frame
        out = tmp_path / "sample_work_train.parquet"
        ev.write_train_split(X, y, out)
        assert out.exists()

    def test_returns_the_path_it_wrote(self, frame, tmp_path):
        X, y = frame
        out = tmp_path / "train.parquet"
        assert ev.write_train_split(X, y, out) == out

    def test_row_count_round_trips(self, frame, tmp_path):
        X, y = frame
        out = tmp_path / "train.parquet"
        ev.write_train_split(X, y, out)
        assert len(pd.read_parquet(out)) == len(X)

    def test_target_column_is_included_and_named(self, frame, tmp_path):
        """MLlib needs `labelCol="fare_capped"` in the same frame — a features
        file without the label is useless to the Spark script."""
        X, y = frame
        out = tmp_path / "train.parquet"
        ev.write_train_split(X, y, out)
        back = pd.read_parquet(out)
        assert list(back.columns) == list(X.columns) + ["fare_capped"]

    def test_feature_and_target_values_stay_row_aligned(self, frame, tmp_path):
        X, y = frame
        out = tmp_path / "train.parquet"
        ev.write_train_split(X, y, out)
        back = pd.read_parquet(out)
        # y was built as 2x the distance column; misalignment breaks this.
        assert (back["fare_capped"] == back["distance_capped"] * 2).all()

    def test_pandas_index_is_not_written_as_a_column(self, frame, tmp_path):
        """pandas writes the index as `__index_level_0__` by default and Spark
        reads it back as a real column — it would land in the numeric
        passthrough and become a feature."""
        X, y = frame
        out = tmp_path / "train.parquet"
        ev.write_train_split(X, y, out)
        back = pd.read_parquet(out)
        assert not any(c.startswith("__index_level_") for c in back.columns)

    def test_shuffled_non_contiguous_index_still_aligns(self, frame, tmp_path):
        """The real input is make_holdout's output, whose index is shuffled and
        gappy. Concatenating on position rather than label would scramble it."""
        X, y = frame
        X_tr, _, y_tr, _ = ev.make_holdout(X, y, test_size=0.2)
        assert not X_tr.index.equals(pd.RangeIndex(len(X_tr)))  # guard the premise
        out = tmp_path / "train.parquet"
        ev.write_train_split(X_tr, y_tr, out)
        back = pd.read_parquet(out)
        assert (back["fare_capped"] == back["distance_capped"] * 2).all()

    def test_creates_missing_parent_directories(self, frame, tmp_path):
        X, y = frame
        out = tmp_path / "nested" / "dir" / "train.parquet"
        ev.write_train_split(X, y, out)
        assert out.exists()

    def test_writes_exactly_the_sweep_train_half(self, frame, tmp_path):
        """End to end: the file Spark reads holds the 80% the sweep trained on
        and none of the sealed 20%."""
        X, y = frame
        X_tr, X_te, y_tr, _ = ev.make_holdout(X, y)
        out = tmp_path / "train.parquet"
        ev.write_train_split(X_tr, y_tr, out)
        back = pd.read_parquet(out)
        assert len(back) == len(X_tr) == 80
        sealed = set(X_te["distance_capped"])
        assert sealed.isdisjoint(back["distance_capped"])

    def test_explicit_target_name_overrides_the_series_name(self, frame, tmp_path):
        X, y = frame
        out = tmp_path / "train.parquet"
        ev.write_train_split(X, y, out, target_name="label")
        assert "label" in pd.read_parquet(out).columns

    def test_unnamed_target_without_explicit_name_raises(self, frame, tmp_path):
        """Silently writing a column called `None` would fail far away, inside
        Spark, with an unrecognisable error."""
        X, y = frame
        with pytest.raises(ValueError):
            ev.write_train_split(X, y.rename(None), tmp_path / "t.parquet")

    def test_target_name_colliding_with_a_feature_raises(self, frame, tmp_path):
        """A duplicate column name silently doubles a feature in Spark."""
        X, y = frame
        with pytest.raises(ValueError):
            ev.write_train_split(X, y, tmp_path / "t.parquet",
                                 target_name="distance_capped")


# ---------------------------------------------------------------------------
# train_split_filename — derived from content, never from --tag
# ---------------------------------------------------------------------------

class TestTrainSplitFilename:
    """The filename must depend only on what changes the file's *contents*:
    which sample, whether duration is in, and any row limit. An earlier version
    keyed it off `--tag`, which is a presentation label for leaderboard outputs
    and is freely overridable — so `--tag worksplit` produced
    `sample_worksplit_train.parquet` and the artifact stopped matching the name
    plan §5b refers to.
    """

    def test_default_work_run_matches_the_name_in_the_plan(self):
        assert ev.train_split_filename("work") == "sample_work_train.parquet"

    def test_full_sample_gets_its_own_name(self):
        assert ev.train_split_filename("full") == "sample_full_train.parquet"

    def test_duration_ablation_is_a_different_file(self):
        """Dropping trip_duration_min changes the column set, so it cannot
        share a filename with the duration-on split."""
        assert (
            ev.train_split_filename("work", include_duration=False)
            == "sample_work_no_duration_train.parquet"
        )

    def test_row_limit_is_in_the_name_so_smoke_runs_cannot_clobber(self):
        assert (
            ev.train_split_filename("work", limit_rows=100_000)
            == "sample_work_limit100000_train.parquet"
        )

    def test_limit_and_ablation_compose_deterministically(self):
        assert (
            ev.train_split_filename("work", include_duration=False, limit_rows=5000)
            == "sample_work_no_duration_limit5000_train.parquet"
        )

    def test_zero_limit_is_still_a_limit_not_a_missing_one(self):
        """`if limit_rows:` would treat 0 as absent and silently reuse the full
        split's name for a 0-row file."""
        assert "limit0" in ev.train_split_filename("work", limit_rows=0)

    def test_no_limit_leaves_the_name_clean(self):
        assert "limit" not in ev.train_split_filename("work", limit_rows=None)
