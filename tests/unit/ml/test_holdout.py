"""Unit tests for spark/ml/src/holdout.py — Phase 5's one score on the sealed
sets (modeling plan §4a, §6 "HOLDOUT").

§4a says the holdout is scored exactly once and calls that "the part no code
enforces". These tests pin the code that now enforces it: a guard that refuses
when a score file already exists, and a scoring helper that fits once, scores
each sealed set, and returns metrics only — never predictions.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from spark.ml.src import holdout
from spark.ml.src.evaluate import compute_metrics


class CountingRegressor(BaseEstimator, RegressorMixin):
    """Predicts 2x the first column; counts its fits across the whole test."""

    fits = 0

    def fit(self, X, y):
        type(self).fits += 1
        return self

    def predict(self, X):
        return 2.0 * np.asarray(X)[:, 0]


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)

    def make(n):
        X = pd.DataFrame({"x": rng.uniform(1, 10, n)})
        y = pd.Series(2.0 * X["x"] + rng.normal(0, 0.5, n), name="fare_capped")
        return X, y

    CountingRegressor.fits = 0
    return {"train": make(50), "holdout": make(20), "temporal": make(12)}


class TestGuardOnce:
    def test_passes_when_no_score_file_exists(self, tmp_path):
        holdout.guard_once(tmp_path / "score.json")

    def test_refuses_when_the_score_file_exists(self, tmp_path):
        path = tmp_path / "score.json"
        path.write_text("{}")
        with pytest.raises(holdout.SealedScoreExists):
            holdout.guard_once(path)


class TestWriteScoreOnce:
    def test_writes_the_result_as_json(self, tmp_path):
        path = tmp_path / "score.json"
        holdout.write_score_once({"a": 1.5}, path)
        assert json.loads(path.read_text()) == {"a": 1.5}

    def test_second_write_refuses_and_keeps_the_first(self, tmp_path):
        path = tmp_path / "score.json"
        holdout.write_score_once({"a": 1.5}, path)
        with pytest.raises(holdout.SealedScoreExists):
            holdout.write_score_once({"a": 9.9}, path)
        assert json.loads(path.read_text()) == {"a": 1.5}


class TestScoreSealed:
    def _score(self, frames, groups=None):
        X_tr, y_tr = frames["train"]
        sealed = {k: frames[k] for k in ("holdout", "temporal")}
        return holdout.score_sealed(CountingRegressor(), X_tr, y_tr, sealed, groups=groups)

    def test_fits_exactly_once_for_all_sets(self, frames):
        self._score(frames)
        assert CountingRegressor.fits == 1

    def test_metrics_equal_compute_metrics_per_set(self, frames):
        out = self._score(frames)
        for name in ("holdout", "temporal"):
            X, y = frames[name]
            expected = compute_metrics(y, 2.0 * X["x"].to_numpy())
            got = out["sets"][name]
            assert got["rows"] == len(y)
            for key in ("mae", "rmse", "r2"):
                assert got[key] == pytest.approx(expected[key], rel=0, abs=1e-12)

    def test_group_metrics_split_one_set(self, frames):
        X, y = frames["temporal"]
        months = pd.Series(["2016-11"] * 5 + ["2016-12"] * 7, index=X.index)
        out = self._score(frames, groups={"temporal": months})
        by = out["sets"]["temporal"]["by_group"]
        assert set(by) == {"2016-11", "2016-12"}
        assert by["2016-11"]["rows"] == 5 and by["2016-12"]["rows"] == 7
        mask = (months == "2016-12").to_numpy()
        expected = compute_metrics(y[mask], 2.0 * X["x"].to_numpy()[mask])
        assert by["2016-12"]["mae"] == pytest.approx(expected["mae"], abs=1e-12)

    def test_returns_numbers_only_never_predictions(self, frames):
        """RULES: print metrics only. Nothing row-level may leave the helper."""
        out = self._score(frames)

        def walk(v):
            if isinstance(v, dict):
                for x in v.values():
                    walk(x)
            else:
                assert isinstance(v, (int, float)), f"non-scalar in the result: {type(v)}"

        walk(out)
        assert out["fit_time_s"] >= 0
