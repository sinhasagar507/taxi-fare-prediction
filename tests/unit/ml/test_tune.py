"""Unit tests for spark/ml/src/tune.py — Phase 5 tuning and diagnostics.

What these pin, each against a named failure:

1. Trial 0 is the sweep. The defaults the study enqueues must build the same
   pipeline the `work_d012` sweep scored, or "tuned beat default" compares two
   different models. Checked by prediction equality, not by parameter names.
2. The defaults sit inside the search space under the same names. Otherwise
   Optuna enqueues a value its sampler can never propose again.
3. The objective is the Phase-3 harness. It returns evaluate()'s MAE on the
   caller's folds, so a tuned number stays comparable to the leaderboard.
4. The study is seeded, so a re-run proposes the same trials.
5. Hour recovered from sin/cos is exact for all 24 hours. A rounding slip
   would move a slice's trips into the next hour.
6. The champion rule follows the owner's decision: lower MAE, unless the gap
   is inside the larger fold std, where the faster fit wins.

optuna, lightgbm and catboost are container-only (CLAUDE.md, Environments), so
the module skips on the host venv.
"""

import numpy as np
import pandas as pd
import pytest

optuna = pytest.importorskip("optuna", reason="tune.py needs optuna (container-only)")
pytest.importorskip("lightgbm", reason="container-only booster")
pytest.importorskip("catboost", reason="container-only booster")

from spark.ml.src import evaluate as ev  # noqa: E402
from spark.ml.src import features as ft  # noqa: E402
from spark.ml.src import sweep as sw  # noqa: E402
from spark.ml.src import tune  # noqa: E402


@pytest.fixture
def data():
    """A small frame shaped like the real feature frame, so every
    preprocessor block runs: numerics, a one-hot column, the corridor key."""
    rng = np.random.default_rng(42)
    n = 120
    X = pd.DataFrame(
        {
            "distance_capped": rng.uniform(0.5, 18.0, n),
            "passenger_count": rng.integers(1, 7, n).astype(float),
            "service_type": rng.choice(["Yellow", "Green"], n),
            "od_corridor": rng.choice(["A→B", "B→C", "C→D", "D→A"], n),
        }
    )
    y = pd.Series(4.0 * X["distance_capped"] + rng.normal(0, 0.5, n), name="fare_capped")
    return X, y


def _registry_spec(name):
    return next(s for s in sw.build_model_specs() if s.name == name)


class TestModelNames:
    def test_tunes_the_two_leaders(self):
        assert tune.TUNED_MODELS == ("lightgbm", "catboost")

    @pytest.mark.parametrize(
        "call",
        [
            lambda: tune.default_params("ridge"),
            lambda: tune.suggest_params("ridge", optuna.trial.FixedTrial({})),
            lambda: tune.build_tuned_pipeline("ridge", {}, ["distance_capped"]),
        ],
    )
    def test_unknown_model_raises(self, call):
        with pytest.raises(ValueError, match="ridge"):
            call()


@pytest.mark.parametrize("name", ["lightgbm", "catboost"])
class TestDefaults:
    def test_defaults_are_inside_the_search_space(self, name):
        defaults = tune.default_params(name)
        suggested = tune.suggest_params(name, optuna.trial.FixedTrial(defaults))
        assert suggested == defaults

    def test_default_pipeline_predicts_like_the_sweep_pipeline(self, name, data):
        X, y = data
        swept = sw.build_pipeline(_registry_spec(name), X.columns).fit(X, y)
        tuned = tune.build_tuned_pipeline(name, tune.default_params(name), X.columns).fit(X, y)
        np.testing.assert_allclose(tuned.predict(X), swept.predict(X), rtol=0, atol=1e-9)


class TestObjective:
    def test_returns_the_harness_mae_and_records_the_other_metrics(self, data):
        X, y = data
        cv = ev.make_cv(n_splits=3)
        defaults = tune.default_params("lightgbm")
        trial = optuna.trial.FixedTrial(defaults)

        value = tune.make_objective("lightgbm", X, y, cv)(trial)

        expected = ev.evaluate(
            tune.build_tuned_pipeline("lightgbm", defaults, X.columns), X, y, cv=cv
        )
        assert value == pytest.approx(expected["mae_mean"], abs=1e-12)
        for key in ("mae_std", "rmse_mean", "rmse_std", "r2_mean", "r2_std", "fit_time_s"):
            assert key in trial.user_attrs


class TestRunStudy:
    def test_trial_zero_is_the_defaults_and_the_sampler_is_seeded(self, data):
        X, y = data
        study = tune.run_study("lightgbm", X, y, ev.make_cv(n_splits=3), n_trials=3)

        assert study.direction == optuna.study.StudyDirection.MINIMIZE
        assert isinstance(study.sampler, optuna.samplers.TPESampler)
        assert study.trials[0].params == tune.default_params("lightgbm")
        assert len(study.trials) == 3

    def test_a_rerun_proposes_the_same_trials(self, data):
        X, y = data
        cv = ev.make_cv(n_splits=3)
        first = tune.run_study("lightgbm", X, y, cv, n_trials=3)
        second = tune.run_study("lightgbm", X, y, cv, n_trials=3)
        assert [t.params for t in first.trials] == [t.params for t in second.trials]


class TestHourFromCyclic:
    def test_recovers_every_hour_exactly(self):
        hours = pd.DataFrame({"pickup_hour": np.arange(24)})
        encoded = ft.encode_cyclic_hour(hours)
        recovered = tune.hour_from_cyclic(
            encoded["pickup_hour_sin"], encoded["pickup_hour_cos"]
        )
        np.testing.assert_array_equal(np.asarray(recovered), np.arange(24))


class TestSliceMae:
    def test_mae_and_rows_per_group(self):
        y_true = pd.Series([1.0, 2.0, 3.0, 10.0])
        y_pred = pd.Series([1.5, 2.0, 2.0, 12.0])
        groups = pd.Series(["b", "a", "b", "a"], name="pickup_borough")

        out = tune.slice_mae(y_true, y_pred, groups)

        assert list(out.columns) == ["pickup_borough", "rows", "mae"]
        assert out["pickup_borough"].tolist() == ["a", "b"]
        assert out["rows"].tolist() == [2, 2]
        assert out["mae"].tolist() == pytest.approx([1.0, 0.75])


class TestPickChampion:
    def test_a_gap_outside_the_std_goes_to_the_lower_mae(self):
        rows = {
            "slow": {"mae_mean": 0.320, "mae_std": 0.001, "fit_time_s": 30.0},
            "fast": {"mae_mean": 0.331, "mae_std": 0.001, "fit_time_s": 5.0},
        }
        assert tune.pick_champion(rows)["champion"] == "slow"

    def test_a_gap_inside_the_larger_std_goes_to_the_faster_fit(self):
        rows = {
            "slow": {"mae_mean": 0.3200, "mae_std": 0.0010, "fit_time_s": 30.0},
            "fast": {"mae_mean": 0.3208, "mae_std": 0.0013, "fit_time_s": 5.0},
        }
        result = tune.pick_champion(rows)
        assert result["champion"] == "fast"
        assert result["gap"] == pytest.approx(0.0008)
        assert result["larger_std"] == pytest.approx(0.0013)
        assert result["inside_std"] is True
