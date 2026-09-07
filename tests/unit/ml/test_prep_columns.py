"""Red tests for the prep column contract — migration plan §5.3 and §4a.

`00_prep_spark.py` writes the samples every downstream stage reads, so its
`keep` list IS the modeling schema. Two things the plan needs are missing from
it, and one thing it would silently admit must stay out:

  - **`tripid`** (§5.3 step 1). The out-of-fold target encoder assigns folds
    with `F.pmod(F.crc32(F.col(row_key)), k)`. The plan rejects
    `monotonically_increasing_id()` because it is not stable across
    re-materialisation, so a wrong fold would be silent. There is no other
    stable row key in the sample. `tripid` is currently in `DROP_COLS`.
  - **`pickup_datetime`** (§4a). The deferred temporal split needs the
    timestamp. `derive_and_guard` reads it for `pickup_hour`/`pickup_dow`, then
    the keep list discards it.
  - **Neither may reach the feature matrix.** `features.build_features` drops
    `[TARGET, *EXCLUDED_COLUMNS]` — a *drop* list, not an allow list — so any
    new prep column becomes a model feature by default. An identifier and a
    raw timestamp as features is leakage and a dtype break. The comment above
    `EXCLUDED_COLUMNS` already anticipates this: the list exists "in case a
    future prep re-adds them". This is that future prep.

The plan ties the two additions together:

> The prep's `keep` list needs `tripid` (or a persisted row id) — the same prep
> re-run §4a's deferred temporal split is already waiting on
> (`pickup_datetime`). Do both in one re-run.

These tests are red on purpose. They pin the contract before the prep changes.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyspark", reason="00_prep_spark.py imports pyspark at module level")

from spark.ml.src import features as ft

REPO_ROOT = Path(__file__).resolve().parents[3]
PREP_PATH = REPO_ROOT / "spark" / "ml" / "00_prep_spark.py"

# The row key §5.3 needs, and the timestamp §4a needs.
OOF_ROW_KEY = "tripid"
TEMPORAL_SPLIT_COL = "pickup_datetime"


def _load_prep():
    """Import `00_prep_spark.py` by path — the leading digit blocks `import`."""
    spec = importlib.util.spec_from_file_location("prep_spark", PREP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prep():
    return _load_prep()


class _SelectSpy:
    """Stands in for a DataFrame: records what `.select()` was handed.

    Keeps this test off a SparkSession. `select_model_columns` does one thing,
    and this is the one thing.
    """

    def __init__(self):
        self.selected = None

    def select(self, *cols):
        self.selected = list(cols)
        return self


# --- the two missing columns -------------------------------------------------

def test_keep_cols_carries_the_oof_row_key(prep):
    """§5.3 cannot assign folds without a stable row key."""
    assert OOF_ROW_KEY in prep.KEEP_COLS


def test_keep_cols_carries_the_temporal_split_column(prep):
    """§4a's deferred temporal split needs the raw pickup timestamp."""
    assert TEMPORAL_SPLIT_COL in prep.KEEP_COLS


def test_tripid_is_no_longer_dropped(prep):
    """`DROP_COLS` and `KEEP_COLS` must not disagree about the same column."""
    assert OOF_ROW_KEY not in prep.DROP_COLS


# --- the constant and the function must not drift apart ----------------------

def test_select_model_columns_selects_exactly_keep_cols(prep):
    """A module constant nobody selects is a lie the tests would not catch."""
    spy = _SelectSpy()
    prep.select_model_columns(spy)
    assert spy.selected == list(prep.KEEP_COLS)


def test_keep_cols_admits_no_leakage_column(prep):
    """Widening the keep list must never re-admit a §5 post-trip field."""
    overlap = sorted(set(prep.KEEP_COLS) & set(prep.LEAKAGE_COLS))
    assert overlap == []


# --- and neither new column may become a feature -----------------------------

def test_new_columns_are_excluded_from_the_feature_matrix():
    """`EXCLUDED_COLUMNS` is the only thing standing between the prep output
    and `X`, because `build_features` drops rather than allows."""
    assert OOF_ROW_KEY in ft.EXCLUDED_COLUMNS
    assert TEMPORAL_SPLIT_COL in ft.EXCLUDED_COLUMNS


def test_build_features_keeps_the_row_key_and_timestamp_out_of_x():
    """The end-to-end guarantee: an identifier and a raw timestamp are not
    model inputs, whatever the prep decides to carry."""
    df = pd.DataFrame(
        {
            "fare_capped": [12.0, 30.0],
            "fare_amount": [12.0, 30.0],
            "trip_distance": [2.0, 7.5],
            "distance_capped": [2.0, 7.5],
            "trip_duration_min": [8.0, 22.0],
            "passenger_count": [1, 2],
            "temperature": [55.0, 71.0],
            "pickup_hour": [9, 18],
            "pickup_dow": [3, 6],
            "service_type": ["Yellow", "Green"],
            "pickup_borough": ["Manhattan", "Brooklyn"],
            "dropoff_borough": ["Queens", "Manhattan"],
            "pickup_zone": ["Midtown", "Park Slope"],
            "dropoff_zone": ["Astoria", "SoHo"],
            "temp_band": ["Mild", "Warm"],
            "is_airport_trip": [0, 1],
            "ratecodeid": [1, 2],
            "humidity": [0.6, 0.5],
            "windSpeed": [5.0, 9.0],
            "visibility": [10.0, 9.0],
            OOF_ROW_KEY: ["abc123", "def456"],
            TEMPORAL_SPLIT_COL: pd.to_datetime(["2015-03-01 09:14", "2016-07-04 18:02"]),
        }
    )
    X, y = ft.build_features(df)
    assert OOF_ROW_KEY not in X.columns
    assert TEMPORAL_SPLIT_COL not in X.columns
    assert y.tolist() == [12.0, 30.0]
