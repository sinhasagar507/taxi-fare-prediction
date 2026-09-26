"""Unit tests for spark/ml/src/load.py — the lean prep-sample loader for the
§5c scale-up (modeling plan §6, "RESCALE").

The SCALE goal's VM was OOM-killed right after `pd.read_parquet` loaded
`sample_full` with every column, before `build_features` ran. The loader reads
only the columns `build_features` keeps or consumes, and casts the decimal128
climate columns to float64 in Arrow, before any pandas object exists. These
tests pin that it drops exactly the unused columns and that everything
downstream of it is equal to the `pd.read_parquet` path.
"""

from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spark.ml.src import features, load
from spark.ml.src.evaluate import make_temporal_test

DECIMAL = pa.decimal128(38, 9)

# sample_work.parquet's schema, read 2026-09-25: 22 columns in file order.
SCHEMA = pa.schema([
    ("tripid", pa.string()),
    ("pickup_datetime", pa.timestamp("ns")),
    ("fare_capped", pa.float64()),
    ("fare_amount", pa.float64()),
    ("trip_distance", pa.float64()),
    ("distance_capped", pa.float64()),
    ("trip_duration_min", pa.float64()),
    ("passenger_count", pa.int32()),
    ("temperature", pa.float64()),
    ("pickup_hour", pa.int32()),
    ("pickup_dow", pa.int32()),
    ("service_type", pa.string()),
    ("pickup_borough", pa.string()),
    ("dropoff_borough", pa.string()),
    ("pickup_zone", pa.string()),
    ("dropoff_zone", pa.string()),
    ("temp_band", pa.string()),
    ("is_airport_trip", pa.int32()),
    ("ratecodeid", pa.int64()),
    ("humidity", DECIMAL),
    ("windSpeed", DECIMAL),
    ("visibility", DECIMAL),
])

UNUSED = ["tripid", "fare_amount", "trip_distance", "temperature", "ratecodeid"]


def _rows(offset: int) -> dict:
    """Four rows; one per part keeps a None decimal and a post-cutoff date,
    so the NaN cast and the temporal carve are both exercised."""
    return {
        "tripid": [f"t{offset + i}" for i in range(4)],
        "pickup_datetime": pd.to_datetime(
            ["2016-01-03 08:00", "2016-03-15 23:10", "2016-05-02 00:05",
             "2016-06-30 17:45"]),
        "fare_capped": [12.5, 45.0, 7.25, 30.0],
        "fare_amount": [12.5, 61.0, 7.25, 30.0],
        "trip_distance": [2.1, 8003318.0, 0.9, 6.3],
        "distance_capped": [2.1, 14.15, 0.9, 6.3],
        "trip_duration_min": [11.0, 42.0, 5.5, 28.0],
        "passenger_count": [1, 3, 1, 2],
        "temperature": [55.0, 88.2, 20.1, 71.0],
        "pickup_hour": [8, 23, 0, 17],
        "pickup_dow": [1, 3, 2, 5],
        "service_type": ["Yellow", "Green", "Yellow", "Green"],
        "pickup_borough": ["Manhattan", "Queens", "Brooklyn", "Bronx"],
        "dropoff_borough": ["Manhattan", "Manhattan", "Queens", "Bronx"],
        "pickup_zone": ["Midtown Center", "JFK Airport", "Bushwick", "Fordham"],
        "dropoff_zone": ["Upper East Side", "Midtown Center", "Astoria", "Fordham"],
        "temp_band": ["Mild", "Hot", "Freezing", "Warm"],
        "is_airport_trip": [0, 1, 0, 0],
        "ratecodeid": [1, 2, 1, 1],
        "humidity": [Decimal("0.610000000"), None, Decimal("0.123456789"),
                     Decimal("0.990000001")],
        "windSpeed": [Decimal("3.250000000"), Decimal("12.100000000"), None,
                      Decimal("0.000000001")],
        "visibility": [Decimal("10.000000000"), Decimal("9.870000000"),
                       Decimal("2.500000000"), None],
    }


@pytest.fixture
def sample_dir(tmp_path):
    """A Spark-style prep sample: a directory of two parquet parts."""
    root = tmp_path / "sample.parquet"
    root.mkdir()
    for part, offset in enumerate((0, 4)):
        table = pa.Table.from_pydict(_rows(offset), schema=SCHEMA)
        pq.write_table(table, root / f"part-0000{part}.snappy.parquet")
    return root


class TestLoadPrepSample:
    def test_drops_exactly_the_unused_columns(self, sample_dir):
        df = load.load_prep_sample(sample_dir)
        expected = [f.name for f in SCHEMA if f.name not in UNUSED]
        assert list(df.columns) == expected

    def test_reads_every_row_of_every_part(self, sample_dir):
        assert len(load.load_prep_sample(sample_dir)) == 8

    def test_decimal_columns_arrive_as_float64(self, sample_dir):
        df = load.load_prep_sample(sample_dir)
        for col in features.DECIMAL_COLUMNS:
            assert df[col].dtype == np.float64

    def test_decimal_values_and_nulls_survive_the_cast(self, sample_dir):
        df = load.load_prep_sample(sample_dir)
        assert df["humidity"].iloc[2] == 0.123456789
        assert df["windSpeed"].iloc[3] == 1e-09
        assert np.isnan(df["humidity"].iloc[1])
        assert np.isnan(df["visibility"].iloc[3])

    def test_build_features_equals_the_read_parquet_path(self, sample_dir):
        X_lean, y_lean = features.build_features(load.load_prep_sample(sample_dir))
        X_full, y_full = features.build_features(pd.read_parquet(sample_dir))
        pd.testing.assert_frame_equal(X_lean, X_full, check_exact=True)
        pd.testing.assert_series_equal(y_lean, y_full, check_exact=True)

    def test_temporal_carve_then_build_equals_the_read_parquet_path(self, sample_dir):
        """03_scale_champion.py carves before it builds; the equality must
        hold on that order too, index labels included."""
        lean, _ = make_temporal_test(load.load_prep_sample(sample_dir))
        full, _ = make_temporal_test(pd.read_parquet(sample_dir))
        X_lean, y_lean = features.build_features(lean)
        X_full, y_full = features.build_features(full)
        pd.testing.assert_frame_equal(X_lean, X_full, check_exact=True)
        pd.testing.assert_series_equal(y_lean, y_full, check_exact=True)

    def test_unused_columns_are_derived_from_the_feature_exclusions(self):
        """The drop list must track features.EXCLUDED_COLUMNS, minus the
        columns build_features or the temporal carve still consume."""
        for col in load.UNUSED_COLUMNS:
            assert col in features.EXCLUDED_COLUMNS
        for col in ("pickup_datetime", "pickup_zone", "dropoff_zone"):
            assert col not in load.UNUSED_COLUMNS
