"""Pure feature-engineering functions for the fare-prediction model (Phase 2).

Everything here is a pure pandas-in / pandas-out function so it can be
unit-tested (tests/unit/ml/test_features.py) and composed inside sklearn
pipelines later. No I/O, no fitting, no global state.

Schema contract (verified against spark/ml/data/sample_work.parquet):
climate columns humidity/windSpeed/visibility arrive as python Decimal
objects (pandas dtype 'object') and are cast to float64 here — decision
locked 2026-07-10: cast, don't drop.
"""

import numpy as np
import pandas as pd

# Climate columns that arrive as Decimal objects from the Spark prep output.
DECIMAL_COLUMNS = ["humidity", "windSpeed", "visibility"]

# Ordinal order for temp_band (coldest → hottest).
TEMP_BAND_ORDER = ["Freezing", "Cold", "Mild", "Warm", "Hot"]

# Columns that must never reach the model. Two distinct reasons:
#
# (a) SUPERSEDED ORIGINALS — the prep emits both halves of a raw/derived pair
#     and only the derived half is a model input. Keeping both would feed the
#     model a column and its own transform.
#     - fare_amount:   uncapped target (fare_capped is derived from it) — also
#                      pure leakage, so it would be excluded either way
#     - trip_distance: uncapped distance; distance_capped is the p99-winsorized
#                      version the prep derives per §2 of the modeling plan.
#                      sample_work carries a corrupt 8,003,318-mile odometer
#                      reading that survives every guard except this cap; left
#                      in, it drove a linear-model fold to a $9.29M prediction.
#     - temperature:   superseded by temp_band_ord (banded at 32/50/68/85°F)
#     - ratecodeid:    consumed into is_airport_trip upstream
#     - pickup_zone / dropoff_zone: consumed into the od_corridor key
#     (pickup_hour and temp_band are the same kind of exclusion, but they are
#      dropped by encode_cyclic_hour / encode_temp_band at the point of
#      transform rather than listed here.)
#
# (b) §5 hard-leakage list from the modeling plan — post-trip fields, absent
#     from the prep output but listed defensively in case a future prep
#     re-adds them.
EXCLUDED_COLUMNS = [
    # (a) superseded originals
    "fare_amount",
    "trip_distance",
    "temperature",
    "ratecodeid",
    "pickup_zone",
    "dropoff_zone",
    # (b) hard leakage
    "tip_amount",
    "total_amount",
    "mta_tax",
    "tolls_amount",
    "improvement_surcharge",
    "extra",
    "revenue_per_mile",
    "has_tip",
    "tip_pct_of_fare",
    "payment_type",
    # (c) carried by the prep on purpose, never model inputs. `build_features`
    #     drops rather than allows, so a column the prep adds becomes a feature
    #     unless it is named here. `tripid` is an identifier; `pickup_datetime`
    #     is a raw timestamp that would also break the numeric preprocessor.
    "tripid",
    "pickup_datetime",
]

# Object-dtype columns that are legitimately categorical model inputs
# (encoded downstream: OHE for the low-cardinality ones, cross-fitted
# target encoding for od_corridor).
CATEGORICAL_COLUMNS = [
    "service_type",
    "pickup_borough",
    "dropoff_borough",
    "od_corridor",
]

TARGET = "fare_capped"


def cast_decimal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast the Decimal-typed climate columns to float64 (None → NaN)."""
    out = df.copy()
    for col in DECIMAL_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col]).astype(np.float64)
    return out


def encode_cyclic_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Replace pickup_hour with sin/cos so 23h and 0h are adjacent."""
    out = df.copy()
    radians = 2 * np.pi * out["pickup_hour"].astype(np.float64) / 24.0
    out["pickup_hour_sin"] = np.sin(radians)
    out["pickup_hour_cos"] = np.cos(radians)
    return out.drop(columns=["pickup_hour"])


def encode_temp_band(df: pd.DataFrame) -> pd.DataFrame:
    """Replace temp_band with its ordinal code (Freezing=0 … Hot=4)."""
    unknown = set(df["temp_band"].dropna()) - set(TEMP_BAND_ORDER)
    if unknown:
        raise ValueError(f"Unknown temp_band values: {sorted(unknown)}")
    mapping = {band: i for i, band in enumerate(TEMP_BAND_ORDER)}
    out = df.copy()
    out["temp_band_ord"] = out["temp_band"].map(mapping)
    return out.drop(columns=["temp_band"])


def add_od_corridor(df: pd.DataFrame) -> pd.DataFrame:
    """Add the directional origin→destination corridor key (high-cardinality
    categorical, target-encoded downstream)."""
    out = df.copy()
    out["od_corridor"] = out["pickup_zone"] + "→" + out["dropoff_zone"]
    return out


def build_features(
    df: pd.DataFrame, include_duration: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble the leakage-safe feature frame.

    Applies the Decimal cast, cyclic hour encoding, temp_band ordinal and
    od_corridor key, then drops the target and every excluded column.
    `include_duration` is the plan's on/off ablation flag for
    trip_duration_min (estimated-duration framing — legitimate feature,
    final call deferred to the ablation run).

    Returns (X, y) with y = fare_capped.
    """
    out = cast_decimal_columns(df)
    out = encode_cyclic_hour(out)
    out = encode_temp_band(out)
    out = add_od_corridor(out)

    y = out[TARGET]
    drop = [TARGET, *EXCLUDED_COLUMNS]
    if not include_duration:
        drop.append("trip_duration_min")
    X = out.drop(columns=drop, errors="ignore")
    return X, y
