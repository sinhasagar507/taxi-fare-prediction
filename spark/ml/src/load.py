"""Lean loader for a prep sample (modeling plan §6, "RESCALE").

`pd.read_parquet` on `sample_full` loads all 22 columns as pandas objects,
and the SCALE goal's 32 GiB VM was OOM-killed at that line, before
`build_features` ran. This loader cuts the peak at the Arrow layer, before
any pandas object exists:

- it projects away the columns `build_features` drops and nothing before it
  reads (`UNUSED_COLUMNS`), so they are never decoded;
- it casts the decimal128 climate columns to float64 in Arrow, so they never
  become per-row python `Decimal` objects.

The frame it returns goes through `make_temporal_test` and `build_features`
unchanged, and gives the same (X, y) as the `pd.read_parquet` path
(tests/unit/ml/test_load.py).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from spark.ml.src.features import DECIMAL_COLUMNS, EXCLUDED_COLUMNS

# Excluded from X, but read before the drop: the temporal carve needs
# pickup_datetime, and add_od_corridor builds the key from the two zones.
CONSUMED_BEFORE_DROP = ("pickup_datetime", "pickup_zone", "dropoff_zone")

UNUSED_COLUMNS = [c for c in EXCLUDED_COLUMNS if c not in CONSUMED_BEFORE_DROP]


def load_prep_sample(path: str | Path) -> pd.DataFrame:
    """Read a prep sample (a file or a Spark part directory) with only the
    columns the feature path uses, decimal climate columns as float64."""
    schema = pq.ParquetDataset(path).schema
    columns = [name for name in schema.names if name not in UNUSED_COLUMNS]
    table = pq.read_table(path, columns=columns)

    # Through the exact decimal string, not a direct cast: pyarrow 21's
    # decimal128 -> float64 cast is off by one ULP on values such as
    # 0.123456789 (measured 2026-09-25), while its string parser rounds
    # correctly and so matches pandas' float(Decimal) on the read_parquet path.
    for col in DECIMAL_COLUMNS:
        if col in table.column_names:
            i = table.schema.get_field_index(col)
            as_float = table.column(col).cast(pa.string()).cast(pa.float64())
            table = table.set_column(i, col, as_float)

    # self_destruct frees each Arrow column as it converts; split_blocks keeps
    # pandas from consolidating same-dtype columns into one fresh copy.
    return table.to_pandas(self_destruct=True, split_blocks=True)
