"""
Phase 1 — Spark data prep for the fare-prediction model.

Reads the local `fact_trips/` parquet backup (128.78M rows, the
dbt-built fact table exported from BigQuery), applies the data-quality
guards and cap logic from the ML handoff note, derives the Looker-only calculated
fields (fare_capped, distance_capped, trip_duration_min, temp_band, is_airport_trip),
and writes two stratified samples for downstream single-machine modeling:

    sample_full.parquet   ~10%  (~12.8M rows)  — final refit of the champion model
    sample_work.parquet   ~0.6% (~0.8M rows)   — fast iteration during the model sweep

Spark's job ends here. Everything downstream (feature engineering, the model sweep,
neural nets) runs in pandas / scikit-learn on these samples — no Spark, no GPU, no cloud.

The backup sits OUTSIDE the working tree (audit item 10, 2026-09-01) — by
default in the repository's sibling `nyc_taxi_migration_backup/`. Override with
the `MIGRATION_BACKUP_DIR` environment variable; see `spark/ml/src/paths.py`.

Run (from repo root):
    .venv/bin/python spark/ml/00_prep_spark.py                 # full 128M pass
    .venv/bin/python spark/ml/00_prep_spark.py --limit-files 3 # fast dry-run
    MIGRATION_BACKUP_DIR=/Volumes/ext/backup \
        .venv/bin/python spark/ml/00_prep_spark.py             # backup elsewhere

Design notes (see spark/2026-07-04-ml-handoff-context.md and
spark/2026-07-10-fare-prediction-modeling-plan.md):
  - `fare_capped` / `distance_capped` do NOT exist in fact_trips; they are derived here
    as per-service p99 caps, re-derived from the data (defensible paper trail) rather
    than hard-coded. Sanity target from the dashboard: Yellow fare p99 ~$52 / Green ~$46,
    Yellow distance p99 ~18.76mi / Green ~14.19mi.
  - timestamps are TIMESTAMP_NTZ -> cast via "timestamp" before "long".
  - numeric columns are decimal(38,9) -> cast to double for percentile math.
  - Leakage columns (tip/tolls/mta/extra/improvement_surcharge/total_amount/payment_type)
    are dropped from the sample so they cannot accidentally enter the feature matrix.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# --- paths -------------------------------------------------------------------
# Import the unit-tested path policy by its package path (repo root on sys.path).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spark.ml.src.paths import (  # noqa: E402
    BACKUP_DIR_ENV,
    LOCAL_SOURCE,
    is_bigquery_table,
    resolve_fact_trips_dir,
)
# The URI helpers live in src/mllib.py because that is where their tests live,
# and M4 needs them in both scripts. Neither module imports pyspark.
from spark.ml.src.mllib import (  # noqa: E402
    is_remote_uri,
    join_uri,
    spark_master,
    write_text,
)

# The 7.1 GB backup lives OUTSIDE the working tree since audit item 10
# (2026-09-01). `spark/ml/src/paths.py` owns the resolution rule; set
# MIGRATION_BACKUP_DIR to point the prep at a different disk or mount.
FACT_TRIPS_DIR = resolve_fact_trips_dir()
OUT_DIR = REPO_ROOT / "spark" / "ml" / "data"
DEFAULT_OUTPUT = str(OUT_DIR)
STATS_NAME = "prep_stats.json"

# --- domain constants (from handoff §2/§4) -----------------------------------
# temp_band boundaries in °F: Freezing <32, Cold 32-50, Mild 50-68, Warm 68-85, Hot >85
TEMP_BANDS = ["Freezing", "Cold", "Mild", "Warm", "Hot"]
AIRPORT_RATECODES = (2, 3)  # JFK / Newark flat-rate -> is_airport_trip
DURATION_CAP_MIN_FLOOR = 1.0   # exclude sub-1-min noise along with negative durations

# Columns dropped from the sample: fare components / post-trip fields (leakage per §5)
LEAKAGE_COLS = [
    "tip_amount", "tolls_amount", "mta_tax", "extra", "improvement_surcharge",
    "total_amount", "payment_type", "payment_type_description",
]
# Identifiers / unused columns dropped to keep the sample lean
DROP_COLS = ["tripid", "vendorid", "store_and_fwd_flag", "climate_date", "mjd",
             "pickup_locationid", "dropoff_locationid", "pickup_date"]


def build_spark(driver_mem: str = "6g", master: str | None = "local[*]") -> SparkSession:
    """`master=None` means take it from the environment — Dataproc Serverless.

    Driver memory rides with the local master: it is a launch-time setting, so
    on Serverless the runtime sizes the driver and this value would be an inert
    but misleading claim in the code.
    """
    builder = SparkSession.builder.appName("fare-prep")
    if master is not None:
        builder = builder.master(master).config("spark.driver.memory", driver_mem)
    spark = (
        builder
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load(spark: SparkSession, limit_files: int | None,
         source: str = LOCAL_SOURCE) -> DataFrame:
    """Read `fact_trips` from the local parquet backup or from BigQuery.

    Migration plan M4 runs both and requires them to agree — 128,408,323 rows
    and the same p99 caps. Everything downstream of this function is identical
    for the two, which is what makes that comparison worth anything.
    """
    if is_bigquery_table(source):
        if limit_files is not None:
            raise SystemExit(
                "[error] --limit-files reads N parquet files and means nothing "
                f"against a BigQuery table. Drop it, or use --source {LOCAL_SOURCE}."
            )
        print(f"[load] reading BigQuery table {source} through the connector")
        # A direct table read streams through the Storage Read API and needs no
        # materialisation dataset. `dbt_prod` stays read-only, per M4.
        return spark.read.format("bigquery").option("table", source).load()

    files = sorted(str(p) for p in FACT_TRIPS_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet under {FACT_TRIPS_DIR}. The backup lives outside the "
            f"repository since audit item 10 — set {BACKUP_DIR_ENV} if it is "
            "somewhere other than the repo's sibling directory."
        )
    if limit_files:
        files = files[:limit_files]
    print(f"[load] reading {len(files)} parquet file(s) from {FACT_TRIPS_DIR}")
    return spark.read.parquet(*files)


def derive_and_guard(df: DataFrame) -> DataFrame:
    """Derive fields, apply the §4 data-quality guards. Caps are applied later."""
    # duration in minutes (TIMESTAMP_NTZ -> timestamp -> long seconds)
    dropoff_s = F.col("dropoff_datetime").cast("timestamp").cast("long")
    pickup_s = F.col("pickup_datetime").cast("timestamp").cast("long")

    df = (
        df
        # cast decimals to double for downstream math
        .withColumn("trip_distance", F.col("trip_distance").cast("double"))
        .withColumn("fare_amount", F.col("fare_amount").cast("double"))
        .withColumn("temperature", F.col("highTemp").cast("double"))
        .withColumn("trip_duration_min", (dropoff_s - pickup_s) / 60.0)
        # pre-trip time features
        .withColumn("pickup_hour", F.hour("pickup_datetime"))
        .withColumn("pickup_dow", F.dayofweek("pickup_datetime"))  # 1=Sun..7=Sat
        # airport flat-rate flag (keeps EWR signal without a borough)
        .withColumn("is_airport_trip",
                    F.col("ratecodeid").isin(list(AIRPORT_RATECODES)).cast("int"))
        # passenger_count: valid 1-6; NULL/0/7-9 -> mode (=1) per ANOM-011
        .withColumn("passenger_count",
                    F.when((F.col("passenger_count") >= 1) & (F.col("passenger_count") <= 6),
                           F.col("passenger_count")).otherwise(F.lit(1)).cast("int"))
        # temp_band ordinal from temperature (32/50/68/85 boundaries)
        .withColumn("temp_band",
                    F.when(F.col("temperature") < 32, "Freezing")
                     .when(F.col("temperature") < 50, "Cold")
                     .when(F.col("temperature") < 68, "Mild")
                     .when(F.col("temperature") < 85, "Warm")
                     .otherwise("Hot"))
    )

    # §4 base guards: non-negative fare, positive distance, sane positive duration
    guarded = df.filter(
        (F.col("fare_amount") >= 0)
        & (F.col("trip_distance") > 0)
        & (F.col("trip_duration_min") >= DURATION_CAP_MIN_FLOOR)
    )
    return guarded


def compute_caps(df: DataFrame) -> dict:
    """Per-service p99 caps for fare & distance; global p99 for duration."""
    rows = (
        df.groupBy("service_type")
        .agg(
            F.expr("percentile_approx(fare_amount, 0.99, 1000)").alias("fare_p99"),
            F.expr("percentile_approx(trip_distance, 0.99, 1000)").alias("dist_p99"),
        )
        .collect()
    )
    per_service = {r["service_type"]: {"fare_p99": float(r["fare_p99"]),
                                       "dist_p99": float(r["dist_p99"])} for r in rows}
    dur_p99 = float(
        df.select(F.expr("percentile_approx(trip_duration_min, 0.99, 1000)")).first()[0]
    )
    return {"per_service": per_service, "duration_p99_min": dur_p99}


def _cap_col(value_col: str, key: str, caps: dict):
    """least(value, per-service p99) as a Spark column."""
    col = F.col(value_col)
    expr = None
    for svc, c in caps["per_service"].items():
        capped = F.least(col, F.lit(c[key]))
        cond = F.col("service_type") == svc
        expr = F.when(cond, capped) if expr is None else expr.when(cond, capped)
    return expr.otherwise(col)


def apply_caps(df: DataFrame, caps: dict) -> DataFrame:
    return (
        df
        .withColumn("fare_capped", _cap_col("fare_amount", "fare_p99", caps))
        .withColumn("distance_capped", _cap_col("trip_distance", "dist_p99", caps))
        .withColumn("trip_duration_min",
                    F.least(F.col("trip_duration_min"),
                            F.lit(caps["duration_p99_min"])))
    )


def select_model_columns(df: DataFrame) -> DataFrame:
    keep = [
        # target + raw reference
        "fare_capped", "fare_amount",
        # numeric predictors
        "trip_distance", "distance_capped", "trip_duration_min",
        "passenger_count", "temperature", "pickup_hour", "pickup_dow",
        # categoricals
        "service_type", "pickup_borough", "dropoff_borough",
        "pickup_zone", "dropoff_zone", "temp_band",
        # binary flag + raw ratecode for audit
        "is_airport_trip", "ratecodeid",
        # optional climate (precip confirmed null-effect; kept for exploration only)
        "humidity", "windSpeed", "visibility",
    ]
    return df.select(*keep)


def stratified_sample(df: DataFrame, frac: float, seed: int) -> DataFrame:
    """10%-style stratified sample on service_type × temp_band."""
    keyed = df.withColumn("_strata", F.concat_ws("|", "service_type", "temp_band"))
    strata = [r["_strata"] for r in keyed.select("_strata").distinct().collect()]
    fractions = {s: frac for s in strata}
    return keyed.stat.sampleBy("_strata", fractions, seed).drop("_strata")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-files", type=int, default=None,
                    help="read only the first N parquet files (fast dry-run)")
    ap.add_argument("--full-frac", type=float, default=0.10)
    ap.add_argument("--work-frac", type=float, default=0.06,
                    help="fraction of the FULL sample taken for the work sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--driver-mem", default="6g")
    ap.add_argument("--source", default=LOCAL_SOURCE,
                    help=f"'{LOCAL_SOURCE}' for the parquet backup, or a "
                         "fully-qualified project.dataset.table read through "
                         "the BigQuery connector")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="directory for the two samples and prep_stats.json; "
                         "a local path or a gs:// URI")
    ap.add_argument("--cluster", action="store_true",
                    help="do not set a Spark master; take it from the "
                         "environment (Dataproc Serverless, plan M4)")
    args = ap.parse_args()

    # Refuse a half-recognised source rather than falling through to the local
    # backup and reporting a row count for a table nobody asked for.
    if args.source != LOCAL_SOURCE and not is_bigquery_table(args.source):
        raise SystemExit(
            f"[error] --source must be {LOCAL_SOURCE!r} or a fully-qualified "
            f"project.dataset.table; got {args.source!r}. The project id is "
            "required — see src/paths.is_bigquery_table."
        )

    if not is_remote_uri(args.output):
        Path(args.output).mkdir(parents=True, exist_ok=True)
    spark = build_spark(args.driver_mem,
                        spark_master(None) if args.cluster else "local[*]")
    try:
        raw = load(spark, args.limit_files, args.source)
        total = raw.count()
        print(f"[count] raw rows: {total:,}")

        guarded = derive_and_guard(raw).cache()
        kept = guarded.count()
        print(f"[guard] kept rows: {kept:,}  ({kept / total:.2%} of raw)")

        caps = compute_caps(guarded)
        print(f"[caps] {json.dumps(caps, indent=2)}")

        capped = apply_caps(guarded, caps)
        model_df = select_model_columns(capped)

        full = stratified_sample(model_df, args.full_frac, args.seed).cache()
        full_n = full.count()
        full_uri = join_uri(args.output, "sample_full.parquet")
        full.repartition(8).write.mode("overwrite").parquet(full_uri)
        print(f"[write] sample_full: {full_n:,} rows -> {full_uri}")

        work = stratified_sample(full, args.work_frac, args.seed + 1)
        work_n = work.count()
        work_uri = join_uri(args.output, "sample_work.parquet")
        work.repartition(2).write.mode("overwrite").parquet(work_uri)
        print(f"[write] sample_work: {work_n:,} rows -> {work_uri}")

        stats = {
            "raw_rows": total, "guarded_rows": kept,
            "sample_full_rows": full_n, "sample_work_rows": work_n,
            "full_frac": args.full_frac, "work_frac": args.work_frac,
            "seed": args.seed, "caps": caps,
            "limit_files": args.limit_files,
            # M4 compares a cloud run against a laptop run. Which source the
            # numbers came from is the first thing that comparison needs, and a
            # stats file that does not say is a file you cannot check.
            "source": args.source,
            "output": args.output,
            "master": spark.sparkContext.master,
        }
        stats_uri = join_uri(args.output, STATS_NAME)
        write_text(spark, stats_uri, json.dumps(stats, indent=2))
        print(f"[stats] wrote {stats_uri}")
        print("PREP OK")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
