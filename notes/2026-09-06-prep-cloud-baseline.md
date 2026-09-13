# The prep re-run from `dbt_prod.fact_trips` — root cause of the cap divergence

**Created:** 2026-09-06
**Milestone:** M4, `notes/2026-09-02-gcp-cloud-migration-plan.md`
**Batch:** `prep-m4-20260907-020206`, region `us-central1`, runtime `3.0`
**Re-run:** `prep-m4-20260913-full`, 2026-09-13, on the rebuilt table — see
"The re-run on the rebuilt table" below.

**Invoke when:** you touch `dbt_prod.fact_trips`, the p99 caps, the staging
models, or any modeling result built on either. **Read this before rebuilding
`dbt_prod` or trusting a row count from it.**

> **The cloud caps are rejected. They are an artifact of a defect in
> `stg_yellow_taxi_data.sql` and `stg_green_taxi_data.sql`, not a property of
> the data.** `spark/ml/data/prep_stats.json` therefore still holds the local
> values, which match the raw source. See "Root cause" below.

---

## The finding, in one paragraph

The staging models deduplicate trips on `(vendorid, pickup_datetime)`. That pair
is not a trip identifier. About 2.7 genuinely different yellow trips begin in
NYC per vendor per second, so the dedup discards **62.86% of all yellow trips**.
M3 then made the survivor deterministic by ordering on `tpep_dropoff_datetime`
ascending. The survivor of each group became **the shortest trip in it**. The
p99 caps for distance and duration collapsed, because the long tail was
systematically deleted.

## The measurement

All figures are yellow, in the 2015-01-01 to 2016-12-31 window, after the same
§4 guards the prep applies (`fare_amount >= 0`, `trip_distance > 0`,
`trip_duration_min >= 1.0`).

| Source | Rows | `dist_p99` | `dur_p99` | `fare_p99` |
| --- | ---: | ---: | ---: | ---: |
| **A.** Raw external table, **no dedup** | 274,556,460 | **18.67** | **58.32** | 52.00 |
| **B.** Local backup — pre-M3, arbitrary tiebreak | 100,357,273 | **18.80** | **58.15** | 52.03 |
| **C.** Cloud `dbt_prod` — post-M3, `dropoff ASC` | 99,983,155 | **15.47** | **38.45** | 51.00 |

Read it as follows.

- **A and B agree.** The pre-M3 build kept an *arbitrary* row per group. An
  arbitrary member of a group is an unbiased sample of it, so the distribution
  survived. The local backup reproduces the raw tail to within 0.7%.
- **C is the outlier.** The deterministic order is `tpep_dropoff_datetime` first,
  ascending. Earliest dropoff means shortest duration. Distance follows duration.
- Applying the current dedup to the raw table reproduces C from A directly:
  distance 18.67 → 15.60, duration 58.32 → 38.42, and 172,598,212 rows discarded.

So the divergence is not noise, not `percentile_approx`, and not a changed
archive. It is one `partition by` clause.

### The dedup removes almost nothing that is actually duplicated

Measured over the same yellow window, before any guard:

| Quantity | Rows |
| --- | ---: |
| Raw yellow rows in the window | 277,171,036 |
| Distinct trips on a full-row key | 277,170,766 |
| Groups on the current `(vendorid, pickup)` key | 102,302,402 |

- **True duplicate rows: 270.** That is 0.0001% of the table.
- **Real trips the current dedup destroys: 174,868,364 — 63.09%.**
- A correct dedup emits **2.71x** the yellow rows the current one does.

The source is already almost perfectly unique. The dedup was solving a problem
that does not exist, and paying for it with two thirds of the data. This is the
strongest single number in this note: 270 against 174,868,364.

### Why `fare_p99` did not move

Fare reads 52.00, 52.03 and 51.00 across three very different row sets. A
percentile that will not move across a 63% change in membership is sitting on a
mass point. The 2015–2016 JFK-to-Manhattan flat fare was **$52.00**, which is the
obvious candidate — **INFERRED, not measured.** What is measured is that
`fare_p99` is insensitive here, so it could not have flagged the problem.
Distance and duration did flag it.

### Why the guard pass rate fell

Local 98.99%, cloud 98.21%. The surviving trips are shorter, so more of them
fall under the 1-minute floor. This is a symptom of the same cause.

### Why the row count fell by 373,323

The dedup keeps one row per group either way, so the order alone cannot change
the count. The survivor's **zone pair** changes, and `fact_trips` inner-joins
`dim_zones` with `borough != 'Unknown'`. A different survivor survives that join
differently. The model's own comment states this, and puts the number of
disagreeing groups at 74.4M.

## Root cause

`dbt/ny_taxi_analytics/models/staging/stg_yellow_taxi_data.sql`, and the same
shape in `stg_green_taxi_data.sql`:

```sql
row_number() over(
  partition by vendorid, tpep_pickup_datetime      -- <-- not a trip key
  order by tpep_dropoff_datetime, pulocationid, ...  -- <-- picks the shortest
) as rn
...
where rn = 1
```

Two separate defects, one older than the other:

1. **The partition key is wrong, and always was.** `(vendorid, pickup_datetime)`
   identifies a vendor-second, not a trip. 274.6M yellow rows collapse to 102.0M
   groups — a mean of 2.69 rows per group, which is simply NYC's yellow trip rate
   per vendor per second. The model's own comment concedes the rows are distinct:
   74.4M groups "hold rows that disagree on the zone pair". Identical duplicates
   cannot disagree about anything.
2. **M3's ordering turned a hidden defect into a biased one.** Before M3 the
   damage was invisible, because an arbitrary winner is an unbiased sample.
   M3 fixed reproducibility and, in the same stroke, made the sample the minimum.
   The fix was correct about determinism and wrong about the statistic.

M3's rebuild is therefore not a trustworthy baseline. Its 128,408,323 is
reproducible, and it counts the wrong rows.

## The fix

Deduplicate on the emitted row, not on a vendor-second. Then `rn = 1` removes
only true duplicates and keeps every distinct trip.

- Replace the `partition by vendorid, tpep_pickup_datetime` window with a
  `select distinct` over the projected columns, or partition by every column the
  model emits. Both are deterministic by construction, which keeps M3's goal.
- **`tripid` must change with it.** It is
  `generate_surrogate_key(['vendorid','tpep_pickup_datetime'])`, which is unique
  only because the broken dedup made it so. Once the distinct trips return, that
  key collides 2.69 ways on average. It needs the dropoff timestamp, both
  location ids and the money columns. **§5.3 depends on this**: the out-of-fold
  encoder uses `tripid` as its stable row key.
- Green needs the same change. Its comment already records that `trip_type` and
  `congestion_surcharge` cannot appear in an `order by`, because the external
  table declares INT64 while the parquet stores DOUBLE. Project first, then
  dedupe, and the problem does not arise.

The dbt project is a **submodule**. Per CLAUDE.md the edit lands in the remote
repo (`github.com/sinhasagar507/ny_taxi_analytics`, currently `3f927a3`), then
the pointer moves. Do not edit inside `dbt/ny_taxi_analytics` here.

## What the fix invalidates

Everything downstream of `dbt_prod`, because the fact table roughly triples.

- The M3 count, 128,408,323, and its checksum agreement. All four builds agreed
  with each other and all four dropped the same 63%.
- The cloud prep run of 2026-09-06 and its caps.
- `spark/ml/data/prep_stats.json` is **kept at the local values** for now. They
  match the raw source to within 0.7%, which is the best number in hand, and they
  describe the samples actually on disk. They are still provisional.
- Every modeling result, because `fare_capped` is the target and its cap moves.
- The sealed holdout partition.

## The run that produced this

`spark/ml/00_prep_spark.py` on Dataproc Serverless, reading
`dtc-de-project-506916.dbt_prod.fact_trips` through
`com.google.cloud.spark:spark-4.0-bigquery:0.45.0`. Its driver log printed
`raw_rows` 128,408,323, an exact match to M3, and a guarded count of 126,111,909
that matched an independent BigQuery SQL computation exactly. **The Spark read is
faithful. The table it read is not.**

The batch never wrote output. Its driver log reaches `[caps]` at 02:56 UTC and
has no `[write]`, `[stats]` or `PREP OK` line before the 03:01 cancel. The
`prep_stats.json` under `ml/prep` at the time was dated 2026-09-05 01:25 UTC,
written by an earlier batch, `m4-prep-20260905b`, on the same table with the same
counts and caps. That file is preserved at
`gs://primary-data-dtc-506916/ml/prep-rejected-m4-prep-20260905b/prep_stats.json`;
the 2026-09-13 re-run overwrote `ml/prep`.

The run also carried two new columns, added in the same change: `tripid` for
§5.3's row key and `pickup_datetime` for §4a's temporal split. Both are held out
of the feature matrix by `features.EXCLUDED_COLUMNS`, because `build_features`
drops rather than allows.

### A separate defect the run exposed: `--cluster` does nothing

**Corrected 2026-09-13.** The first version of this section inferred the cause.
Three of its claims were wrong: the `"master": "local[*]"` record came from
another batch, no executor idled because none ever ran, and the job never wrote
its output. The measured account:

- **The `local[*]` record is from `m4-prep-20260905b`, not this batch.** That
  batch's args hold no `--cluster`, so `build_spark` pinned `local[*]` as
  designed. It finished at 01:25:37 UTC; the file is dated 01:25:12.
- **This batch passed `--cluster` and still ran local.** Its billed DCU,
  16,741 DCU-seconds, is 1.03x a driver-only figure (4 vCPU × 0.6 + 22.4 GB × 0.1
  = 4.64 DCU, over 3,488 s running). Every earlier batch in the project matches
  its driver alone the same way, at 1.00–1.03x, including `m4-wiring-195925`,
  which also passed `--cluster`. All 8,019 driver-log lines come from the driver
  node; no worker node ever logged; the RDD cache-spill warnings are in
  `driver.log`, so the driver JVM computed the blocks.
- **Root cause, measured by `probe-master-20260913a`:** Serverless runtime 3.0
  sets the master to `local`. The image's `spark-defaults.conf` declares
  `spark.master=dataproc`, and the service appends `spark.master=local` after it;
  it also exports `MASTER=local`. `build_spark` set no master, so the JVM's
  `spark.master` was `local`: `defaultParallelism` 1, **one thread**, 0 executors.
  That is why the batch ran 57 minutes on 128M rows. Why the service injects
  `local` is **not known** — only that it does.
- **The fix, measured by `probe-master-20260913b`:** passing `dataproc` explicitly
  gave 2 registered executors, and every task ran on the two worker hosts.
  `mllib.spark_master(None)` now returns `DATAPROC_SERVERLESS_MASTER`, and both
  scripts always call `.master()` (commit `24175b0`). `probe-master-20260913c`
  ran the committed code with no override: master `dataproc`, 2 executors.

## Cost

| Item | Measured |
| --- | --- |
| Dataproc Serverless batch | 16,741,285 milliDCU-seconds = 16,741 DCU-seconds, about **$0.28** |
| BigQuery — caps by service and year | 6.67 GiB, about **$0.04** |
| BigQuery — dedup before/after on the raw table | 20.65 GiB, about **$0.13** |
| BigQuery — true-duplicate count, fix validation | 20.65 GiB, about **$0.13** |
| BigQuery — row count and schema check | $0 — table metadata |
| **Total** | **about $0.58** |

Rate from `notes/2026-09-04-cloud-cost-baseline.md`. The DCU bought one driver
running on one thread, because of the `--cluster` defect; no executor was billed.

## The re-run on the rebuilt table (2026-09-13)

`prep-m4-20260913-full`: `00_prep_spark.py` at `24175b0`, runtime 3.0, reading the
D-012 `dbt_prod.fact_trips`. **SUCCEEDED on its own**, no cancel, `PREP OK` in the
driver log. Four executor nodes logged work — the cap set below.

```bash
gcloud dataproc batches submit pyspark \
  gs://primary-data-dtc-506916/dependencies/m4prep-24175b0/00_prep_spark.py \
  --batch=prep-m4-20260913-full --region=us-central1 --version=3.0 \
  --service-account=dataproc-batch@dtc-de-project-506916.iam.gserviceaccount.com \
  --py-files=gs://primary-data-dtc-506916/dependencies/m4prep-24175b0/prep_deps.zip \
  --properties=spark.jars.packages=com.google.cloud.spark:spark-4.0-bigquery:0.45.0,spark.dynamicAllocation.maxExecutors=4 \
  --ttl=90m -- --cluster --source dtc-de-project-506916.dbt_prod.fact_trips \
  --output gs://primary-data-dtc-506916/ml/prep
```

`maxExecutors=4` is a cost cap, not a tuning result. The global quota is 32 vCPUs,
so dynamic allocation could reach 7 executors; 4 keeps a 90-minute TTL run under
$3. `prep_deps.zip` is `spark/ml/src/*.py` plus empty `__init__.py` files — the
repository uses namespace packages, and the zip copies the 2026-09-07 layout.

| Field | Value |
| --- | ---: |
| `raw_rows` | **307,339,039** — exact match to M3's `__TABLES__` count |
| `guarded_rows` | **304,766,876** (99.16%) — exact match to M3's SQL, 270,075,802 + 34,691,074 |
| `sample_full_rows` / `sample_work_rows` | 30,482,494 / 1,828,181 |
| `master` | **`dataproc`** |

Caps, three ways. Local is the pre-M3 backup on disk; BigQuery is M3's
`APPROX_QUANTILES` on the rebuilt table; cloud Spark is this run's
`percentile_approx(…, 0.99, 1000)` on the same table. Duration is per-service in
BigQuery and global in the prep.

| Cap | Local `prep_stats.json` | BigQuery SQL (M3) | Cloud Spark (this run) |
| --- | ---: | ---: | ---: |
| Yellow `fare_p99` | 52.0 | 52.00 | 52.0 |
| Yellow `dist_p99` | 18.7 | 18.51 | 18.5 |
| Green `fare_p99` | 45.0 | 45.00 | **44.5** |
| Green `dist_p99` | 14.15 | 14.15 | **13.9** |
| `duration_p99_min` | 57.57 | 57.75 Y / 59.97 G | 57.5 |

Yellow agrees across all three to 0.2. Green sits **below** BigQuery by 0.5 on fare
and 0.25 on distance, with the same rows under both. Both are approximate
quantiles with different algorithms; the cause of the green gap was **not
measured**. It is an open anomaly that blocks the baseline — see the next section.

| Cost | Measured |
| --- | --- |
| Wall time | 25.6 min (05:46:31–06:12:05 UTC), 24.9 min running |
| Dataproc | 32,053.8 DCU-seconds × $0.06/DCU-hour = **$0.53**; mean 21.5 DCU while running |
| Shuffle storage | 2,337,247 GB-seconds; its price is **UNVERIFIED** and not added |
| BigQuery | `JOBS_BY_PROJECT` holds one job from the batch account, the connector's row count, **0 bytes billed**. The Storage Read API reads are not jobs, so their bytes are **UNVERIFIED** |
| Probes a/b/c | 295.2 + 1,064.4 + 1,202.2 DCU-seconds = **$0.04** |

Against the cancelled batch: 2.4x the rows, 25.6 minutes instead of 57-plus,
finished instead of cancelled, for $0.53 instead of $0.28 that bought nothing
usable.

## Open anomaly — the green caps disagree between Spark and BigQuery (2026-09-13)

**Status: OPEN. It blocks "set the baseline" (Next, item 4).** Found in the re-run
above and recorded the same day.

On the same rows — `guarded_rows` 304,766,876 in both — yellow agrees and green does not:

| Green cap | BigQuery, M3 (`APPROX_QUANTILES`) | Cloud Spark (`percentile_approx`, accuracy 1000) | Gap |
| --- | ---: | ---: | ---: |
| `fare_p99` | 45.00 | 44.5 | −$0.50 (−1.1%) |
| `dist_p99` | 14.15 | 13.9 | −0.25 mi (−1.8%) |

Both gaps point the same way. Yellow's gaps are 0 and 0.01.

**How it was missed.** The goal's pass band for green distance was 13.7–14.6, so 13.9
passed and the run report carried the gap as a footnote. A pass band is not agreement;
the gap should have blocked the next step when it appeared.

**Size — an arithmetic bound, not a measurement.** A p99 cap touches about 1% of green
guarded rows: about 347,000 rows, 0.11% of all guarded rows. Each moves by at most
$0.50 of `fare_capped`, so the mean target moves by at most $0.0006 over all rows and
$0.005 over green. §5.2's measured fold spread is ±0.0015 MAE, on the pre-D-012 data.
The gap cannot move a model comparison. What it can do is make two runs of the prep
disagree.

**Candidate causes — neither is measured:**

1. **The method.** Both engines compute approximate quantiles.
   `percentile_approx(x, 0.99, 1000)` may return any value whose rank lies within
   0.1% of N of the true p99 — for green, ±34,691 rows, so anywhere from p98.9 to
   p99.1. The value that rank error lands on depends on how many trips sit near the
   cap. Yellow's fare cap sits on a large mass point: 52.00 held across three very
   different row sets (D-012). Green's caps move more between row sets — 45.0 / 14.15
   on the old backup, 44.00 / 13.96 on D-012's one-month dev build, 45.00 / 14.15 in
   BigQuery, 44.5 / 13.9 in Spark — which fits a thinner tail. **INFERRED.**
2. **The rows.** The totals agree exactly, but `prep_stats.json` records no
   per-service counts, so offsetting per-service differences are not ruled out.

Cause 1 has a consequence of its own. An approximate cap can change with the
partitioning, so the laptop and the cloud can disagree on identical data, and the
target moves with them. M4's gate, "prep stats identical", cannot be relied on while
the caps are approximate.

**The check that decides it — Q1, 16.00 GiB by dry run, about $0.10.** Exact p99 per
service from a value histogram and a running count, using the discrete definition: the
smallest observed value with at least 99% of rows at or below it. It also returns the
exact p98.9–p99.1 band and the exact share of rows at or below each engine's cap. If
44.5 and 13.9 fall inside green's band, cause 1 holds and nothing is broken. If either
falls outside, it is cause 2 or a code defect, and that comes before anything else.

**Also before the baseline — Q3, at most about $0.15, estimated.** The dry run cannot
price external tables; the estimate is bounded by the 20.65 GiB an earlier full-row
query on the raw yellow table billed. Green was never checked against its raw source —
D-012 validated yellow only. Q3 reconciles raw → `fact_trips` for both services, and
tests whether the zone join (`borough != 'Unknown'`) is the whole yellow gap of
4,480,658 guarded rows (1.63%, which M3 left unmeasured). The baseline freezes the rows,
so a green defect found after it would force a redo.

**Then one decision:**

- **A — keep the Spark caps.** No code change and no re-run: the baseline comes from
  the 2026-09-13 output. A future run is gated on the exact band, not on equality.
- **B — exact caps in the prep.** Test first; per-service counts added to
  `prep_stats.json`; one re-run, about $0.53 as measured on 2026-09-13; gated on exact
  equality with Q1. Worth it if the laptop and the cloud must agree, or if the prep
  will run again.

<details><summary>The prepared queries — Q1 and Q3 for this anomaly, Q2 for §5.3</summary>

All three dry-ran clean on 2026-09-13. None has been run.

```sql
-- Q1. Exact p99 caps on dbt_prod.fact_trips, the §4 guards the prep applies.
-- p99 = the smallest value v with count(x <= v) >= 0.99 * N (discrete).
-- p989 / p991 bound what percentile_approx(x, 0.99, 1000) may return.
WITH g AS (
  SELECT service_type,
         CAST(fare_amount AS FLOAT64)   AS fare,
         CAST(trip_distance AS FLOAT64) AS dist,
         TIMESTAMP_DIFF(dropoff_datetime, pickup_datetime, SECOND) / 60.0 AS dur
  FROM `dtc-de-project-506916.dbt_prod.fact_trips`
  WHERE fare_amount >= 0
    AND trip_distance > 0
    AND TIMESTAMP_DIFF(dropoff_datetime, pickup_datetime, SECOND) >= 60
),
hist AS (
  SELECT 'fare' AS metric, service_type AS grp, fare AS v, COUNT(*) AS n FROM g GROUP BY 1, 2, 3
  UNION ALL SELECT 'dist', service_type, dist, COUNT(*) FROM g GROUP BY 1, 2, 3
  UNION ALL SELECT 'dur', service_type, dur, COUNT(*) FROM g GROUP BY 1, 2, 3
  UNION ALL SELECT 'dur', 'All', dur, COUNT(*) FROM g GROUP BY 1, 2, 3
),
cum AS (
  SELECT metric, grp, v, n,
         SUM(n) OVER (PARTITION BY metric, grp ORDER BY v) AS cum_n,
         SUM(n) OVER (PARTITION BY metric, grp) AS tot
  FROM hist
),
observed AS (
  SELECT * FROM UNNEST([
    STRUCT('fare' AS metric, 'Yellow' AS grp, 52.0 AS spark, 52.00 AS bq),
    STRUCT('fare', 'Green',  44.5, 45.00),
    STRUCT('dist', 'Yellow', 18.5, 18.51),
    STRUCT('dist', 'Green',  13.9, 14.15),
    STRUCT('dur',  'Yellow', NULL, 57.75),
    STRUCT('dur',  'Green',  NULL, 59.97),
    STRUCT('dur',  'All',    57.5, NULL)
  ])
)
SELECT c.metric, c.grp, ANY_VALUE(c.tot) AS guarded_rows,
       COUNT(*) AS distinct_values,
       MIN(IF(c.cum_n >= 0.989 * c.tot, c.v, NULL)) AS p989,
       MIN(IF(c.cum_n >= 0.99  * c.tot, c.v, NULL)) AS p99_exact,
       MIN(IF(c.cum_n >= 0.991 * c.tot, c.v, NULL)) AS p991,
       ANY_VALUE(o.spark) AS spark_cap, ANY_VALUE(o.bq) AS bq_cap,
       MAX(IF(c.v <= o.spark, c.cum_n, 0)) / ANY_VALUE(c.tot) AS cdf_at_spark,
       MAX(IF(c.v <= o.bq, c.cum_n, 0)) / ANY_VALUE(c.tot) AS cdf_at_bq
FROM cum c JOIN observed o USING (metric, grp)
GROUP BY 1, 2 ORDER BY 1, 2;

-- Q3. Raw source -> fact_trips, both services, the same guards, split by whether
-- both zones are known, with exact p99s on the zones-known rows.
WITH zones AS (
  SELECT locationid, borough FROM `dtc-de-project-506916.dbt_prod.dim_zones`
),
raw AS (
  SELECT 'Yellow' AS service_type,
         CAST(fare_amount AS FLOAT64) AS fare, CAST(trip_distance AS FLOAT64) AS dist,
         TIMESTAMP_DIFF(CAST(tpep_dropoff_datetime AS TIMESTAMP),
                        CAST(tpep_pickup_datetime AS TIMESTAMP), SECOND) AS dur_s,
         CAST(pulocationid AS INT64) AS pu, CAST(dolocationid AS INT64) AS do_
  FROM `dtc-de-project-506916.nyc_taxi_data.yellow_taxi_external_table`
  WHERE vendorid IS NOT NULL
    AND CAST(tpep_pickup_datetime AS DATE) BETWEEN '2015-01-01' AND '2016-12-31'
  UNION ALL
  SELECT 'Green',
         CAST(fare_amount AS FLOAT64), CAST(trip_distance AS FLOAT64),
         TIMESTAMP_DIFF(CAST(lpep_dropoff_datetime AS TIMESTAMP),
                        CAST(lpep_pickup_datetime AS TIMESTAMP), SECOND),
         CAST(pulocationid AS INT64), CAST(dolocationid AS INT64)
  FROM `dtc-de-project-506916.nyc_taxi_data.green_taxi_external_table`
  WHERE vendorid IS NOT NULL
    AND CAST(lpep_pickup_datetime AS DATE) BETWEEN '2015-01-01' AND '2016-12-31'
),
guarded AS (
  SELECT r.*,
         (pz.borough IS NOT NULL AND pz.borough != 'Unknown'
          AND dz.borough IS NOT NULL AND dz.borough != 'Unknown') AS zones_known
  FROM raw r
  LEFT JOIN zones pz ON r.pu = pz.locationid
  LEFT JOIN zones dz ON r.do_ = dz.locationid
  WHERE r.fare >= 0 AND r.dist > 0 AND r.dur_s >= 60
),
counts AS (
  SELECT service_type, COUNT(*) AS guarded_raw,
         COUNTIF(zones_known) AS guarded_raw_zones_known,
         COUNTIF(NOT zones_known) AS guarded_raw_zones_unknown
  FROM guarded GROUP BY 1
),
hist AS (
  SELECT 'fare' AS metric, service_type AS grp, fare AS v, COUNT(*) AS n
  FROM guarded WHERE zones_known GROUP BY 1, 2, 3
  UNION ALL SELECT 'dist', service_type, dist, COUNT(*) FROM guarded WHERE zones_known GROUP BY 1, 2, 3
  UNION ALL SELECT 'dur', service_type, dur_s / 60.0, COUNT(*) FROM guarded WHERE zones_known GROUP BY 1, 2, 3
),
cum AS (
  SELECT metric, grp, v,
         SUM(n) OVER (PARTITION BY metric, grp ORDER BY v) AS cum_n,
         SUM(n) OVER (PARTITION BY metric, grp) AS tot
  FROM hist
),
p99 AS (
  SELECT grp AS service_type,
         MIN(IF(metric = 'fare' AND cum_n >= 0.99 * tot, v, NULL)) AS fare_p99_raw,
         MIN(IF(metric = 'dist' AND cum_n >= 0.99 * tot, v, NULL)) AS dist_p99_raw,
         MIN(IF(metric = 'dur'  AND cum_n >= 0.99 * tot, v, NULL)) AS dur_p99_raw
  FROM cum GROUP BY 1
)
SELECT * FROM counts JOIN p99 USING (service_type) ORDER BY 1;

-- Q2, for §5.3 (11.99 GiB, about $0.07). schema.yml claims tripid is unique in
-- fact_trips, but only the staging models test it.
SELECT COUNT(*) AS n_rows, COUNT(DISTINCT tripid) AS n_distinct_tripid,
       COUNT(DISTINCT CONCAT(service_type, '|', tripid)) AS n_distinct_service_tripid
FROM `dtc-de-project-506916.dbt_prod.fact_trips`;
-- Q2b, free: one climate row per date, or a fan-out on the left join?
SELECT COUNT(*) AS n_rows, COUNT(DISTINCT climate_date) AS n_dates
FROM `dtc-de-project-506916.dbt_prod.stg_climate_data`;
-- Q2c (4.58 GiB, about $0.03): sub-second timestamps would make the prep's
-- duration and TIMESTAMP_DIFF disagree by one second.
SELECT COUNTIF(MOD(UNIX_MICROS(pickup_datetime), 1000000) != 0) AS pickup_subsecond,
       COUNTIF(MOD(UNIX_MICROS(dropoff_datetime), 1000000) != 0) AS dropoff_subsecond
FROM `dtc-de-project-506916.dbt_prod.fact_trips`;
```

</details>

### The detour, and the way back to the plan

The detour inserts one step before an existing one. It adds nothing after it.

1. **Q1 + Q3**, one analysis, at most about $0.25.
2. **A or B.** A costs nothing further. B is a test-first change and one re-run.
3. **Back on plan:** Next item 4, "set the baseline" → M4 smoke → §5.3 → §5.4 → §5c →
   M5 → M6, as the migration plan's Status list orders them.

Steps 1 and 2's evidence run as one goal command. Paste it as written; it is 3,815
characters, under the 4,000 limit. It stops at the A/B decision, which is the owner's.

<details><summary>The GREENCAP goal command</summary>

```text
/goal Measure and record the green cap anomaly; ready the A/B decision for the owner. MET only when the transcript shows a final report headed "GREENCAP DONE" with all of:
(a) Q2b: climate rows = distinct dates (no fan-out);
(b) Q3 per service: guarded raw, zones-known, zones-unknown, zones-known minus fact_trips guarded; whether the zone join explains the yellow 4,480,658 gap; exact raw p99s;
(c) Q1 per cap: guarded rows, distinct values, p989, exact p99, p991, Spark and BigQuery caps, exact CDF at each;
(d) a verdict per green cap: IN BAND (p989 <= Spark cap <= p991) or OUT OF BAND, with numbers;
(e) bytes billed and dollars per query from JOBS_BY_PROJECT at $6.25/TiB;
(f) a recommendation, A or B, or "defect, no A/B" if any cap is OUT OF BAND or Q3 fails to reconcile; the owner decides;
(g) two commits: today's anomaly note, then the results plus the migration plan pointer;
(h) gate before and after: 343 passed, 1 skipped, 0 failed;
(i) git status clean; git log origin/refactor/wire-pipeline..HEAD shows the new commits unpushed.
Judge IMPOSSIBLE if a line starts "GREENCAP STOPPED:". Stop after 25 turns.

FACTS
- The owner suspends the CLAUDE.md per-step review for this goal.
- Read first: notes/2026-09-06-prep-cloud-baseline.md, "Open anomaly" and its <details> SQL; then notes/decisions.md.
- BigQuery: .venv/bin/python, google-cloud-bigquery, GOOGLE_APPLICATION_CREDENTIALS=secrets/gcp-credentials.json, project dtc-de-project-506916. One statement per call; a semicolon in an SQL comment breaks a naive split.
- Dry runs 2026-09-13: Q1 16.00 GiB (~$0.10), Q2b 0. Q3 reads external tables, so its dry run shows 0; estimate <= ~$0.15.
- fact_trips: modified 2026-09-12T23:43:45Z, 307,339,039 rows.
- M3 guarded: Yellow 270,075,802, Green 34,691,074. Spark caps Y 52.0/18.5, G 44.5/13.9, dur 57.5 global. BigQuery Y 52.00/18.51/57.75, G 45.00/14.15/59.97.
- Yellow raw guarded 274,556,460; yellow true duplicates 270; green duplicates not measured.
- Gate: .venv/bin/pytest tests/ --ignore=tests/unit/ml/test_oof_encode.py. Last: 343 passed, 1 skipped.

STEPS
1. Gate. Commit the uncommitted edit to notes/2026-09-06-prep-cloud-baseline.md as it stands.
2. Check fact_trips modified time and rows against FACTS. Run Q2b.
3. Dry-run, then run Q3. Compare zones-known guarded with M3 guarded.
4. Dry-run, then run Q1. Verdict per green cap.
5. Bytes billed per job from JOBS_BY_PROJECT.
6. Docs, measured numbers only (D-009): the anomaly section gets results, verdicts, Status (RESOLVED-as-method or DEFECT) and the Next-list line. Migration plan: the M4 Status line points at the anomaly, and the M4 gate compares to M3's BigQuery numbers, not the local prep_stats.json. Commit.
7. Final gate. Print "GREENCAP DONE" with (a)-(i) and the next step: the owner's A/B decision, then "set the baseline".

PRINT "GREENCAP STOPPED: <reason>" AND END WHEN
- fact_trips modified time or rows differ from FACTS;
- Q2b shows more rows than dates;
- Q1 guarded rows per service differ from M3;
- Q3 yellow zones-known minus 270,075,802 is outside 0..270, or green's is below 0 or above 3,469 (0.01%); report, do not explain away;
- a dry run exceeds its estimate 2x, or BigQuery spend passes $1;
- the gate shows a new failure;
- an action conflicts with a LOCKED entry in notes/decisions.md;
- the same auth or tool failure happens twice.

RULES
- Stay on refactor/wire-pipeline. Never push. No Co-Authored-By.
- BigQuery read-only, one query at a time. No Dataproc batches. Docs only; no code or test changes.
- dbt_prod read-only; never rebuild. Leave spark/ml/data/prep_stats.json and the samples.
- Do not choose A or B or start the baseline; both are the owner's.
- Out of scope: Q2, Q2c, exact-cap code, hash sampling, §5.3, 01_mllib_baseline.py, CASE_STUDY.md (D-005), terraform apply, dbt edits.
```

</details>

The same analysis found other items. None joins the detour; each rides with the plan
step it already belongs to.

| Item | Rides with | Why there |
| --- | --- | --- |
| The samples, 30,482,494 and 1,828,181 rows, exceed modeling plan §8's tiers (~12.8M and ~500K–1M) | Set the baseline | The baseline fixes the sizes, and §8 states the intent |
| §4a's deferred temporal test set: `pickup_datetime` is now in the samples (`4985c03`), so its blocker is gone | Set the baseline | §4a says "when the prep is next re-run" |
| The local backup holds pre-D-012 data, and `--source local` is still the default | Set the baseline | Once the baseline comes from the cloud, the local source is stale |
| `tripid` uniqueness in `fact_trips` is untested — Q2, Q2b, Q2c | §5.3, first step | §5.3's fold key needs a unique row key |
| The red test's row key (crc32 over features) protects the 0.5202 and 0.4828 baselines, which D-012 voided | §5.3, first step | The deviation's reason is gone once the baseline regenerates the samples |
| Sample membership may depend on partitioning: `sampleBy` draws `rand`. **UNVERIFIED** on this pipeline | Option B's change, or any prep re-run after the baseline | A re-run would change the holdout. Hash sampling on the row key removes the dependence |
| The duration cap is global in the prep and per-service in M3 | Option B only | Q1 measures both |
| M4's gate compares against the local `prep_stats.json`, which is pre-D-012 | The migration plan's M4 text | The comparison target is now M3's BigQuery numbers |

## Next

1. [x] Fix the two staging models upstream, and the `tripid` surrogate key with them.
   **Done 2026-09-12:** `sinhasagar507/ny_taxi_analytics#11` merged as `25f3186`;
   pointer bumped in `c25fb96`.
2. [x] Rebuild `dbt_prod`. Yellow grows **2.71x**, measured. Green is the same defect
   but its factor is **NOT MEASURED** — green is lower volume, so fewer trips
   collide per vendor-second, and its factor is probably smaller. Budget the
   rebuild for a fact table of roughly 300M+ rows, not 128M. Re-measure the caps.
   **Done 2026-09-12:** `fact_trips` = 307,339,039 rows, 130.81 GiB. Guarded rows
   grew **2.70x** on yellow (99,983,155 → 270,075,802) and **1.33x** on green
   (26,128,754 → 34,691,074; the old green figure is 126,111,909 − 99,983,155).
   Caps, same guards and percentile form as the table above:

   | Service | `fare_p99` | `dist_p99` | `dur_p99` |
   | --- | ---: | ---: | ---: |
   | Yellow | 52.00 | **18.51** | **57.75** |
   | Green | 45.00 | 14.15 | 59.97 |

   Yellow returns to the raw source (row A: 18.67, 58.32) within 1.0%. Build $1.56,
   measured. Full record: M3 in `notes/2026-09-02-gcp-cloud-migration-plan.md`.
3. [x] Fix `--cluster` before the next batch, so it uses its executors.
   **Done 2026-09-13:** `24175b0`; root cause and probes in the defect section above.
4. [x] Re-run the prep. **Done 2026-09-13:** `prep-m4-20260913-full`, section above.
   - [ ] **First, resolve the green cap anomaly** — "Open anomaly" above: Q1 + Q3,
     then decision A or B. This is the only detour; it blocks the next line.
   - [ ] **Then set the baseline — still open.** It also settles the sample sizes
     against modeling plan §8, §4a's temporal test set, and the local source default
     — see the table under "The detour, and the way back". `spark/ml/data/prep_stats.json`
     and the samples on disk still describe the pre-M3 local run. Replacing them
     with the `gs://…/ml/prep` output regenerates the holdout partition, so it is
     its own step.
5. [ ] Then §5.3's out-of-fold encoder, which needs a `tripid` that is actually unique.
