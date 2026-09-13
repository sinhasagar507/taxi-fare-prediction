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
measured**.

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
   - [ ] **Then set the baseline — still open.** `spark/ml/data/prep_stats.json`
     and the samples on disk still describe the pre-M3 local run. Replacing them
     with the `gs://…/ml/prep` output regenerates the holdout partition, so it is
     its own step.
5. [ ] Then §5.3's out-of-fold encoder, which needs a `tripid` that is actually unique.
