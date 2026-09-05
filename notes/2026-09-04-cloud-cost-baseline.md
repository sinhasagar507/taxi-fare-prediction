# Cloud cost baseline — measured before any optimisation

**Invoke when:** any cost question, any optimisation of the BigQuery or dbt workflow, or
writing up the cost-reduction work for a case study or resume. This file is the **before**
number. Do not re-derive it; measure the **after** the same way and compare.

**Created:** 2026-09-04
**Project:** `dtc-de-project-506916`, billing account `01E445-18A569-B9097E`
**Measured from:** `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`, 30-day window, plus
`__TABLES__` for storage and `gcloud storage du` for GCS. The metadata queries used to
produce this table are themselves free.

**Why this file exists.** The owner intends to optimise this workflow and write the result
up. A saving is only claimable against a number that was measured before the change, by a
method that can be re-run afterwards. That method is recorded below with the figures, so
the "after" is comparable rather than merely smaller.

---

## The total

| Item | Cost | Basis |
| --- | ---: | --- |
| BigQuery query | **$20.48** | 3.28 TiB billed, 265 billed jobs, at $6.25/TiB |
| BigQuery storage | ~$0.20 | 109.11 GiB active, prorated over the days held |
| GCS storage | ~$0.03 | 11.16 GiB (11,983,090,018 bytes), US multi-region |
| Dataproc Serverless | **$0.0188** | 1,128.8 DCU-seconds across all M4 batches |
| **Total** | **~$20.7** | |

Dataproc is 0.09% of the bill. Every conclusion below is about BigQuery.

## By day

| Date (UTC) | Jobs | TiB billed | Cost |
| --- | ---: | ---: | ---: |
| 2026-09-01 | 6 | 0.0000 | $0.00 |
| 2026-09-02 | 32 | 0.0000 | $0.00 |
| 2026-09-03 | 378 | 2.6356 | **$16.47** |
| 2026-09-04 | 103 | 0.6409 | $4.01 |
| 2026-09-05 | 10 | 0.0000 | $0.00 |

2026-09-03 is M3, the day `dbt_prod` was rebuilt and verified.

## The finding: verification cost 5x what building cost

| Statement type | Jobs | TiB billed | Cost |
| --- | ---: | ---: | ---: |
| `SELECT` | 237 | 2.7743 | **$17.34** |
| `CREATE_TABLE_AS_SELECT` | 27 | 0.5022 | $3.14 |

Building the marts cost $3.14. Checking them cost $17.34 — the checksums across four
builds, the row counts, and the non-deterministic-tiebreak investigation M3 records.

The mechanism is simple and worth stating plainly, because it is the thing to fix.
`fact_trips` is 54.56 GiB. A bare `COUNT(*)` on a native BigQuery table is answered from
metadata and is free. Any `SELECT` carrying an expression — a checksum, a `SUM`, a
`GROUP BY` — scans the whole table and bills 54.56 GiB. The M3 verification ran that
pattern repeatedly.

## By identity

| Identity | Jobs | TiB billed | Cost |
| --- | ---: | ---: | ---: |
| `dtc-de-course@…iam.gserviceaccount.com` | 201 | 2.8831 | $18.02 |
| `saggysimmba@gmail.com` | 64 | 0.3934 | $2.46 |

The service account figure is the dbt CLI, the Airflow DAG and the scripted checks. The
user figure is ad-hoc console and CLI work.

## Storage, by dataset

| Dataset | Size | Tables |
| --- | ---: | ---: |
| `dbt_prod` | 54.56 GiB | 7 |
| `dbt_ci` | **54.56 GiB** | 7 |
| `dbt_dev` | 0.00 GiB | 0 |
| `nyc_taxi_data` | 0.00 GiB | 3 |
| `nyc_climate_data` | 0.00 GiB | 1 |
| **Total** | **109.11 GiB** | |

`dbt_ci` holds a byte-for-byte-sized duplicate of `dbt_prod`, left behind by a CI run.
Nothing reads it between runs. It is ~$1.09/month for nothing, and it is half of all
BigQuery storage in the project.

## Candidate optimisations — not yet done, not yet costed

Recorded so the "after" measurement has a hypothesis to test. None of these has been
applied, and `dbt_prod` is read-only from M4 onward.

1. **Stop full-scan verification.** Use `COUNT(*)` (free, metadata) and
   `INFORMATION_SCHEMA.TABLE_STORAGE` for row counts and sizes. Reserve a checksum for a
   partition or a sample, not the whole 54.56 GiB table. This is the $17.34 line.
2. **Drop `dbt_ci` after each CI run**, or point CI at a partition-limited subset. Halves
   storage.
3. **Partition and cluster `fact_trips`** on the pickup date. A verification or a dashboard
   query then scans one partition instead of 54.56 GiB. This changes a dbt model in the
   submodule, so it is upstream work.
4. **Use the BigQuery Storage Read API for Spark**, which M4 already proved works and does
   not bill query bytes.

## How to re-measure

```sql
-- cost by day
SELECT DATE(creation_time) d, COUNT(*) jobs,
       SUM(total_bytes_billed)/POW(2,40) tib_billed,
       SUM(total_bytes_billed)/POW(2,40)*6.25 usd
FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY d ORDER BY d;

-- build vs verify
SELECT statement_type, COUNT(*) jobs,
       SUM(total_bytes_billed)/POW(2,40)*6.25 usd
FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND total_bytes_billed > 0
GROUP BY statement_type ORDER BY usd DESC;

-- storage by dataset (run per dataset)
SELECT SUM(size_bytes)/POW(2,30) gib, COUNT(*) tables
FROM `dtc-de-project-506916.<dataset>.__TABLES__`;
```

Dataproc spend, which needs no SQL:

```bash
gcloud dataproc batches list --region us-central1 --project dtc-de-project-506916 \
  --format="value(name.basename(),state,runtimeInfo.approximateUsage.milliDcuSeconds)"
# cost = milliDcuSeconds / 1000 * 0.06 / 3600
```

**Unit prices used:** BigQuery on-demand query $6.25/TiB; BigQuery active storage
$0.02/GiB/month; GCS Standard US multi-region $0.026/GiB/month; Dataproc Serverless
$0.06/DCU-hour. Confirm each against the current pricing pages before quoting a saving.
