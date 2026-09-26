# GCP Infrastructure — reference layout

**Invoke when:** you need the bucket or dataset layout, the test layout, or the E2E smoke steps.

Moved out of CLAUDE.md 2026-08-22. **The project is provisioned and live since M1/M2
(2026-09-02/03).** Everything below was measured against the cloud on 2026-09-03, not
copied from a design document.

The dbt-built marts are held locally, **outside the repository** since audit item 10
(2026-09-01), in the sibling `nyc_taxi_migration_backup/`; `spark/ml/src/paths.py`
resolves that path and `MIGRATION_BACKUP_DIR` overrides it. That local copy is what lets
the fare model run with no cloud.

To point the pipeline at a project of your own, supply a keyfile via
`GOOGLE_APPLICATION_CREDENTIALS` and set `GCP_PROJECT_ID` / `GCP_GCS_BUCKET`. The layout
below then reads as a structural reference for the data model.

## GCS bucket layout

Bucket `primary-data-dtc-506916` (location US, uniform bucket-level access on):

- `nyc_taxi_data/yellow_taxi_data/` — 24 parquet files (2015-01 … 2016-12), 3,860,122,756 B
- `nyc_taxi_data/green_taxi_data/`  — 24 parquet files (2015-01 … 2016-12), 540,892,554 B
- `nyc_taxi_data/taxi_lookup_data/taxi_zone_lookup.csv` — 12,331 B
- `nyc_climate_data/climate_data.parquet` — 1,037,285 B
- `dbt_prod_restore/fact_trips/` — 204 objects, 7.053 GiB. **The `fact_trips` restore.
  Ingest never writes here. Do not delete it as part of an ingest rollback.**

Per-file sizes are in the archive fingerprint below.

## BigQuery datasets

Five datasets, all at location `US`.

- `nyc_taxi_data` — `yellow_taxi_external_table` **277,171,036 rows**,
  `green_taxi_external_table` **35,619,306 rows**, `taxi_zone_external_table`
  **265 rows**. All three are `EXTERNAL`. The zone table **does** exist as of M2; the
  earlier note here said it was never created, and that is now wrong.
- `nyc_climate_data` — `climate_external_table` (EXTERNAL), **19,260 rows**
- `dbt_prod` — prod dbt target: `fact_trips`, `dim_zones`, `dim_monthly_zones_revenue`,
  `taxi_zone_lookup`, + staging views
- `dbt_dev` — dev dbt target, empty until M3. An earlier draft called this `dbt_ssinha`.
- `dbt_ci` — CI target for `.github/workflows/dbt.yml`, empty until CI runs

Note the zone table is a **parallel path dbt does not use.** dbt reads the taxi zones from
the seed `ref('taxi_zone_lookup')`, as CLAUDE.md states.

## Test structure (moved from CLAUDE.md Testing section)

```
tests/
├── conftest.py                     # constants, auto-skip integration without creds
├── unit/
│   └── dags/
│       ├── test_dag_integrity.py   # all 9 DAG files exist + parse as valid Python
│       └── test_dag_config.py      # no stale IDs, env var usage, DAG chaining
└── integration/
    ├── test_gcs.py                 # bucket reachable, file counts per prefix
    ├── test_bigquery.py            # datasets + external tables exist + queryable
    └── test_dbt.py                 # dbt compile returns 0
```

(`tests/unit/ml/` has since been added for the fare-model modules; run `ls tests/` for
the current truth.)

## E2E smoke test (manual)

```bash
docker compose -f airflow/docker-compose.yaml up --build -d
# → open localhost:8080, trigger nyc_taxi_gcs_yellow_dag, watch task states
```

### On a GCE VM, with no keyfile (M5, 2026-09-26)

The VM runs the attached SA `dtc-de-course@…` with the `cloud-platform` scope. Create it
with `--max-run-duration` and `--instance-termination-action`, so its end never depends on
an interactive step. Ship the code with `git archive` over IAP `scp` while the branch is
unpushed, then on the VM:

```bash
echo "AIRFLOW_UID=$(id -u)" > .env
sudo docker compose -f docker-compose.yaml -f docker-compose.vm.yaml up -d --build
```

Wait for `ssh.service` on the serial console before the first IAP `scp`; an early attempt
fails with "Connection closed". The measured run is in the migration plan, "The M5 result".

### Post-run checks — nothing billing

```bash
gcloud compute instances list        # every VM TERMINATED, or the list empty
gcloud dataproc batches list --region=us-central1 --filter="state=RUNNING OR state=PENDING"
```

Both ran clean after M5 on 2026-09-26: the one VM `TERMINATED`, zero batches.

## Development plan history (Phases 0–4, full text)

Compressed in CLAUDE.md 2026-08-22; original detail:

- **Phase 0 — Checkpoint.** Branch off `main`; committed current state. `.gitignore`
  covers `secrets/`, `.venv/`, `google-cloud-sdk/`, local data dumps.
- **Phase 1 — Reconcile config with live infra.** All stale project/bucket IDs fixed.
  Green ingest DAG created; climate path bug fixed; zone ingest DAG created; all 4
  external-table DAGs migrated to env vars; ingest DAGs chained to external-table DAGs
  via `TriggerDagRunOperator`; dbt submodule + dbt Core Airflow DAG + CI/CD wired;
  PySpark pointed at `dbt_prod`. Verified: `pytest tests/` green; `dbt compile` clean;
  `docker compose up --build` succeeds; all DAGs load in Airflow UI without import errors.
- **Phase 2 — Prune redundancy.** Removed truly-unreferenced learning-exercise artifacts
  (legacy BQ homework, experimental notebooks, `docker_nana_tutorial`).
- **Phase 3 — Restructure (mechanical moves only).** `05_batch_processing/` → `spark/`;
  import paths, Docker volume mounts, dbt project paths updated. No behavioral changes.
- **Phase 4 — Refactor & harden.** DRY'd config to `GCP_PROJECT_ID` / `GCP_GCS_BUCKET`;
  stable credentials path `secrets/gcp-credentials.json` + guard test; parametrized the
  ingest window via `INGEST_START_DATE` / `INGEST_END_DATE`.

## Archive fingerprint — per-file bytes, measured 2026-09-03

Recorded by M2 after the eight ingest and external-table DAGs ran from the laptop. Plan
§6.1 depends on this: it is the record of exactly which TLC files this pipeline consumed,
so a later schema change or a re-published file at the TLC end is detectable by size.

| Month | yellow bytes | green bytes |
| --- | ---: | ---: |
| 2015-01 | 175,325,767 | 22,732,044 |
| 2015-02 | 171,639,741 | 23,596,236 |
| 2015-03 | 185,355,459 | 25,895,603 |
| 2015-04 | 180,491,380 | 25,043,726 |
| 2015-05 | 182,487,456 | 26,706,231 |
| 2015-06 | 171,760,764 | 24,706,467 |
| 2015-07 | 160,689,531 | 23,394,300 |
| 2015-08 | 154,240,802 | 23,292,101 |
| 2015-09 | 156,401,569 | 22,720,762 |
| 2015-10 | 171,091,467 | 24,605,888 |
| 2015-11 | 157,669,006 | 23,209,037 |
| 2015-12 | 159,905,170 | 24,334,716 |
| 2016-01 | 151,251,087 | 22,088,171 |
| 2016-02 | 158,113,739 | 22,771,978 |
| 2016-03 | 170,019,864 | 23,783,208 |
| 2016-04 | 165,552,992 | 23,357,449 |
| 2016-05 | 165,807,271 | 23,493,391 |
| 2016-06 | 156,288,749 | 21,485,727 |
| 2016-07 | 144,621,113 | 20,529,044 |
| 2016-08 | 139,875,738 | 19,290,983 |
| 2016-09 | 141,848,574 | 18,039,876 |
| 2016-10 | 152,579,388 | 19,277,013 |
| 2016-11 | 141,148,933 | 17,783,235 |
| 2016-12 | 145,957,196 | 18,755,368 |
| **total** | **3,860,122,756** | **540,892,554** |

Two single files complete the set:

- `taxi_zone_lookup.csv` — 12,331 bytes, 265 rows
- `climate_data.parquet` — 1,037,285 bytes, 19,260 rows

Row counts read from the external tables, so they count what BigQuery actually sees:
yellow **277,171,036**, green **35,619,306**. The four ingest prefixes hold **50 objects**
and **4,402,064,926 bytes** (4.10 GiB) in total.

