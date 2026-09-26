# NYC Taxi Fare Prediction

End-to-end data engineering pipeline. It ingests NYC Taxi and Limousine Commission (TLC) trip records and daily climate data, transforms them through a dbt ELT layer, and surfaces analytical insights via a Looker Studio dashboard and an ML model that predicts the trip fare (`fare_capped`). The model lives in `spark/ml/` (scikit-learn sweep + Spark MLlib baseline); its design and phase log live in `spark/2026-07-10-fare-prediction-modeling-plan.md`.

> The project began as a trip-*duration* predictor and was reoriented to fare prediction;
> the repository folder name (`nyc_taxi_durationprediction`) predates the pivot.

---

## Pipeline Architecture

```text
TLC source data (2015-2016)
        |
        v
  GCS raw storage          <-- Airflow ingest DAGs
        |
        v
  BigQuery external tables <-- Airflow external-table DAGs
        |
        v
  dbt ELT (staging + marts)<-- Airflow dbt DAG / CI via GitHub Actions
        |
        v
  fact_trips + dim tables  <-- dbt_prod dataset
        |
        +---> Looker Studio dashboard
        +---> Fare prediction model (spark/ml/)
```

---

## GCP Infrastructure — reference layout

> **No GCP project is provisioned for this repository, and the pipeline does not assume
> one.** Treat nothing below as live, and do not expect credentials to be present. The
> dbt-built marts are held locally, outside the working tree, which is what lets the
> fare model in `spark/ml/` run end to end with no cloud at all.
>
> To point the pipeline at a project of your own: drop a service-account key at the
> stable path `secrets/gcp-credentials.json`, export `GOOGLE_APPLICATION_CREDENTIALS`,
> and set `GCP_PROJECT_ID` / `GCP_GCS_BUCKET`. Nothing else needs editing — those two
> env vars are the single source of truth. The layout below is the structure the DAGs
> and dbt models expect.

| Resource | Value |
| --- | --- |
| GCS bucket | `$GCP_GCS_BUCKET` (US) |
| BigQuery datasets | `nyc_taxi_data`, `nyc_climate_data`, `dbt_prod`, `dbt_ssinha` |
| dbt prod target | `dbt_prod` |
| dbt dev target | `dbt_ssinha` |

**GCS data layout:**

```text
primary-data-dtc/
├── nyc_taxi_data/
│   ├── yellow_taxi_data/   # 24 parquet files, 2015-01 to 2016-12
│   └── green_taxi_data/    # 24 parquet files, 2015-01 to 2016-12
├── nyc_taxi_data/taxi_lookup_data/taxi_zone_lookup.csv
└── nyc_climate_data/climate_data.parquet
```

---

## Repository Structure

```text
nyc_taxi_durationprediction/
├── airflow/                        # Orchestration
│   ├── dags/                       # 9 DAG files (ingest, external tables, dbt)
│   ├── docker-compose.yaml
│   └── dockerfile
├── dbt/
│   ├── ny_taxi_analytics/          # dbt project (git submodule)
│   ├── profiles.yml                # BigQuery profile (env-var keyfile)
│   └── requirements.txt            # dbt-bigquery==1.11.1
├── docker/dev/                     # Whole-repo dev container (ML, pytest, dbt, Jupyter)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements-dev.txt
├── spark/                          # Batch processing + ML
│   ├── ml/                         # Fare model: prep, features, sweep, MLlib baseline
│   ├── 2026-07-10-fare-prediction-modeling-plan.md
│   └── pyspark_bigquery_hybrid.py  # (+ legacy course notebooks)
├── terraform/                      # Infrastructure as code (reference only)
├── bigquery/                       # SQL reference queries
├── tests/                          # Project-wide test suite
│   ├── conftest.py
│   ├── unit/dags/
│   └── integration/
├── .github/workflows/dbt.yml       # CI/CD: dbt compile on PR, dbt build on main
├── notes/                          # Course notes and documentation
├── secrets/                        # Service-account keys (gitignored)
└── CLAUDE.md                       # AI agent guidance and pipeline reference
```

---

## What Has Been Implemented

### Phase 0 — Repository baseline

- Hardened `.gitignore` covering `secrets/`, `.venv/`, vendored SDKs, and local data dumps.
- Git history cleaned: removed accidentally committed GCP credentials and vendored `google-cloud-sdk/` (repository shrank from 11 GB to 82 MB via `git filter-repo`).

### Phase 1 — Pipeline reconciliation with infrastructure

All configuration reconciled against the infrastructure the pipeline ran on. Stale
project and bucket IDs were fixed, and the DAGs switched to reading config from
environment variables.

**Airflow DAGs (9 total):**

| DAG | Purpose |
| --- | --- |
| `nyc_taxi_gcs_yellow_dag` | Download yellow TLC parquets (2015-16) to GCS |
| `nyc_taxi_gcs_green_dag` | Download green TLC parquets (2015-16) to GCS |
| `nyc_climate_gcs_dag` | Download NOAA climate CSV, convert to parquet, upload to GCS |
| `nyc_taxi_zone_gcs_dag` | Download TLC zone lookup CSV to GCS |
| `yellow_taxi_external_table` | Create BigQuery external table over yellow parquets |
| `green_taxi_external_table` | Create BigQuery external table over green parquets |
| `climate_data_external_table` | Create BigQuery external table over climate parquet |
| `nyc_taxi_zone_external_table` | Create BigQuery external table over zone CSV |
| `dbt_run_dag` | Run `dbt build` inside an isolated venv in the Airflow container |

Each ingest DAG chains to its corresponding external-table DAG via `TriggerDagRunOperator`. All DAGs read GCP configuration from environment variables (`GCP_PROJECT_ID`, `GCP_GCS_BUCKET`) — no hardcoded project or bucket IDs.

**dbt models (`dbt_prod` dataset):**

| Model | Type | Description |
| --- | --- | --- |
| `stg_yellow_taxi_data` | View | Staged yellow trips, typed and renamed |
| `stg_green_taxi_data` | View | Staged green trips, typed and renamed |
| `stg_climate_data` | View | Staged daily climate records |
| `taxi_zone_lookup` | Seed | Zone ID to borough/zone name mapping |
| `fact_trips` | Table | Union of yellow + green, joined to zones and climate; filters Unknown borough |
| `dim_zones` | Table | Distinct zone dimension |
| `dim_monthly_zones_revenue` | Table | Monthly revenue aggregated by zone |

**CI/CD (`.github/workflows/dbt.yml`):**
- `dbt compile` runs on every pull request (validates all model SQL).
- `dbt build` runs on merge to `main` (materialises all models to `dbt_prod`).

### Phase 2 — Redundancy pruning

Removed learning-exercise artifacts that were not part of the live pipeline: legacy BigQuery homework directories, experimental notebooks, and the vendored `docker_nana_tutorial`.

### Phase 3 — Directory restructure

Renamed `05_batch_processing/` to `spark/` to match the target directory layout documented in `CLAUDE.md`.

### Phase 4 — Refactor and hardening

- **Config decoupled from credentials.** The service-account key is referenced only via a
  stable, project-agnostic path (`secrets/gcp-credentials.json`) and the
  `GOOGLE_APPLICATION_CREDENTIALS` env var. Swapping GCP projects is a one-file drop plus a
  single `.env` edit — no repo-wide hunt. A guard test (`test_credential_decoupling.py`)
  scans tracked code/config and fails if any project-specific key filename reappears.
- **Ingest date range parametrized.** The backfill window is driven by `INGEST_START_DATE`
  / `INGEST_END_DATE` env vars (default `2015-01-01` … `2016-12-31`) rather than hardcoded
  dates, so extending coverage needs no DAG edits.
- **Single source of truth** for project/bucket remains the Airflow `docker-compose.yaml`
  env vars (`GCP_PROJECT_ID`, `GCP_GCS_BUCKET`), read by DAGs and tests via `os.environ`.

### Phase 5 — Documentation

This README and `CLAUDE.md` reconciled to the final structure, and to running without a
provisioned GCP project (no live resources or credentials assumed).

### Test suite (165 tests on the host, 166 in the dev container)

```text
tests/
├── unit/
│   ├── dags/
│   │   ├── test_dag_integrity.py       # all 9 DAG files exist and parse as valid Python
│   │   └── test_dag_config.py          # no stale IDs, env var usage, DAG chaining
│   ├── ml/                             # fare-model prep/feature/eval unit tests
│   │   ├── test_features.py
│   │   ├── test_preprocess.py
│   │   ├── test_evaluate.py
│   │   └── test_prep_cli.py            # guard: a prep run never starts without a named --source
│   ├── test_credential_decoupling.py   # guard: old keyfile name never re-hardcoded
│   └── test_docker_runtime.py          # guard: build context, dbt pins, Jupyter exposure
└── integration/
    ├── test_gcs.py                     # bucket reachable, file counts per prefix
    ├── test_bigquery.py                # datasets, external tables, dbt_prod tables
    └── test_dbt.py                     # dbt compile returns exit code 0
```

The container runs one extra test — a dbt-isolation check that skips outside the image.

Unit tests require no credentials. Integration tests auto-skip if `GOOGLE_APPLICATION_CREDENTIALS` is not set.

---

## Running the Pipeline

### Prerequisites

- Docker and Docker Compose — required for the ML/modeling work and for Airflow
- Python 3.12+ with a virtual environment — optional convenience path for tests and dbt
- A GCP service-account key at the stable path `secrets/gcp-credentials.json` (gitignored;
  supply your own if reconnecting to a live project)

### Dev container

The whole-repo development runtime. It carries the ML stack (scikit-learn, XGBoost,
LightGBM, CatBoost, SHAP, Optuna), PySpark, the pytest suite, the dbt CLI, and JupyterLab.
These libraries are installed **in the container only**, never in the host venv.

```bash
# Build (run from the repo root)
docker compose -f docker/dev/docker-compose.yml build

# Run the test suite
docker compose -f docker/dev/docker-compose.yml run --rm dev pytest tests/

# One-off script, interactive shell, dbt
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python spark/ml/00_prep_spark.py --source local --limit-files 3
docker compose -f docker/dev/docker-compose.yml run --rm dev bash
docker compose -f docker/dev/docker-compose.yml run --rm dev dbt --version

# JupyterLab -> http://127.0.0.1:8888/lab?token=nyc-taxi-dev
docker compose -f docker/dev/docker-compose.yml up
```

The repo is bind-mounted at `/workspace`, so host edits are live inside the container and
outputs written by a container run appear in your working tree. Jupyter is published on
loopback only; override the token with `JUPYTER_TOKEN=... docker compose ... up`.

Airflow runs its own separate stack — the two are never merged.

### Start Airflow

```bash
docker compose -f airflow/docker-compose.yaml up --build -d
```

Open `http://localhost:8080`. Trigger DAGs in this order:

1. `nyc_taxi_gcs_yellow_dag` (triggers `yellow_taxi_external_table` automatically)
2. `nyc_taxi_gcs_green_dag` (triggers `green_taxi_external_table` automatically)
3. `nyc_climate_gcs_dag` (triggers `climate_data_external_table` automatically)
4. `nyc_taxi_zone_gcs_dag` (triggers `nyc_taxi_zone_external_table` automatically)
5. `dbt_run_dag` (run after all external tables are created)

### Run dbt locally

```bash
GOOGLE_APPLICATION_CREDENTIALS=secrets/gcp-credentials.json \
  dbt build --project-dir dbt/ny_taxi_analytics --profiles-dir dbt
```

Note: use `.venv/bin/dbt` explicitly if the Homebrew `dbt` on your PATH is the dbt Cloud CLI rather than dbt Core.

### Run the test suite

Always pass the explicit `tests/` path — a bare `pytest` also picks up the legacy
`airflow/tests/` stubs and fails at collection.

```bash
# Unit tests only (no credentials required)
.venv/bin/pytest tests/unit/ -v

# Integration tests (reads credentials automatically from secrets/)
.venv/bin/pytest tests/integration/ -v

# Full suite
.venv/bin/pytest tests/ -v

# Same suite inside the dev container
docker compose -f docker/dev/docker-compose.yml run --rm dev pytest tests/
```

`ModuleNotFoundError: No module named 'xgboost'` on the host is expected — the ML
libraries are container-only, and the tests that need them skip cleanly outside the
container.

---

## Looker Studio Dashboard

A 6-page analytical dashboard is under active development against `$GCP_PROJECT_ID.dbt_prod`. The dashboard connects to BigQuery via the native Looker Studio connector.

### Validated baselines

| Metric | Green | Yellow |
| --- | --- | --- |
| Total trips | 28,424,373 | 100,357,273 |
| Total revenue | ~$416M | ~$1.62B |
| Average fare | $12.15 | $12.97 |
| Climate join match | 100% | 100% |
| Zone join drop rate | 0% | 0% |
| Months covered | 24 (2015-01 to 2016-12) | 24 (2015-01 to 2016-12) |

Approximately 2.1 million trips (1.6% of staged records) are excluded by the `WHERE borough != 'Unknown'` filter in `fact_trips`. This is by design.

### Outlier caps applied in calculated fields (p99 thresholds)

| Service | Fare cap | Distance cap |
| --- | --- | --- |
| Green | $46 | 14.19 mi |
| Yellow | $52 | 18.76 mi |

### Calculated fields (defined in the Looker Studio data source)

| Field | Formula |
| --- | --- |
| `fare_capped` | `IF((service_type="Yellow" AND fare_amount>52) OR (service_type="Green" AND fare_amount>46) OR fare_amount<=0, NULL, fare_amount)` |
| `distance_capped` | `IF((service_type="Yellow" AND trip_distance>18.76) OR (service_type="Green" AND trip_distance>14.19) OR trip_distance<=0, NULL, trip_distance)` |
| `revenue_per_mile` | `IF(distance_capped IS NULL OR distance_capped=0, NULL, fare_capped/distance_capped)` |
| `trip_duration_min` | `DATETIME_DIFF(dropoff_datetime, pickup_datetime, MINUTE)` |
| `weather_condition` | `IF(precipIntensity > 0, "Rainy", "Dry")` |
| `temp_band` | CASE on `highTemp`: Freezing / Cold / Mild / Warm / Hot |
| `pickup_hour` | `HOUR(pickup_datetime)` |
| `pickup_weekday_ordered` | CASE on `WEEKDAY(pickup_datetime)` producing sortable labels (1. Mon ... 7. Sun) |

Report-level filters applied to the data source: `trip_distance > 0`, `fare_amount >= 0`.

### Dashboard pages

| Page | Status | Description |
| --- | --- | --- |
| 1 — Revenue overview | Complete | KPIs: total revenue, avg fare, revenue per mile, total distance. Monthly revenue trend, fare by service, revenue by borough, avg fare by hour. |
| 2 — Demand heatmap | In progress | Hourly demand heatmap (weekday x hour), top 10 pickup zones, demand by day of week, daily trip volume trend (Yellow vs Green). |
| 3 — Weather impact analysis | Planned | Demand multiplier by weather condition, fare uplift, speed degradation, temperature vs fare scatter by borough. |
| 4 — Route and corridor deep-dive | Planned | Airport revenue share, top origin-destination corridors, revenue by rate code, borough-to-borough flow matrix. |
| 5 — Customer behavior | Planned | Tip conversion rate, tip rate by hour and borough, tip distribution by service. Card trips only. |
| 6 — Fare prediction feature explorer | Planned | Predictive feature correlations, trip distance and duration distributions, fare component breakdown, distance vs fare scatter. |

### Design system

All pages follow a dark theme defined in the `looker_climate_template.html` reference mockup.

| Element | Hex |
| --- | --- |
| Page background | `#080C18` |
| Card background | `#0F1525` |
| Yellow service | `#F7C426` |
| Green service | `#2ECFB1` |
| Decline / negative | `#FF6B5B` |
| Neutral series A | `#4C8EFF` |
| Neutral series B | `#9B7FFF` |
| Labels / captions | `#8892AB` |

Fonts: Inter (body), DM Mono (KPIs and data values), Syne (section headers).

---

## Future Scope

**Borough route data integration:**
The pipeline architecture supports adding route-distance features derived from the OSRM routing API. The planned approach is to precompute a 263x263 zone-centroid distance and duration matrix (one API call to the OSRM `/table` endpoint), store it as a BigQuery lookup table, and join it to `fact_trips` in PySpark to add `osrm_distance_mi` and `osrm_duration_min` as ML features.

Reference data sources:
- NYC Taxi Zones GeoParquet: https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/8meu-9t5y
- OSRM API documentation: https://project-osrm.org/docs/v5.5.1/api/
- NYC LION street network: https://www.nyc.gov/content/planning/pages/resources/datasets/lion

**Extended date ranges:**
The pipeline accepts additional TLC parquet files without schema changes. As of Phase 4 the
backfill window is parametrized via the `INGEST_START_DATE` / `INGEST_END_DATE` env vars, so
adding 2017+ data is a config change (no DAG edits); BigQuery external tables and dbt models
pick up the new files automatically.

---

## Known Constraints

- `terraform/` contains infrastructure definitions but TF state is empty. Run `terraform import` before any `terraform apply` to avoid conflicting with existing GCP resources.
- The Homebrew `dbt` binary on macOS may resolve to the dbt Cloud CLI. Always use `.venv/bin/dbt` for dbt Core commands.
- The system `GOOGLE_APPLICATION_CREDENTIALS` environment variable may point to a stale path. `tests/conftest.py` overrides it automatically with `secrets/gcp-credentials.json` when that file is present.
- The Airflow Docker image targets Linux (arm64). Do not install macOS-specific binaries into it.

---

## Credentials

Service account keys in `secrets/` are gitignored. Do not commit them. The CI pipeline reads credentials from the `GCP_SA_KEY` GitHub Actions repository secret.
