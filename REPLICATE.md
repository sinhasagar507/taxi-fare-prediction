# Replicate the Project End to End

**Invoke when:** someone wants to rebuild the whole pipeline and the fare model in their
own GCP project, or asks "how do I run this from scratch?".

This guide rebuilds the NYC taxi fare project from nothing. The pipeline has five stages:

- The ingest DAGs put the raw TLC trip data, the climate data and the zone lookup in GCS.
- The external-table DAGs make that data visible to BigQuery.
- dbt builds the marts in `dbt_prod`, with `fact_trips` at the centre.
- Spark prepares two samples, and LightGBM learns to predict `fare_capped`.
- Looker Studio shows the dashboard.

Every stage runs against real cloud resources, so you need your own GCP project. Each stage
ends with a check: a number you must see before you go to the next stage. The numbers are the
owner's measured results for the 2015–2016 window.

---

## Stage 0 — Tools and cost controls

1. Install these tools:
   - Git
   - Docker, with Compose 2.24.4 or later
   - Terraform
   - The `gcloud` CLI (<https://cloud.google.com/sdk/docs/install>)
   - Python 3.12, only if you want the host fast path for tests and dbt
2. Get the code. The dbt project is a git submodule, so clone with it:
   ```bash
   git clone --recurse-submodules https://github.com/sinhasagar507/taxi-fare-prediction.git
   cd taxi-fare-prediction
   ```
3. Create a GCP project and link a billing account to it. The exact commands are in
   `notes/gcp-setup-runbook.md`.
4. Set budget alerts **before** you create anything else. The owner used three alerts, at
   $50, $150 and $250:
   ```bash
   gcloud services enable billingbudgets.googleapis.com --project=YOUR_PROJECT
   gcloud billing budgets create --billing-account=YOUR_BILLING_ACCOUNT \
     --display-name="nyc-taxi-guard-50usd" --budget-amount=50USD
   ```
   Budgets apply to the whole billing account, not to one project.

## Stage 1 — Identity and configuration

1. Enable the APIs:
   ```bash
   gcloud services enable bigquery.googleapis.com bigquerystorage.googleapis.com \
     storage.googleapis.com dataproc.googleapis.com compute.googleapis.com \
     --project=YOUR_PROJECT
   ```
2. Create one pipeline service account. Give it `roles/bigquery.admin` and
   `roles/storage.admin`. `notes/gcp-setup-runbook.md` has the commands.
3. Download its key to exactly this path: `secrets/gcp-credentials.json`. Git ignores
   `secrets/`. Every consumer (Airflow, dbt, pytest, Terraform) reads this one path. The key
   must never enter git or a Docker image.
4. Replace the owner's project ID and bucket name with your own. The owner's values are
   hardcoded as **defaults** in these places, and a missed one sends work to the wrong
   project:

   | File | What to change |
   | --- | --- |
   | `airflow/docker-compose.yaml` | `GCP_PROJECT_ID`, `GCP_GCS_BUCKET` |
   | `terraform/variables.tf` | `project`, `gcs_bucket_name` |
   | `dbt/profiles.yml` | the fallback in `env_var('GCP_PROJECT_ID', ...)` |
   | the `gcloud dataproc` commands in this guide | project, bucket, service account |

   A bucket name is global, so pick one nobody else has.
5. Export the two variables in your shell, for dbt and the tests:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=secrets/gcp-credentials.json
   export GCP_PROJECT_ID=YOUR_PROJECT
   ```

## Stage 2 — Infrastructure with Terraform

```bash
cd terraform
terraform init
terraform apply
cd ..
```

Terraform creates one bucket and five BigQuery datasets: `nyc_taxi_data`,
`nyc_climate_data`, `dbt_prod`, `dbt_dev` and `dbt_ci`, all in location `US`.

A fresh project has no existing resources, so a plain `apply` is correct. The
"`terraform import` first" rule in `CLAUDE.md` applies only to the owner's live project,
where the resources already exist.

**Check:** `gcloud storage ls` shows your bucket, and `bq ls` shows the five datasets.

## Stage 3 — Ingest the data with Airflow

1. Start the Airflow stack:
   ```bash
   cd airflow
   docker compose up --build -d
   ```
   The UI runs at <http://localhost:8080>. The user is `airflow` and the password is `airflow`.
2. All DAGs start paused. Unpause and trigger the four ingest DAGs:

   | Ingest DAG | Downloads | Then triggers |
   | --- | --- | --- |
   | `nyc_taxi_data_ingestion_dag` | yellow TLC parquet, 2015–2016 | `create_external_table_yellow_taxi` |
   | `nyc_green_taxi_data_ingestion_dag` | green TLC parquet, 2015–2016 | `create_external_table_green_taxi` |
   | `nyc_climate_data_ingestion_dag` | the daily weather CSV, as parquet | `create_external_table_climate_data` |
   | `nyc_taxi_zone_ingestion_dag` | the zone lookup CSV | `create_external_table_taxi_zone` |

   Unpause the four external-table DAGs too, so the triggers can run them. The window is
   set by `INGEST_START_DATE` and `INGEST_END_DATE` in `airflow/docker-compose.yaml`.

**Check:** the bucket holds 50 objects, 4.10 GiB, and the external tables hold
312,790,342 trip rows across yellow and green.

## Stage 4 — Build the marts with dbt

1. Unpause and trigger the `dbt_build_marts` DAG. It runs `dbt deps` and
   `dbt build --target prod` inside the Airflow image.

   As an alternative, run dbt from the dev container or the host:
   ```bash
   dbt deps --project-dir dbt/ny_taxi_analytics
   dbt build --project-dir dbt/ny_taxi_analytics --profiles-dir dbt --target prod
   ```
2. Stop the Airflow stack when you are done with it: `docker compose down` in `airflow/`.

**Check:** `dbt build` reports `ERROR=0`, and `dbt_prod.fact_trips` holds
**307,339,039** rows. The owner's build cost $1.56.

If you see about 128 million rows, your submodule is on an old commit. That commit carried a
staging dedup bug that discarded 63% of yellow trips (D-012). Run
`git submodule update --init` and build again.

**CI (optional).** Add the service-account JSON as the `GCP_SA_KEY` repository secret on
GitHub. `.github/workflows/dbt.yml` then runs `dbt compile` on every pull request and
`dbt build` into `dbt_ci` on every push to `main`. CI never writes to `dbt_prod`.

## Stage 5 — Prepare the model data with Spark

The prep reads `dbt_prod.fact_trips` through the BigQuery connector on Dataproc Serverless.

1. Create a service account for the Dataproc batches, for example `dataproc-batch`. It
   must read BigQuery and read and write the bucket. The owner did not record its exact role
   list. `roles/dataproc.worker`, `roles/bigquery.dataViewer`,
   `roles/bigquery.readSessionUser` and `roles/storage.objectAdmin` cover those needs.
2. Package the code. The repository has no script for this step:
   ```bash
   mkdir -p /tmp/prep_deps/spark/ml/src
   cp spark/ml/src/*.py /tmp/prep_deps/spark/ml/src/
   touch /tmp/prep_deps/spark/__init__.py /tmp/prep_deps/spark/ml/__init__.py \
     /tmp/prep_deps/spark/ml/src/__init__.py
   (cd /tmp/prep_deps && zip -r ../prep_deps.zip spark)
   gcloud storage cp /tmp/prep_deps.zip spark/ml/00_prep_spark.py \
     gs://YOUR_BUCKET/dependencies/prep/
   ```
3. Submit the prep:
   ```bash
   gcloud dataproc batches submit pyspark \
     gs://YOUR_BUCKET/dependencies/prep/00_prep_spark.py \
     --region=us-central1 --version=3.0 \
     --service-account=dataproc-batch@YOUR_PROJECT.iam.gserviceaccount.com \
     --py-files=gs://YOUR_BUCKET/dependencies/prep/prep_deps.zip \
     --properties=spark.jars.packages=com.google.cloud.spark:spark-4.0-bigquery:0.45.0,spark.dynamicAllocation.maxExecutors=4 \
     --ttl=90m -- --cluster --source YOUR_PROJECT.dbt_prod.fact_trips \
     --output gs://YOUR_BUCKET/ml/prep
   ```
   `maxExecutors=4` and `--ttl=90m` are cost caps. The full record of this run is in
   `notes/2026-09-06-prep-cloud-baseline.md`.
4. Download the work sample and its stats:
   ```bash
   gcloud storage cp gs://YOUR_BUCKET/ml/prep/sample_work.parquet \
     gs://YOUR_BUCKET/ml/prep/prep_stats.json spark/ml/data/
   ```

**Check:** the driver log ends with `PREP OK`. The job reads 307,339,039 rows and keeps
304,766,876 after the guards. It writes `sample_full` (30,482,494 rows) and `sample_work`
(1,828,181 rows). The owner's run took 25.6 minutes and cost $0.53.

## Stage 6 — Train and score the model

All commands in this stage run in the dev container, on your machine, at no cloud cost. The
container holds XGBoost, LightGBM and CatBoost. The host `.venv` does not.

```bash
DEV="docker compose -f docker/dev/docker-compose.yml run --rm dev"

# 1. Compare 14 models on identical folds; save the train split for Spark
$DEV python spark/ml/01_run_sweep.py --write-train

# 2. The Spark MLlib comparison, on the same train rows
$DEV python spark/ml/01_mllib_baseline.py

# 3. Tune the two leading boosters, then diagnose the champion
$DEV python spark/ml/02_tune.py --phase tune
$DEV python spark/ml/02_tune.py --phase diagnose

# 4. The one score on the sealed sets -- run it ONE time only
$DEV python spark/ml/04_holdout_score.py
```

The sweep seals two sets before any model sees the data: a temporal set (every trip from
2016-11-01 on) and a random 20% holdout. Step 4 scores both sets one time.
`04_holdout_score.py` refuses to run a second time, so the score cannot be tuned against.

**Check:**

| Result | Owner's value |
| --- | --- |
| Train / holdout / temporal rows | 1,355,641 / 338,911 / 133,629 |
| Sweep winner after tuning | LightGBM |
| Holdout MAE | $0.311 |
| Temporal-set MAE | $0.316 |

**Optional — the full-scale run.** `03_scale_champion.py` refits the tuned LightGBM on the
`sample_full` train split (22,603,549 rows). It needs about 64 GiB of memory. The owner used
an `e2-highmem-8` VM, and the run cost $2.97 with the matching MLlib arm. Create the VM with
`--max-run-duration` and `--instance-termination-action=STOP`, so it stops without a
manual step.

## Stage 7 — Dashboard, tests and cleanup

1. Connect Looker Studio to `dbt_prod.fact_trips`. Follow
   `notes/2026-09-04-looker-studio-reconnect-runbook.md`. The calculated fields and the
   colour set are in the README, under "Looker Studio Dashboard".
2. Run the test suite. Always give the `tests/` path:
   ```bash
   docker compose -f docker/dev/docker-compose.yml run --rm dev pytest tests/
   ```
   The integration tests check your bucket, datasets and dbt build. They run once
   `GOOGLE_APPLICATION_CREDENTIALS` resolves, and skip otherwise.

   On the host fast path (`.venv/bin/pytest tests/`), point the Spark worker at the venv
   Python. Otherwise Spark starts the system `python3`, which on macOS is 3.9, and every
   Spark test fails with `TypeError: unsupported operand type(s) for |: 'type' and 'type'`:
   ```bash
   PYSPARK_PYTHON=$PWD/.venv/bin/python .venv/bin/pytest tests/
   ```
3. Make sure that nothing bills while idle:
   ```bash
   gcloud compute instances list          # only TERMINATED VMs
   gcloud dataproc batches list --region=us-central1   # nothing RUNNING or PENDING
   ```

**Optional — Airflow on a VM with no keyfile.** Clone the repository on a GCE VM, attach
the pipeline service account to the VM, and start the stack with the override file:
```bash
cd airflow
docker compose -f docker-compose.yaml -f docker-compose.vm.yaml up -d --build
```
The override removes the keyfile and uses the VM's attached identity. Build the images on
the VM itself. An image built on an Apple-Silicon Mac does not match the VM's CPU.

---

## Cost

The owner's measured costs for the stages above:

| Stage | Cost |
| --- | ---: |
| dbt build of `dbt_prod` | $1.56 |
| Spark prep on Dataproc | $0.53 |
| Full-scale run on `e2-highmem-8` (optional) | $2.97 |
| Airflow on a VM, one DAG pair (optional) | about $0.02 |

BigQuery queries are the main cost. The owner's first 30 days cost about $20.7, and 99% of
that was BigQuery queries, most of them checks rather than builds. A clean run with fewer
checks costs less. `notes/2026-09-04-cloud-cost-baseline.md` has the method.

## Known gaps

- **The model uses the recorded trip duration.** `trip_duration_min` is known only after the
  trip ends. Without it, the MAE rises by $0.91, to about $1.23. State this next to the $0.31
  result. `notes/2026-09-26-project-story.md`, section 7, has the detail.
- **The repository has no script that builds `prep_deps.zip`.** Stage 5 gives the commands.
- **The project ID is a default in several files.** Stage 1 lists them.
