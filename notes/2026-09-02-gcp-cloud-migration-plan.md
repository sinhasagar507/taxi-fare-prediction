# GCP Cloud Migration Plan

**Invoke when:** any cloud, Terraform, Dataproc, BigQuery or cost question, and before spending anything.

**Created:** 2026-09-02
**Author:** Sagar Sinha (with Claude Code)
**Answers:** audit item 5 (`notes/2026-08-22-repo-audit.md`) — "record a decision: provision a
project, build a local substitute, or archive-as-reference." Also carries items 6 and 8 and
resolves D-006.
**Companion docs:** `spark/2026-07-10-fare-prediction-modeling-plan.md` (§5b, §5c, §10),
`notes/gcp-setup-runbook.md`, `notes/gcp-reference.md`, `notes/decisions.md`.

**Decision, stated first: provision.** Stand the pipeline up on Google Cloud under
`$GCP_PROJECT_ID` (default `dtc-de-project-506916`), on the free-trial credit, with every
billable resource either serverless or deleted when idle. Not a local substitute (the
pipeline's GCS → BigQuery external-table → dbt path is the portfolio artefact, and a
substitute proves nothing about it) and not archive-as-reference (the §5c full-scale run has
no other target).

---

## 0. Framing

**What this plan is for.** Two things that both need a live project: proving the Airflow /
dbt / Terraform half of the pipeline, which has never run against real infrastructure, and
running modeling-plan §5c (full-scale training on the 12,748,027-row `sample_full`). It also
answers the owner's requirement that modeling move to PySpark — §5 argues that honestly.

**What it assumes.**

- A $300 Welcome credit, usable over 90 days, attached to the billing account for
  `$GCP_PROJECT_ID`. Verified 2026-09-02 against `docs.cloud.google.com/free/docs/free-cloud-features`.
  Eligibility is the owner's to confirm. If no credit applies, every figure in §3 becomes
  real dollars — and at this project's scale the figures are the same size.
- The 7.1 GB local `fact_trips` backup (128,781,646 rows, 204 parquet files, resolved by
  `spark/ml/src/paths.py`) stays the modeling input until BigQuery reproduces it.
- The target is `fare_capped` (D-001, LOCKED). The folder name says "duration"; it is wrong
  and stays.
- The sealed 153,153-row holdout is scored once, at the end of modeling Phase 5. Nothing in
  this plan touches it.

**What it excludes.** D-002 / D-003 / D-004 (DEFERRED — not raised here). `CASE_STUDY.md`
(D-005 — waits on Phase 5). Modeling Phase 6 (neural nets). Dashboard v3 beyond
reconnecting Looker Studio to the rebuilt `dbt_prod`. The OSRM routing integration is
scoped in §6 but not scheduled.

**Measured vs assumed — the house rule.** This project has twice built on library
behaviour it never checked (`d83141b`, the §5b "MLlib has no TargetEncoder" claim). Every
external figure below carries one of three labels: **VERIFIED** (read from the primary
page on 2026-09-02, with the literal text captured), **CORROBORATED** (consistent across
secondary sources, primary page not captured), or **UNVERIFIED / ASSUMED**. A figure with
no label is measured in this repository.

**A note on rotting links.** Google has folded the Dataproc pricing page into "Managed
Service for Apache Spark (formerly Dataproc)", the Composer page into "Managed Service for
Apache Airflow", and `cloud.google.com/vertex-ai/pricing` now 301-redirects to
`cloud.google.com/products/gemini-enterprise-agent-platform/pricing`. `cloud.google.com/*/docs/*`
redirects to `docs.cloud.google.com`. Expect every URL in this document to move; the
product names above are the search terms that still find them.

---

## 1. Where the repository is today

### 1.1 Works today, no cloud needed

| Component | Evidence |
|---|---|
| `spark/ml/` end to end: `00_prep_spark.py` → `features.py` → `01_run_sweep.py` → `01_mllib_baseline.py` | Phase 4b complete 2026-09-01; results in `spark/ml/results/` |
| Unit suite | `.venv/bin/pytest tests/` on 2026-09-02: **285 passed, 9 failed, 1 skipped**; all 9 failures are `tests/integration/` |
| `dbt compile` against the submodule SQL | needs no warehouse |
| Dev container (`docker/dev/Dockerfile`: Python 3.12, JRE 17, `pyspark==4.1.2`) | builds and runs the suite |

### 1.2 Wired, never proven against live infrastructure

- All 9 DAGs in `airflow/dags/`: four ingest (`nyc_taxi_gcs_{yellow,green}_dag.py`,
  `nyc_climate_gcs_dag.py`, `nyc_taxi_zone_gcs_dag.py`), four `CREATE OR REPLACE EXTERNAL
  TABLE` DAGs, and `dbt_run_dag.py` (`dbt deps && dbt build --target prod` in
  `/opt/dbt-venv`). Every one reads `GCP_PROJECT_ID` / `GCP_GCS_BUCKET` from `os.environ`;
  no DAG hardcodes an ID. Ingest DAGs chain to their external-table DAG via
  `TriggerDagRunOperator`. None has ever been triggered against a bucket that exists.
- `dbt build` / `dbt run` (any target), and the CI `dbt build` in `.github/workflows/dbt.yml`
  (needs the `GCP_SA_KEY` secret, target `ci` → `dbt_ci`).
- `terraform/main.tf`: declares one resource, `google_bigquery_dataset.demo_dataset`
  (default name `nyc_tlc_trips`), provider `hashicorp/google` 5.6.0. That dataset name
  appears nowhere else in the pipeline. The bucket in `variables.tf`
  (`primary-data-dtc-506916`) is not declared at all.
- The GCP half of the README pipeline diagram.

Dead config worth knowing about, not fixing now: `nyc_climate_gcs_dag.py` reads
`BIGQUERY_DATASET` / `CLIMATE_TABLE` and never uses them (its inline external-table block
is commented out); the yellow/green DAGs define `TAXI_BIGQUERY_DATASET_ID` /
`..._TABLE_ID` and never use them.

### 1.3 Does not exist

- A provisioned project answering to `dtc-de-project-506916`. Every file that names it
  names it as a *default* (`dbt/profiles.yml` via `env_var('GCP_PROJECT_ID', ...)`,
  `terraform/variables.tf`, the Airflow compose env). Nothing confirms it is live, and the
  tracked Terraform state points elsewhere (1.4).
- The §5c full-scale run. OSRM route features (README Future Scope, zero code). Any
  Vertex AI reference.
- `taxi_zone_external_table` — never created; dbt gets zones from
  `seeds/taxi_zone_lookup.csv`, not the external table (the zone DAGs are a parallel path).

### 1.4 Three stale project IDs — the cleanup the migration must start with

CLAUDE.md says "TF state is empty". It is not empty; it is foreign.

| Where | What it holds | Project ID it names |
|---|---|---|
| `terraform/terraform.tfstate` (serial 3, lineage `c1b3ce47…`) | `google_bigquery_dataset.demo_dataset` = `projects/…/datasets/nyc_tlc_trips` | `dtc-de-course-457315` |
| `terraform/terraform.tfstate.1753224766.backup` (serial 1, lineage `16740d2c…`) | `google_storage_bucket.demo-bucket` = `dtc-de-project_1` | `dtc-de-course-466501` |
| `terraform/terraform.tfstate.backup` (serial 2, lineage `16740d2c…`) | nothing | — |
| `dbt/ny_taxi_analytics/models/staging/schema_taxi.yml:5` and `schema_climate.yml:5` | `database:` for the source definitions | `dtc-de-project-492321` |

The live tfstate and the two backups have different lineages — the state was re-initialised
at some point, so the live file is a second, disconnected history. None of the three
resources matches the current `variables.tf` bucket or the datasets the pipeline actually
uses (`nyc_taxi_data`, `nyc_climate_data`, `dbt_prod`). All three tfstate files are
tracked in git, which is itself the defect: state can hold secrets and is never
merge-safe.

The submodule IDs are the runbook's "known snag". They are fixed **upstream**
(`github.com/sinhasagar507/ny_taxi_analytics`) and the pointer bumped; CLAUDE.md forbids
editing inside `dbt/ny_taxi_analytics`. The fix is `database: "{{ env_var('GCP_PROJECT_ID') }}"`
(dbt resolves `env_var` in source YAML), which also makes the `ci` target's project follow
the same variable.

### 1.5 The 9-failure baseline

`tests/conftest.py` picks up `secrets/gcp-credentials.json` if present. On 2026-09-02 a
credential resolved and authenticated, and every integration test then failed on missing
resources (0 of 24 yellow files, 0 of 24 green, no datasets). That is the expected shape of
"no project provisioned", identical before and after the 2026-09-01 session. §4's gates
count down from 9 to 0; a change that moves the number in the other direction is a
regression.

---

## 2. Target architecture on GCP

```
 TLC CDN (d37ci6vzurychx.cloudfront.net)            NOAA/gist CSV       TLC zone CSV
        │ curl (ingest DAGs, INGEST_START_DATE..INGEST_END_DATE)   │                │
        ▼                                                          ▼                ▼
 ┌────────────────────────── GCS  gs://$GCP_GCS_BUCKET  (us-central1) ─────────────────────┐
 │ nyc_taxi_data/yellow_taxi_data/*.parquet   nyc_climate_data/*.parquet   taxi_lookup_data/ │
 │ nyc_taxi_data/green_taxi_data/*.parquet    ml/samples/*.parquet  ml/models/  tfstate?     │
 └───────────────┬───────────────────────────────────────────────┬───────────────────────────┘
                 │ CREATE OR REPLACE EXTERNAL TABLE (4 DAGs)      │ spark-4.0-bigquery 0.45.0 /
                 ▼                                                │ plain parquet read
 ┌──────────── BigQuery ─────────────┐                            ▼
 │ nyc_taxi_data.{yellow,green}_…    │   ┌──────── Dataproc Serverless, runtime 3.0 ───────┐
 │ nyc_climate_data.…                │   │ Spark 4.0.1 · Scala 2.13 · Python 3.12          │
 │        │ dbt build --target prod   │   │ 00_prep_spark.py  ·  01_mllib_baseline.py       │
 │        ▼                          │◄──│ §5c full-scale run; §5 OOF encoder              │
 │ dbt_prod.fact_trips (partition+   │   │ bills per second, scales to zero                 │
 │   cluster), dim_zones, dim_…      │   └─────────────────────────────────────────────────┘
 └───────────────┬───────────────────┘
                 │                          ┌──────── GCE VM (amd64), attached SA (ADC) ────┐
                 ▼                          │ airflow/docker-compose.yaml, images built here │
        Looker Studio (dashboard v3)        │ STOPPED whenever no DAG is scheduled to run    │
                                            └────────────────────────────────────────────────┘
 Terraform: bucket + datasets, state reset, import-before-apply.   Budget alerts: $50 / $150 / $250.
```

### 2.1 Storage — GCS bucket, one, regional `us-central1`

**Why.** The four ingest DAGs and the four external-table DAGs already target
`gs://{GCP_GCS_BUCKET}/nyc_taxi_data/…`, `nyc_climate_data/…`, `taxi_lookup_data/…`
(`notes/gcp-reference.md`). Keeping the layout keeps eight DAGs and three integration
tests unchanged. A US region is the Always Free condition for the 5 GB-month allowance
(VERIFIED). New prefixes for the ML side: `ml/samples/` (uploaded `sample_work_train.parquet`
and `sample_full.parquet`), `ml/models/`.

**Rejected: loading raw parquet straight into native BigQuery tables.** It would drop the
external-table layer the DAGs and dbt sources are built on, and the raw files must land in
GCS anyway for Spark to read them without the BigQuery connector.

**Location question — RESOLVED 2026-09-02, VERIFIED against the BigQuery docs.** A `US`
multi-region dataset **can** define and query an external table over a `us-central1` bucket,
and pays no data-transfer charge for it. `docs.cloud.google.com/bigquery/docs/locations`:
"If your BigQuery dataset is in the `US` multi-region, then the Cloud Storage bucket can be
in the Iowa (`us-central1`) single region, or any dual-region that includes Iowa." The
stricter sentence on the external-table page — "The Cloud Storage bucket must be in the same
location as the dataset that contains the table you're creating" — is the general rule; the
locations page carries the multi-region exception. Any **other** single region, `us-west1`
for example, does incur transfer charges even though it sits inside the US multi-region.

So the two Terraform variables already agree: `var.location = "US"` for the datasets,
`var.region = "us-central1"` for a new bucket. `dbt/profiles.yml` stays on `location: US`.
Nothing has to change together.

**Correction to this section.** It said "`dbt/profiles.yml` sets no `location`". That is false
as measured: all three targets (`dev`, `ci`, `prod`) hardcode `location: US`.

### 2.2 Warehouse — BigQuery, external tables + dbt marts

**Why.** Already built: three staging views and three core models in the submodule;
`dbt_prod` as the prod dataset (never `dbt_production`). BigQuery costs nothing at this
scale (§3), and Looker Studio reads it natively.

**Change to make upstream (in the same submodule PR as the ID fix):** `fact_trips.sql`
gets `{{ config(partition_by={"field": "pickup_datetime", "data_type": "timestamp",
"granularity": "month"}, cluster_by=["pickup_locationid", "dropoff_locationid"]) }}`.
Monthly, not daily: 128.8M rows over 730 days is ~176K rows a day — under BigQuery's own
several-GB-per-partition guidance, so daily partitions would be many and thin. Cluster on
the two zone IDs because `od_corridor` and the dashboard's corridor/borough-flow pages
filter on them; `service_type` has two values and is not worth a slot.

**Read 2026-09-02, as this bullet asked.** `models/core/fact_trips.sql` line 1 is
`{{ config(materialized='table') }}` and nothing else: no `partition_by`, no `cluster_by`,
so the change is additive and overwrites no existing choice. The three columns it names all
exist in the model's own `select` (`tu.pickup_datetime`, `tu.pickup_locationid`,
`tu.dropoff_locationid`), and `pickup_datetime` really is a timestamp — the staging models
build it with `cast(lpep_pickup_datetime as timestamp)`, so `"data_type": "timestamp"` is
right and not assumed. The patch is prepared and dry-run in M0.

**Rejected:** BigQuery Editions / slot reservations (on-demand with a 1 TiB free month is
the right model for a pipeline that builds marts a handful of times).

### 2.3 Orchestration — self-hosted Airflow, compose stack, one GCE VM, stopped when idle

**Why.** `airflow/docker-compose.yaml` + `airflow/dockerfile` (`apache/airflow:2.10.3`,
isolated `/opt/dbt-venv` with `dbt-bigquery==1.11.1`) is built and documented. The
"Deployment split" design (CLAUDE.md, decided 2026-07-10) already says: single VM, images
built on the VM, attached service account, no keyfiles in images or git. This plan keeps
it — which resolves **D-006: keep the subsection.**

**The operational rule that makes it affordable:** the VM is *stopped* (not merely idle)
between runs. A VM bills for every second it exists in the running state, whether or not a
DAG is executing, and the budget is a fixed credit. Persistent disk still bills while
stopped — 30 GB-months of standard PD are Always Free (VERIFIED), so a ≤30 GB boot disk
costs nothing stopped.

**Before the VM exists at all,** the DAGs are proven from the laptop: the same compose
stack runs locally with `GOOGLE_APPLICATION_CREDENTIALS` pointed at
`secrets/gcp-credentials.json` and talks to the live bucket and datasets. That is §4 Phase
M2. It costs zero compute and catches every DAG bug before any VM is paid for. **How the
compose file mounts the keyfile into the containers was not read in the research** — check
`airflow/docker-compose.yaml` for the volume and env line before M2.

**Rejected: Cloud Composer.** It does not scale to zero — Google's own scaling docs say the
scheduler, DAG processor, triggerer and web server run continuously as long as the
environment exists; only workers autoscale, and to a floor you set (VERIFIED). The rate is
$0.06 per 1,000 milliDCU-hours (VERIFIED); the minimum monthly bill for the smallest
persistent environment is **UNVERIFIED** (secondary sources put it at $300–450). Even the
low end would consume the credit in a month for a pipeline that runs a DAG a few times.
Composer 1 and 2.0.x also reach end of life on 2026-09-15 (CORROBORATED).

### 2.4 Spark — Dataproc Serverless, runtime 3.0 (Spark 4.0.1). Accept the minor-version step down.

Three options were on the table: accept Spark 4.0.1 on Dataproc Serverless; build a custom
Dataproc image with 4.1.2; run a plain GCE VM with the local Spark install. Two facts settle
it.

**Fact 1 — the 4.1.2 pin is not load-bearing.** The only Spark-4-specific class the
pipeline uses is `pyspark.ml.feature.TargetEncoder`, added in **4.0.0**. It exists in 4.0.1.
Every other class in `spark/ml/src/mllib.py` and the two scripts — `Pipeline`,
`StringIndexer`, `OneHotEncoder`, `SQLTransformer`, `VectorAssembler`, `GBTRegressor`,
`RegressionEvaluator` — predates Spark 3. `requirements.txt` pins 4.1.2 because it was
current on 2026-07-28, not because anything needs it.

**Fact 2 — the managed runtime is closer to the dev container than the host venv is.**
Runtime 3.0.13 (released 2026-03-27) is Spark 4.0.1, Scala 2.13, Python 3.12 (VERIFIED,
`docs.cloud.google.com/dataproc-serverless/docs/concepts/versions/spark-runtime-3.0`).
`docker/dev/Dockerfile` is Python 3.12 on JRE 17. The host `.venv` is Python 3.13. The
BigQuery connector `com.google.cloud.spark:spark-4.0-bigquery:0.45.0` has been GA since
0.44.0 (2026-02-11) and needs Java 17+ (VERIFIED, connector `CHANGES.md`). `spark-4.1-bigquery`
went GA in 0.45.0 on 2026-08-21 — so a 4.1 path exists too, but nothing here needs it.

**Why not a custom image with 4.1.2.** It buys a version nobody depends on and costs an
image build and a maintenance surface on the Apple-silicon/amd64 boundary CLAUDE.md
already warns about.

**Why not a plain GCE VM for Spark — the decisive argument.** A VM bills while it *exists*,
not while it *works*. The budget is a fixed credit. Serverless bills per second from
submit to finish and then stops (VERIFIED: "scale-to-zero ensures you never pay for idle
capacity"). For one-shot 15–60 minute batch jobs with no persistent cluster to amortise,
that is the whole difference. Classic Dataproc clusters were rejected for the same reason
plus the $0.01/vCPU-hour management fee on top of the VM price (VERIFIED, twice on the page).

**Two things the runtime page says that the plan must respect.** Runtime 3.0 does not
support Lightning Engine / Native Query Execution (irrelevant here), and "end-user
credentials are used for all workloads by default" — the batch's service account and its
roles must be set explicitly. The required role set for a Serverless batch is not in the
research — **UNVERIFIED**; look it up before M4. Whether runtime 3.0 bundles a BigQuery
connector, and which version, is also UNVERIFIED — pass the 0.45.0 coordinate explicitly via
`spark.jars.packages` so the run does not depend on it.

**The cheap de-risking test, before any cloud spend.** Two Spark 4.1.2 facts landed in
`mllib.py` on 2026-09-01 (`targetType` defaults to `"binary"`; `TargetEncoderModel` copies
nominal metadata onto its output, tripping `VectorAssembler` / `maxBins`). Either could
differ in 4.0.1. So: pin `pyspark==4.0.1` in a throwaway dev-container build, re-run the
Phase 4b row 2 configuration (`01_mllib_baseline.py`, 612,608 rows, 5 folds,
`maxIter=100`, `maxDepth=5`, smoothing 5) and compare against the recorded baseline —
**MAE 0.5202 ±0.0050, RMSE 1.4502, R² 0.9778, 170.5 s/fold**; sweep-wide fold-to-fold SE
was ~$0.002. Inside fold noise → the version decision is closed with evidence and
`requirements.txt` moves to 4.0.1. Outside it → the difference is a finding in its own
right, and the custom-image option reopens with a reason. Cost: one container build and
~17 minutes of laptop time.

**RESULT — 2026-09-02: the two versions are bit-identical. Decision 2 closes on (a).**
The recorded baseline ran on the *host* (Python 3.13.2, macOS), so 4.0.1 in a container
against that record alone would have moved the Spark version and the environment together.
So 4.1.2 ran in the same container as the control, and the version became the only
variable. Two throwaway images off one base, differing by a single `pip install`; the
repository mounted read-only with the results directory overmounted, so no file in the tree
changed. Phase 4b row 2 exactly: 612,608 rows, 5 folds, seed 42, `maxIter=100`,
`maxDepth=5`, smoothing 5, `--cores 8`.

| Run | Spark | Python | `mae_mean` | `rmse_mean` | `r2_mean` | s/fold |
|---|---|---|---|---|---|---|
| recorded baseline (host) | 4.1.2 | 3.13.2 | 0.5201744916052814 | 1.4501613318146291 | 0.977753809523173 | 170.5 |
| parity (container) | **4.0.1** | 3.12.13 | 0.5201744916052814 | 1.4501613318146291 | 0.977753809523173 | 173.9 |
| control (container) | 4.1.2 | 3.12.13 | 0.5201744916052814 | 1.4501613318146291 | 0.977753809523173 | 192.0 |

Every metric agrees to full float64 precision, and so does every standard deviation. The
five per-fold figures agree individually too, not only in the mean — 0.5184, 0.5226,
0.5165, 0.5287, 0.5147. There is no difference to compare against the ~$0.002 fold noise
floor, which is a stronger result than the acceptance condition asked for.

The obvious objections, tested rather than assumed (D-009): the metadata records
`"spark": "4.0.1"` on `Linux-6.10.14-linuxkit-aarch64`, so the run used the pinned version;
the results directory was an empty overmount and the repository was read-only, so nothing
was read back from the old files; and mean *and* standard deviation matching across five
folds requires the fold assignment to have matched, which `--cores 8` fixes by pinning
`defaultParallelism` and therefore the read partitioning.

**Two 4.1.2 behaviours re-measured on 4.0.1, both unchanged.** `TargetEncoder` exists in
4.0.1 (Fact 1 confirmed by measurement, not by the release note). Its `targetType` still
defaults to `"binary"`, so the explicit `"continuous"` stays mandatory. `TargetEncoderModel`
still copies the indexer's nominal `ml_attr` metadata onto its output, and `* 1.0` still
clears it, so the demoting `SQLTransformer` stays mandatory. Neither is a 4.1-only quirk.

**Timing is NOT a version finding — recorded as UNVERIFIED.** The container ran 4.0.1 at
173.9 s/fold and 4.1.2 at 192.0 s/fold, but 4.1.2 ran second, after seventeen minutes of
sustained load on a laptop, and its fold 2 alone took 211.7 s. Run order and thermal state
are not separated by a single ordering, so the 10% gap buys nothing. Only the metrics close
the decision.

**Consequence, pre-registered above and now applied:** `spark/ml/requirements.txt` moves to
`pyspark==4.0.1`. Four statements that read "we run 4.1.2" were corrected with it, in
`src/mllib.py`, `01_mllib_baseline.py`, `tests/unit/ml/test_mllib.py` and the modeling
plan — a stale present-tense claim beside a changed pin is the exact failure D-009 exists
to prevent. Cost: $0, about 40 minutes of laptop time.

### 2.5 Model training — no Vertex AI

**Why not.** The fare model is a persisted estimator in `spark/ml/models/`, served by
nothing online; predictions reach Looker Studio through BigQuery. Vertex AI's value is
managed training infrastructure, tuning services and a registry/serving layer, none of
which the modeling plan calls for. A Serverless batch (MLlib) or a short-lived VM (sklearn,
§8 decision 3) that writes the artefact to `gs://…/ml/models/` is simpler and cheaper. The
cheapest general-purpose Vertex training machine is `e2-standard-4` at ~$0.154–0.161/hour
(VERIFIED on the rebranded page) — not expensive, just unnecessary.

### 2.6 Infrastructure as code — Terraform, reset then import

`terraform/main.tf` is rewritten to declare what the pipeline actually uses: the bucket
(`google_storage_bucket`, name from `var.gcs_bucket_name`, `us-central1`, uniform
bucket-level access, no public access) and the datasets `nyc_taxi_data`,
`nyc_climate_data`, `dbt_prod`, `dbt_dev`, `dbt_ci` (`google_bigquery_dataset`, location per
the 2.1 check). Provider pin `hashicorp/google` 5.6.0 stays unless `terraform init` objects.
State handling is §8 decision 5; the default is local state, **gitignored**, with the three
tracked tfstate files removed from the tree in M0. The CLAUDE.md rule survives unchanged:
never `apply` against resources that exist without `import` first.

### 2.7 Identity — one service account, keyfile locally, ADC on the VM

Per `notes/gcp-setup-runbook.md`: SA `nytaxi-pipeline`, `roles/bigquery.admin` +
`roles/storage.admin`, keyfile at `secrets/gcp-credentials.json` (gitignored), exported
through `GOOGLE_APPLICATION_CREDENTIALS`. CI gets the same key as the `GCP_SA_KEY` secret.
The VM gets the SA *attached* and uses ADC — no keyfile on the VM, in an image, or in git,
ever. The Dataproc batch runs as the same SA (roles to add: UNVERIFIED, see 2.4).
`roles/*.admin` is wider than the pipeline needs; narrowing is optional later work, not a
gate.

---

## 3. Cost model on the free trial

All prices US / `us-central1`; BigQuery in the US multi-region. Fetched from the live pages
on 2026-09-02 unless labelled otherwise.

### 3.1 Unit prices

| Service | Price | Free allowance | Label |
|---|---|---|---|
| BigQuery on-demand queries | $6.25 / TiB | 1 TiB / month | VERIFIED. The "$5/TB" seen in older material was a 25% increase on 2023-07-05 per consistent secondary sources; not a TB→TiB unit artefact (that would be ~10%). Date: CORROBORATED |
| BigQuery active logical storage | $0.000031507 / GiB-hour ≈ $0.023 / GiB-month | 10 GiB / month | VERIFIED. Long-term (90 days unmodified, per partition) ≈ $0.016 |
| BigQuery Storage Read API (what the Spark connector uses) | $1.10 / TiB | 300 TiB / month | VERIFIED |
| Dataproc Serverless, DCU-hour | $0.06 standard, $0.089 premium; per second, 1-minute minimum | none | VERIFIED. All-inclusive — no separate VM charge |
| Dataproc Serverless shuffle storage | $0.000054795 / GiB-hour standard | none | VERIFIED |
| Serverless minimum footprint: 12 DCU (driver 4 vCPU/16 GB + 2 executors 4 vCPU/16 GB; 1 vCPU = 0.6 DCU, RAM 0.1 DCU/GB) | ⇒ **$0.72 / hour at the floor** | — | CORROBORATED (composition from secondary summaries of the pricing page) |
| Dataproc classic management fee | $0.01 / vCPU-hour + Compute Engine | none | VERIFIED — rejected anyway |
| Cloud Composer | $0.06 / 1,000 milliDCU-hour; no scale-to-zero | none | VERIFIED rate; monthly floor UNVERIFIED (~$300–450 secondary) — rejected |
| GCS standard storage, US regional | not captured | 5 GB-month, 5,000 class A + 50,000 class B ops, 100 GB egress from NA | Allowance VERIFIED; unit price UNVERIFIED — at single-digit GB it is cents |
| Compute Engine, the Airflow VM | not captured for GCE proper | 1 `e2-micro`/month + 30 GB-month standard PD + 1 GB egress | Allowance VERIFIED. Proxy: Vertex's `e2-standard-4` at $0.154–0.161/h (VERIFIED there, not on the GCE page). **Verify on the Compute Engine page before M5** |

### 3.2 What the pipeline will consume

| Item | Estimate | Basis |
|---|---|---|
| Raw parquet in GCS, 2015-01–2016-12 | ~4–5 GB | `yellow_tripdata_2015-06.parquet` is 172 MB (measured on download); 24 yellow months ≈ 4.1 GB, green is far smaller. Sits on the edge of the 5 GB free allowance — expect cents, not dollars |
| ML samples in GCS | ~1–2 GB | `sample_full` + `sample_work_train`; sizes not measured, ASSUMED from row counts |
| `dbt_prod.fact_trips` logical storage | **unknown; ASSUMED 15–30 GiB** | BigQuery logical bytes are *uncompressed*; the 7.1 GB parquet is compressed. At 30 GiB: (30 − 10) × $0.023 ≈ **$0.46 / month**. Measure after M3 |
| One `dbt build --target prod` | well under 1 TiB | scans ~5 GB of raw parquet plus the marts; **free** |
| Dashboard / ad-hoc queries | free until ~40 full scans of a 25 GiB table per month | partition + cluster (2.2) shrink most scans; **free** |
| Spark reads of `fact_trips` via the connector | free | 300 TiB/month allowance against a table of tens of GiB |
| Serverless: `01_mllib_baseline.py` at work-612k (M4 smoke) | 989 s locally ⇒ ~17 min at 12 DCU ⇒ **~$0.20** | measured wall time × $0.72/h; assumes Serverless is no slower than the 11-core laptop — ASSUMED |
| Serverless: MLlib GBT on `sample_full` (10.2M train), 5 folds | 170.5 s/fold × 16.6 ≈ 47 min/fold ⇒ ~4 h ⇒ **~$3** | linear extrapolation of §5c's scale factor; a 10× miss is still $30 |
| Serverless: `00_prep_spark.py` on all 128.78M rows from BigQuery | not measured; ASSUMED < 1 h ⇒ < $1 | |
| sklearn §5c scope (top 4 + corridor-dropped champion) | ~8 h single-machine ceiling; on a short-lived VM ⇒ **single-digit dollars** at any plausible 8-vCPU on-demand rate | §5c measured 8.2 h for all 14 models; the scoped run is dominated by `stacking` (~4 h). VM price UNVERIFIED |

**Total for the whole migration, if every rule above is followed: low tens of dollars.**
The $300 credit is not at risk from work. It is at risk from *idleness*.

### 3.3 The credit-exhaustion story

| How the credit dies | Rate | Time to $300 |
|---|---|---|
| One 4-vCPU VM left running | ~$0.16/h proxy × 730 h ≈ $115/month | **under 3 months** — i.e. the whole 90-day trial |
| A Composer environment left up | ≥ $300/month (UNVERIFIED floor) | **≤ 1 month** |
| A classic Dataproc cluster left up (2 × n-standard-4) | VM cost + $0.08/h fee | weeks |
| Everything in this plan, run as written | tens of dollars | never, inside 90 days |

Controls, all set in M1 before anything else is created: a Cloud Billing budget on the
project with alerts at $50 / $150 / $250; the VM's stop rule (2.3); Serverless only for
Spark; a `gcloud compute instances list` check in the post-run steps of M4 and M5. The
90-day clock starts at signup — do not sign up before M0 is finished (§8 decision 6).

---

## 4. Migration sequence

Smallest reversible steps first; each phase has a gate and a rollback; **the per-step
review gate in CLAUDE.md applies inside every phase** — a phase is a grouping for this
document, not a batch to run unattended. TDD rule applies: where a phase changes code or
config that a test can pin, the failing test lands first.

### M0 — Hygiene, local only, no cloud, fully reversible

- [x] **Test first:** extend `tests/unit/dags/test_dag_config.py` (or a new
      `tests/unit/test_stale_ids.py`) with two guards — (a) no file under `terraform/`
      matching `*.tfstate*` is tracked by git; (b) `dbt/ny_taxi_analytics/models/staging/*.yml`
      contains no literal `dtc-de-` project ID in a `database:` line (read-only scan of the
      submodule; that is allowed). Both fail today. **Done `6315b87`:** new
      `tests/unit/test_stale_ids.py`. Guard (a) passes now. Guard (b) is
      `xfail(strict=True)`, blocked on the upstream push, and carries positive controls so
      a pattern that stopped matching cannot pass silently.
- [x] Remove the three tfstate files from the index (`git rm --cached`), add
      `terraform/*.tfstate*` and `terraform/.terraform/` to `.gitignore`. Keep the files on
      disk until M1 confirms the fresh state, then delete. **Done `6315b87`:**
      `git ls-files -- 'terraform/*.tfstate*'` is empty, both ignore patterns are in
      `.gitignore`, and all three files are still on disk as intended. **M1 deletes them.**
- [x] Rewrite `terraform/main.tf` per 2.6; keep `variables.tf` defaults. `terraform
      validate` only — no `init` against a backend yet. **Done `0b6421c`:** the bucket plus
      the five datasets, `demo_dataset` gone, `variables.tf` untouched (`var.bq_dataset_name`
      is now unreferenced, left deliberately). `terraform init -backend=false` +
      `terraform validate` → Success; no plan, no apply. The provider lock file is committed
      with darwin_arm64 *and* linux_amd64 hashes so `init` also works on the M1 VM.
- [x] **Upstream** in `ny_taxi_analytics`: **delete** the `database:` line from both
      staging schema files — *corrected 2026-09-02, see the note below;* the `fact_trips`
      partition/cluster config (2.2) after reading its current block. Merge there, then bump
      the submodule pointer here.

      **DONE 2026-09-02 — upstream `305868f`, pointer moved off `d11219d`.** The owner
      pushed all three changes in one commit. Both guards in `tests/unit/test_stale_ids.py`
      reported `XPASS(strict)` on the bump, exactly as designed, and the
      `xfail(strict=True)` marker was deleted in the same commit that recorded the pointer.
      They now assert plainly and stay as regression guards. The record of what was applied,
      kept for the next reader:

      The `database:`
      half was prepared and dry-run: it is two deletions, verified to apply cleanly against
      the currently pinned commit `d11219d`, and re-verified 2026-09-02 against the
      installed dbt 1.11.11 (`dbt/parser/sources.py:154,158` —
      `database=(source.database or default_database)`), so an absent `database` inherits
      the profile's project. In the `ny_taxi_analytics` clone:

      ```diff
      --- a/models/staging/schema_taxi.yml
      +++ b/models/staging/schema_taxi.yml
      @@ sources: - name: staging
      -    database: dtc-de-project-492321
           schema: nyc_taxi_data

      --- a/models/staging/schema_climate.yml
      +++ b/models/staging/schema_climate.yml
      @@ sources: - name: staging
      -    database: dtc-de-project-492321 # new dataset name
           schema: nyc_climate_data
      ```

      Keep `schema:` — that is the dataset, and it differs from the target. The stray
      `# new dataset name` comment goes with the line it annotated; it labelled a project
      as a dataset, which is the vocabulary collision that caused this. Then here:
      `git submodule update --remote dbt/ny_taxi_analytics && git add dbt/ny_taxi_analytics`.
      The moment the pointer bumps, the two `xfail(strict=True)` guards in
      `tests/unit/test_stale_ids.py` fail hard — that failure is the instruction to delete
      the marker.

      **The `fact_trips` half, also prepared.** 2.2 said to read the config block before
      writing it. Read: line 1 is `{{ config(materialized='table') }}` and nothing else, so
      this adds and overwrites nothing. Dry-run against `d11219d`; it applies cleanly.

      ```diff
      --- a/models/core/fact_trips.sql
      +++ b/models/core/fact_trips.sql
      -{{ config(materialized='table') }}
      +{{ config(
      +    materialized='table',
      +    partition_by={"field": "pickup_datetime", "data_type": "timestamp",
      +                  "granularity": "month"},
      +    cluster_by=["pickup_locationid", "dropoff_locationid"]
      +) }}
      ```

      All three columns exist in the model's own `select`, and `pickup_datetime` is a real
      timestamp — staging builds it with `cast(lpep_pickup_datetime as timestamp)` — so the
      `data_type` is verified, not assumed. Both halves go in **one** submodule PR, then one
      pointer bump here.
- [x] Update `.github/workflows/dbt.yml` to export `GCP_PROJECT_ID` if it does not already.
      **Done `c091af8`:** it did not, so every CI run silently targeted the dead fallback
      project. The auth step now resolves it — `vars.GCP_PROJECT_ID` first, else the
      `project_id` inside `GCP_SA_KEY` itself, else a hard error — and exports it through
      `$GITHUB_ENV`. Empty is never exported: dbt's `env_var()` returns `""` rather than its
      default when the name is set. Unverified end to end; that needs M1's live project.
- [x] The Spark 4.0.1 parity test from 2.4 (throwaway container build; no file in the tree
      changes unless the result says so). **Done 2026-09-02: bit-identical.** Full numbers,
      method and objections in 2.4. The result said so, so the pre-registered consequence
      applied: `spark/ml/requirements.txt` → `pyspark==4.0.1`, plus the four "we run 4.1.2"
      statements it makes stale. Open decision 2 is taken; **D-011** records it.
- **Gate:** `pytest tests/` — unit count up by the new guards, all green; the 9 integration
  failures unchanged. **Rollback:** `git checkout` — nothing outside the repo moved.
- **Cost:** $0.

> **Correction — 2026-09-02, the `database:` fix.** An earlier draft of this bullet said to
> write `database: "{{ env_var('GCP_PROJECT_ID') }}"` into both staging schema files. That
> works, but it is the wrong fix. **Delete the line instead.**
>
> The defect is a vocabulary collision, not a typo. dbt names things generically and
> BigQuery names them concretely, so the same two concepts carry two sets of words:
> `database:` in a schema YAML is the **project**, and `schema:` is the **dataset**, while
> `dbt/profiles.yml` calls the identical things `project` and `dataset`. The collision has
> already produced a wrong comment in the repository — `schema_climate.yml` reads
> `database: <project> # new dataset name`, labelling a project as a dataset.
>
> Verified against the installed dbt 1.11.11, `dbt/parser/sources.py:158`:
>
> ```python
> default_database = self.root_project.credentials.database
> ...
> database=(source.database or default_database),
> ```
>
> An absent `database` inherits the profile's `project`, which already resolves from
> `GCP_PROJECT_ID`. Templatising the source YAML would duplicate a value the profile
> already owns — a second place to edit and a second place to drift, which is the config
> equivalent of the raw/derived duplicate that `d83141b` cost us. **`schema:` stays**: the
> sources read `nyc_taxi_data` and `nyc_climate_data`, which are not the target dataset.
>
> Still blocked upstream either way. Deleting a line inside `dbt/ny_taxi_analytics` is
> still editing the submodule, and pushing there is the owner's.

### M1 — Provision, via Terraform, one bucket and five datasets

**Opening measurement, 2026-09-02, keyfile `secrets/gcp-credentials.json`.** The statement in
`CLAUDE.md` that "No GCP project is provisioned" is **false as measured**. The project is
live, billing is enabled, and it already holds the migrated data.

| Thing | Measured |
| --- | --- |
| Keyfile project | `dtc-de-project-506916`, SA `dtc-de-course@dtc-de-project-506916.iam.gserviceaccount.com` |
| `gcloud` active account | `saggysimmba@gmail.com`; `core/project` = `dtc-de-project-506916` |
| Billing account | `01E445-18A569-B9097E` "My Billing Account", OPEN, linked (`billingEnabled: true`) |
| BigQuery datasets | **`dbt_prod` only**, location `US`. `nyc_taxi_data`, `nyc_climate_data`, `dbt_dev`, `dbt_ci` are absent |
| Tables in `dbt_prod` | `fact_trips` 128,781,646 rows / 58.75 GB; `dim_monthly_zones_revenue` 11,572; `dim_zones` 265; `taxi_zone_lookup` 265 |
| GCS bucket | `primary-data-dtc-506916`, location **`US` multi-region**, STANDARD |
| Bucket contents | 204 objects, 7.05 GiB, all under `dbt_prod_restore/fact_trips/` |
| `billingbudgets.googleapis.com` | **not enabled** — `gcloud billing budgets list` fails `SERVICE_DISABLED` |

Three consequences the plan did not anticipate.

1. **M1 is not a $0 step on this project.** 58.75 GB in BigQuery and 7.05 GiB in GCS already
   accrue. Rough monthly list price: BigQuery active storage at $0.02/GB-month over the 10 GB
   free allowance ≈ **$0.98/month**; GCS standard US multi-region ≈ **$0.19/month**. Call it
   **~$1.20/month, already running**. The unit prices are UNVERIFIED (no primary page
   captured); the storage figures are measured.
2. **The live bucket's location contradicts `terraform/main.tf`.** The bucket is `US`
   multi-region; `main.tf` sets the bucket `location = var.region` = `us-central1`. A bucket's
   location is immutable, so importing this bucket and planning would show a **replace** —
   which destroys 7.05 GiB. On branch (A) the bucket's location must be set to `US` before any
   plan. This is separate from the dataset-location question in 2.1, which is resolved.
3. **One of the five datasets exists, not none.** A branch-(A) run imports `dbt_prod` and
   creates the other four.

**Branch chosen: (A) reuse `dtc-de-project-506916`** (owner, 2026-09-02). The live bucket and
`dbt_prod` are imported; the four missing datasets are created.

**Budget guard — DONE, 2026-09-02, before any resource was created.**

```
gcloud services enable billingbudgets.googleapis.com --project=dtc-de-project-506916
gcloud billing budgets create --billing-account=01E445-18A569-B9097E \
  --display-name="nyc-taxi-guard-50usd" --budget-amount=50USD \
  --threshold-rule=percent=1.0 --threshold-rule=percent=1.0,basis=forecasted-spend
```

The same command ran for 150 and 250. Verified by `gcloud billing budgets list`:

| Budget | Amount | Thresholds |
| --- | --- | --- |
| `nyc-taxi-guard-50usd` | 50 USD | 100% actual, 100% forecast |
| `nyc-taxi-guard-150usd` | 150 USD | 100% actual, 100% forecast |
| `nyc-taxi-guard-250usd` | 250 USD | 100% actual, 100% forecast |

Notes on the guard, all measured.

- The budgets are **billing-account wide**, not project-scoped. `gcloud billing budgets`
  takes `--billing-account`; the earlier UNVERIFIED note is now settled — it does target the
  billing account, and `billingbudgets.googleapis.com` was indeed disabled and had to be
  enabled first.
- Notification goes to the default IAM recipients (billing admins) by email.
  `disableDefaultIamRecipients` is unset.
- **A second project shares this billing account:** `project-672ad9c7-bfa8-470e-9e1`, billing
  enabled. Its spend counts against these budgets. The owner should check what it is.
- **The project is `ACTIVE`.** The stored note that the account was disabled with an appeal
  pending is stale.


- [x] Budget alerts in place before anything was created (see the block above).
- [x] Dataset-location question resolved (2.1) — `US` is correct, nothing had to change.
- [x] `terraform init` on fresh state, `import` of the two live resources, `plan`, `apply`.
- [x] Gate run. **7 failures, not the predicted 5-6.** The measurement wins (D-009).

**M1 DONE — 2026-09-02.** Verified against the cloud, not from the apply output.

*What was applied.* `Apply complete! Resources: 4 added, 1 changed, 0 destroyed.`

| Resource | Action | Verified after |
| --- | --- | --- |
| `nyc_taxi_data`, `nyc_climate_data`, `dbt_dev`, `dbt_ci` | created | all present, `location=US` |
| `dbt_prod` | imported, **no-op** in the plan | present, `location=US` |
| `primary-data-dtc-506916` | imported, updated in place | UBLA `True`, public access `enforced`, abort-incomplete-upload rule present |
| bucket objects | untouched | **204 before, 204 after** |

*How the plan was made safe.* Three things had to happen before `apply`.

1. **The bucket location was decoupled from `var.region`.** New `var.gcs_bucket_location`,
   default `"US"`, matches the immutable live location. Without it the post-import plan is a
   REPLACE, which destroys 7.05 GiB. The price of staying on the multi-region is the loss of
   the Always Free 5 GB-month allowance, which is us-central1-only — about $0.19/month.
2. **The state was made genuinely fresh.** `main.tf` now declares
   `backend "local" { path = "pipeline.tfstate" }`. The legacy `terraform.tfstate` held
   serial 3 for a **third**, dead project — `dtc-de-course-457315` — and two resources this
   config no longer declares (`demo_dataset`, `demo-bucket`). Migrating it would have made
   `plan` propose destroys. `terraform init` was answered `no` at the migration prompt. The
   three legacy `*.tfstate*` files are now deleted.
3. **The plan was not creates-only, and the owner approved the difference.** It was
   `4 to add, 1 to change, 0 to destroy`. The one change was the bucket: the config is
   stricter than the hand-made 2025 bucket (UBLA off → on, access `inherited` → `enforced`,
   lifecycle rule added). No destroy, no replace, no object deleted.

*Gate — measured, `.venv/bin/pytest tests/`.* **7 failed, 294 passed, 1 skipped, 0 xfailed.**
Baseline was 9 failures.

- **Flipped to passing (2):** `test_nyc_taxi_data_dataset_exists`,
  `test_nyc_climate_data_dataset_exists`. These were the only two failures M1 could fix.
- **Still failing (7), all for the same reason — no data is ingested yet, which is M2:**
  `test_yellow_external_table_has_rows`, `test_green_external_table_has_rows`,
  `test_climate_external_table_has_rows`, `test_yellow_taxi_has_24_parquet_files`,
  `test_green_taxi_has_24_parquet_files`, `test_taxi_zone_csv_exists`,
  `test_climate_parquet_exists`.
- **Why the 5-6 prediction was wrong.** It assumed three or four of the nine failures were
  dataset-shaped. Only two are. The other seven need objects in GCS and external tables over
  them, and both are built by the DAGs in M2, not by Terraform.
- The 1 skip is `test_docker_runtime.py`, marked "runs inside the dev container only".
  Normal on the host.
- **The passed count rose by 9, not by 2. That is not a contradiction.** The suite grew:
  295 tests were collected at the 9-failure baseline, 302 now. M0 added the other 7
  (`tests/unit/test_stale_ids.py` and the credential-shape tests). So 9 = 2 flipped +
  7 newly added and passing.

*No edit was needed to `airflow/docker-compose.yaml`.* Branch (A) reuses the same project and
bucket, and lines 69-70 already read `GCP_PROJECT_ID: 'dtc-de-project-506916'` and
`GCP_GCS_BUCKET: 'primary-data-dtc-506916'`. Recorded as a negative result.

*APIs, measured.* `bigquery`, `bigquerystorage`, `storage`, `storage-api` and
`billingbudgets` are enabled. **`dataproc` and `compute` were deliberately left disabled.**
Enabling `compute.googleapis.com` can auto-create a default VPC network, which is a resource
M1 has no business creating. They belong to M4 and M5.

- **Rollback:** `terraform destroy` would now also delete `dbt_prod` and the bucket, which
  hold real data. Do **not** use a bare destroy. Target the four new datasets instead:
  `terraform destroy -target='google_bigquery_dataset.pipeline["nyc_taxi_data"]'` and so on.
- **Cost:** the four new datasets are empty and add $0. The pre-existing ~$1.20/month
  continues. The bucket hardening changes no price.

### M2 — Prove the ingest and external-table DAGs from the laptop

**Readiness audit, 2026-09-02, before any DAG ran.** Two defects, one fixed here.

**Defect 1 — the Airflow keyfile named a dead project. FIXED.** `docker-compose.yaml`
mounted `~/.google/credentials/` from the host home and pointed
`GOOGLE_APPLICATION_CREDENTIALS` at `google_credentials.json` inside it. That host file
names **`dtc-de-project-492321`**, the exhausted trial, not the live
`dtc-de-project-506916`. Every DAG would have failed on auth before downloading anything.
The mount now reads the repo's gitignored `../secrets` and the stable filename
`gcp-credentials.json` — the same decoupling `tests/conftest.py` documents, so Airflow,
pytest and dbt share one key. Verified with `docker compose config`: the source resolves to
`<repo>/secrets` and `GOOGLE_APPLICATION_CREDENTIALS` reads
`/.google/credentials/gcp-credentials.json`. A directory mount was chosen over a single-file
mount because Docker silently creates a *directory* at a missing single-file source, which
would break a fresh clone.

**Defect 2 — the climate filename disagreed with its test. FIXED.** The DAG uploaded
`nyc_climate_data/weather_cache_sm.parquet`; `test_gcs.py:49` looked for
`nyc_climate_data/climate_data.parquet`. So `test_climate_parquet_exists` failed **even
after a perfect ingest**.

The DAG was the odd one out, not the test. Three places already named `climate_data.parquet`
— `README.md:61`, `notes/gcp-reference.md:24` and `tests/integration/test_gcs.py:49` — and
the project vocabulary is `nyc_climate_data`, `climate_external_table`, `stg_climate_data`.
`weather_cache_sm` was an accident of the upstream gist filename. So
`CLIMATE_DATA_TARGET_PARQUET` is now `climate_data.parquet`. That one constant drives the
GCS object name, the local intermediate, the conversion output and the cleanup, so one line
fixed all four. `CLIMATE_DATA_TARGET_CSV` keeps the gist's name on purpose: it is a local
intermediate that never reaches GCS, so the name documents the source.

*Left alone:* `airflow/tests/test_nyc_climate_dag.py` still says `weather_cache_sm.parquet`
under a `raw/nyc_climate_data/` prefix. Both halves were already stale — the DAG dropped the
`raw/` prefix long ago. CLAUDE.md classes those legacy TDD stubs as documentation, outside
the standard `pytest tests/` run.

**This corrects the M2 gate below.** It predicts failures drop to 1 and names `test_dbt` as
the survivor. Both halves are wrong. `test_dbt` already passes today, and with both defects
fixed M2 should reach **0 integration failures** — meeting M3's gate before M3 starts.
Treat 0 as a prediction, not a fact, and measure it (D-009).

**Verified clean — everything else M2 needs.**

| Check | Result |
| --- | --- |
| Hardcoded project/bucket IDs in the 8 DAGs | none; all read `os.environ` |
| GCS prefixes: ingest → external table → test | match for yellow, green and zone |
| Ingest window in compose | `2015-01-01`…`2016-12-31` = 24 months, matching the 24-file tests |
| TLC yellow, TLC green, zone CSV, climate gist | all HTTP 200 |
| `yellow_tripdata_2015-01.parquet` | 175,325,767 bytes, matching the plan's ~170 MB |
| Docker | running, 27.4.0 |
| M1's UBLA hardening vs the DAGs | safe — no `make_public`, `predefined_acl` or `.acl` anywhere |

**Cosmetic, not a blocker.** `nyc_taxi_gcs_{yellow,green}_dag.py:17` set
`TAXI_BIGQUERY_DATASET_ID = os.environ.get("BIGQUERY_DATASET", "nyc_tlc_trips")`. That
dataset does not exist, but the constant is assigned and never used — the ingest DAGs only
upload to GCS, and the external-table DAGs carry their own correct hardcoded dataset names.
Dead code worth deleting.

**M2 opening measurement, 2026-09-02, re-measured before the first DAG ran (D-009).**
Nothing here is carried over from M1; every row was read from the cloud or the laptop
again. All of it reproduced the M1 closing numbers exactly.

| Measured | Value |
| --- | --- |
| `pytest tests/` | **7 failed, 294 passed, 1 skipped**, 302 collected, 21.94 s |
| Failing tests | 4 × `test_gcs.py` (yellow 24, green 24, zone CSV, climate parquet), 3 × `test_bigquery.py::TestExternalTablesQueryable` (yellow, green, climate) |
| Keyfile `secrets/gcp-credentials.json` | project `dtc-de-project-506916`, SA `dtc-de-course@…` — the live project |
| Datasets, location `US` | `nyc_taxi_data` 0 tables, `nyc_climate_data` 0, `dbt_prod` 4, `dbt_dev` 0, `dbt_ci` 0 |
| `dbt_prod` tables | `fact_trips`, `dim_zones`, `dim_monthly_zones_revenue`, `taxi_zone_lookup` |
| `gs://primary-data-dtc-506916` total | **204 objects, 7.053 GiB** — all of it under `dbt_prod_restore/fact_trips/` |
| The four ingest prefixes | **absent** — zero objects, which is exactly why the 7 tests fail |
| Disk free | 48 GiB available (89 % capacity), plus 7.77 GB reclaimable in Docker images |
| Docker | 11 images, 10.49 GB; build cache 1.69 GB |

Two consequences worth stating. The restore prefix is untouched and stays untouched — M2
writes only to `nyc_taxi_data/{yellow,green}_taxi_data/`, `nyc_taxi_data/taxi_lookup_data/`
and `nyc_climate_data/`, none of which overlap it. And the 48 GiB of headroom clears the
~10 GiB floor set for the image build, so the build is not a disk risk.

**Both fixes above re-verified in the tree, not from the commit messages.**
`airflow/docker-compose.yaml:94` mounts `../secrets:/.google/credentials:ro` and line 70
reads `/.google/credentials/gcp-credentials.json`; `nyc_climate_gcs_dag.py:31` sets
`CLIMATE_DATA_TARGET_PARQUET = "climate_data.parquet"`. Neither regressed.


- [x] Credential mount and `GCP_PROJECT_ID` / `GCP_GCS_BUCKET` confirmed and fixed — see
      the readiness audit above. The two env vars already held the M1 values.
- [x] `docker compose -f airflow/docker-compose.yaml up --build` locally. Trigger, in
      order and one at a time: `nyc_taxi_zone_ingestion_dag`, `nyc_climate_data_ingestion_dag`,
      `nyc_green_taxi_data_ingestion_dag`, `nyc_taxi_data_ingestion_dag` (yellow, the big
      one — 24 × ~170 MB through the laptop's uplink). Each triggers its external-table DAG.

**The DAGs do not land paused. The plan's premise was wrong (measured 2026-09-03).**
This bullet list said `DAGS_ARE_PAUSED_AT_CREATION=true` would hold every DAG until it was
triggered by hand. It does not. That setting applies **only when Airflow first creates a
DAG record**. The compose stack's `postgres` volume survives `down` and has carried this
project's metadata since 2025, so every DAG already had a row — and with it the pause state
the owner last left it in. Six of the nine came up **unpaused**: the four external-table
DAGs, `nyc_climate_data_ingestion_dag`, and `nyc_taxi_data_ingestion_dag`.

That last one is yellow, the largest DAG, with `catchup=True` and a monthly schedule from
2015-01. The scheduler queued its first backfill run 65 seconds before the pause landed.
The run finished `download_dataset_task` and `local_to_gcs_task` before it could be caught.

*The damage, measured, not assumed.* One object,
`nyc_taxi_data/yellow_taxi_data/yellow_tripdata_2015-01.parquet`, **175,325,767 bytes** —
byte-for-byte the size this plan already recorded for that file, so the upload is correct
rather than truncated. `dbt_prod_restore/` re-counted at **204 objects, 7.053 GiB**,
unchanged. No stop condition fired.

*The decision (owner, 2026-09-03):* keep the file and resume the stated order; leave the
`postgres` volume alone and rely on explicit pausing instead. The stray run was marked
`failed` so it stops holding `cleanup_local_file_task`, and because a failed run still
counts as run, catchup will not re-issue 2015-01 when yellow's turn comes.

**The lesson, for M5.** A persisted metadata volume outranks
`DAGS_ARE_PAUSED_AT_CREATION`. On the GCE VM the first `up` will have an empty volume, so
the flag will work there exactly as written — but never trust it on a stack that has run
before. Pause explicitly, then verify with `airflow dags list`, before triggering anything.

**DAG 1 — `nyc_taxi_zone_ingestion_dag`, the credential canary. PASSED.**

| Measured | Value |
| --- | --- |
| Run | `m2_zone_20260903T035438`, **success**, 03:54:43 → 03:54:50 UTC |
| Tasks | all four success: download, upload, cleanup, trigger |
| Chained DAG | `create_external_table_taxi_zone` **success**, 03:54:50 → 03:54:53 |
| Object | `nyc_taxi_data/taxi_lookup_data/taxi_zone_lookup.csv`, **12,331 bytes** |
| BigQuery | `nyc_taxi_data.taxi_zone_external_table`, type `EXTERNAL`, **265 rows** |
| Suite | **7 failed → 6 failed**, 295 passed; `test_taxi_zone_csv_exists` flipped to pass |

This settles the question DAG 1 exists to answer. The keyfile authenticates against
`dtc-de-project-506916`, the service account may write to the bucket under UBLA, and
`BigQueryInsertJobOperator` may create an external table at location `US`. DAGs 2–4 will
not fail on credentials. Airflow is **2.10.3**; the image built clean, exit 0, and disk
held at 50 GiB free.

**DAG 2 — `nyc_climate_data_ingestion_dag`. PASSED.** Run `m2_climate_20260903T035704`,
success, 03:57:07 → 03:57:15 UTC, all five tasks green: download, convert, upload, cleanup,
trigger. `create_external_table_climate_data` followed and succeeded, 03:57:15 → 03:57:18.

| Measured | Value |
| --- | --- |
| Object | `nyc_climate_data/climate_data.parquet`, **1,037,285 bytes** |
| BigQuery | `nyc_climate_data.climate_external_table`, `EXTERNAL`, **19,260 rows** |

This proves the second blocker fix end to end. The object carries the project's filename,
`climate_data.parquet`, which is the name `tests/integration/test_gcs.py:49` looks for. The
old `weather_cache_sm.parquet` would have failed the test after a perfect ingest.

**DAG 3 — `nyc_green_taxi_data_ingestion_dag`. PASSED, 24 of 24.** Unpausing was enough;
`catchup=True` created the runs and `max_active_runs=1` ran them one at a time, exactly as
designed. 03:58:07 → 04:20:13 UTC, about 22 minutes for the whole backfill.

| Measured | Value |
| --- | --- |
| Ingest runs | **24 success, 0 failed** (2015-01 … 2016-12) |
| External-table runs | **24 success** — one per month, each triggered by its ingest run |
| Objects | **24 parquet files, 540,892,554 bytes** total |
| Per file | 22.7 MB … 26.7 MB; smallest `2015-01` at 22,732,044 B |
| BigQuery | `nyc_taxi_data.green_taxi_external_table`, `EXTERNAL`, **35,619,306 rows** |

One timing note worth keeping. Month `2015-05` took 6 minutes while its neighbours took 20
to 30 seconds each. Nothing failed and no retry fired, so this was uplink variance, not a
defect. Plan for a long tail on yellow rather than a flat per-file rate.

**DAG 4 — `nyc_taxi_data_ingestion_dag` (yellow). PASSED, 24 of 24.** 04:21:17 → 05:15:35
UTC, about 54 minutes for 3.86 GB. All 24 ingest runs and all 24 external-table runs
succeeded.

*The stray 2015-01 run recovered by itself, and that is worth stating precisely.* It was
left `failed` with two tasks forced to `failed`. On unpause Airflow applied `retries: 1`,
re-ran `cleanup_local_file_task` and `trigger_external_table`, and both passed, so the run
finished `success` with its external table triggered. `local_to_gcs_task` did **not**
re-run — the object's timestamp stayed at 03:51:53 — so the 175 MB was never re-uploaded.
The earlier note in this section predicted catchup would skip 2015-01; the file was indeed
not re-sent, but the mechanism was task retry, not a skipped run.

*One real failure, retried and recovered.* Month **2016-08** failed its first
`local_to_gcs_task` attempt after 222 seconds:

```
requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='storage.googleapis.com',
port=443): Read timed out. (read timeout=60)
```

That is a resumable-upload chunk timing out, not an auth or data fault. `retries: 1` re-ran
the task at 05:01:45 and it succeeded. Green showed the same shape earlier, where 2015-05
took 6 minutes against 20–30 seconds for its neighbours. **Conclusion: the 60-second read
timeout on a chunked upload is the one recurring failure mode of this ingest over a home
uplink, and the existing single retry absorbs it.** On the M5 VM the upload is
Google-internal, so this should disappear; if a future backfill runs from a laptop again,
raise `retries` before raising anything else.

**M2 GATE — MET. 0 integration failures.**

| Suite | Result |
| --- | --- |
| M2 opening baseline | 7 failed, 294 passed, 1 skipped |
| M2 close | **0 failed, 301 passed, 1 skipped**, 302 collected, 22.68 s |

All seven flipped, and none of them was `test_dbt`:

- `test_gcs.py::TestYellowTaxiData::test_yellow_taxi_has_24_parquet_files`
- `test_gcs.py::TestGreenTaxiData::test_green_taxi_has_24_parquet_files`
- `test_gcs.py::TestReferenceData::test_taxi_zone_csv_exists`
- `test_gcs.py::TestReferenceData::test_climate_parquet_exists`
- `test_bigquery.py::TestExternalTablesQueryable::test_yellow_external_table_has_rows`
- `test_bigquery.py::TestExternalTablesQueryable::test_green_external_table_has_rows`
- `test_bigquery.py::TestExternalTablesQueryable::test_climate_external_table_has_rows`

Nothing regressed; the one skip is the same pre-existing skip as in the baseline. This
**confirms the corrected prediction and refutes the gate text as written**, which expected
one surviving failure named `test_dbt`. `test_dbt` passed before M2 and passes now. M3's
gate is therefore already met before M3 starts.

**Closing state, measured 2026-09-03 05:16 UTC.**

| Measured | Value |
| --- | --- |
| Ingest objects | **50 objects, 4,402,064,926 bytes** (4.10 GiB) across the four prefixes |
| Yellow | 24 files, 3,860,122,756 B, **277,171,036 rows** |
| Green | 24 files, 540,892,554 B, **35,619,306 rows** |
| Zone | 12,331 B, **265 rows** |
| Climate | 1,037,285 B, **19,260 rows** |
| `dbt_prod_restore/` | **204 objects, 7.053 GiB — unchanged throughout M2** |
| All 9 DAGs | paused again |
| Disk | 48 GiB free, unchanged |

Actual upload was 4.10 GiB against the ~5 GB estimate. Per-file sizes are recorded in
`notes/gcp-reference.md` under *Archive fingerprint*, which is what §6.1 depends on.

**M2 DONE — 2026-09-03.**
- [x] Record per-file byte sizes and row counts from the GCS listing into
      `notes/gcp-reference.md` — this is the archive fingerprint §6.1 depends on.
- **Gate: MET.** `test_gcs.py` 24 + 24 parquet files, zone CSV, climate parquet;
  `test_bigquery.py::TestExternalTablesQueryable` rows > 0. Failures: → **0**, not the 1
  (`test_dbt`) this line predicted. See the gate table above.
- **Rollback:** delete the prefixes; external tables are `CREATE OR REPLACE`.
- **Cost:** 4.10 GiB in GCS, measured, against the ~5 GB estimate.

### M3 — dbt marts in the cloud, and the reproducibility check that matters

**Current record — re-run 2026-09-12, after D-012.** Everything below this block is the
voided 2026-09-03 record, kept as history. Do not quote its counts.

- [x] **Fix merged upstream.** `sinhasagar507/ny_taxi_analytics#11` merged as `25f3186`
      (admin merge, owner-approved: branch protection wants one review and the owner cannot
      approve their own PR). Pointer moved `3f927a3` → `25f3186` in `c25fb96`.
- [x] **One `dbt build --target prod`, CLI, alone.** Docker was down, so the Airflow stack
      and its 9 DAGs could not run; no dbt process on the host; no running query job in the
      project; no Actions run in progress. 2026-09-12 23:38–23:43 UTC:
      `PASS=21 WARN=0 ERROR=0 SKIP=0`. The `unique` tests on both staging `tripid`s pass.
- [x] **Counts, from `__TABLES__`:** `fact_trips` **307,339,039** rows, **130.81 GiB**
      (was 128,408,323 and 54.56 GiB — 2.39x rows, 2.40x bytes).
      `dim_monthly_zones_revenue` 11,750 rows, `dim_zones` 265.
- [x] **Caps, one grouped query**, full window, §4 guards (`fare >= 0`, `dist > 0`,
      `dur >= 1.0`), `APPROX_QUANTILES(x, 100)[OFFSET(99)]`:

  | Service | Rows | Guarded | `fare_p99` | `dist_p99` | `dur_p99` |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Yellow | 271,905,544 | 270,075,802 | 52.00 | **18.51** | **57.75** |
  | Green | 35,433,495 | 34,691,074 | 45.00 | 14.15 | 59.97 |

  Yellow guarded is **2.70x** the voided build's 99,983,155. Yellow `dist_p99` 18.51 and
  `dur_p99` 57.75 sit within 1.0% of the raw source's 18.67 and 58.32 (D-012). Yellow
  guarded is 98.37% of the raw source's 274,556,460; the cause of the 1.63% gap was not
  measured here.
- [x] **Reproducibility — NOT re-measured by a second build.** The new staging models are
      `select distinct` over the projected row, deterministic by construction (D-012), so no
      checksum scan was run. A second build would cost about $1.56 (below).
- **Cost, measured** from `JOBS_BY_PROJECT`: the build billed 256.29 GiB = **$1.56**
      (tables 96.59 GiB $0.59, tests 159.70 GiB $0.97). Verification billed 16.02 GiB
      = $0.10. Session total **$1.66**. The pre-build estimate was 302.18 GiB = $1.84, from
      the 2026-09-03 build's per-node bytes with `dim_monthly_zones_revenue` scaled 3x. A
      plain dry-run said 24.38 GiB = $0.15, because it reports 0 bytes for every read of
      the GCS external tables.
- **Gate:** `pytest tests/ --ignore=tests/unit/ml/test_oof_encode.py` — **340 passed,
      1 skipped** before the bump and after it (the ignored file is the deliberate red test
      for §5.3).
- **Not done:** Looker Studio still reads `dbt_prod.fact_trips` and every chart moved;
      `dbt_dev` holds PR 11's 2016-01 test build and was not checked.

<details><summary>Voided 2026-09-03 record</summary>

- [x] `dbt_build_marts` from the local compose stack (target `prod`), and the CLI form.
      Built to `dbt_dev` first as the dry run.
- [x] **The check, as revised:** the plan's original target of **128,781,646** turned out
      to be unreproducible — see the finding below — so the check became "does the cloud
      rebuild reproduce *itself*", not "does it match the restore". **Complete
      2026-09-03.** Two consecutive `dbt_dev` builds, a `dbt_prod` CLI build, and a
      `dbt_build_marts` DAG run (paused DAG, unpaused for the trigger, re-paused after)
      all produced **`fact_trips` = 128,408,323 rows**, with an identical content
      checksum (`BIT_XOR(FARM_FINGERPRINT(TO_JSON_STRING(t)))` over every row) across all
      four builds. The cloud rebuild reproduces itself; the restore does not, because it
      was never reproducible to begin with (next point).
- [x] **Finding — the restore's exact count was never reproducible, and it is now fixed.**
      `stg_yellow_taxi_data` and `stg_green_taxi_data` deduplicate on
      `(vendorid, pickup_datetime)` with `row_number()` and **no `ORDER BY`**. Measured:
      74,402,554 tied groups in the yellow source alone over 2015-01–2016-12, of which
      74,359,200 disagree on the pickup/dropoff zone pair. `fact_trips` inner-joins zones
      on both ends, so an arbitrary tie winner changed the row count on every build — the
      128,781,646 figure was one arbitrary draw, not a fixed target. Fixed upstream in the
      submodule (`sinhasagar507/ny_taxi_analytics@3f927a3`, pointer bumped here): both
      `row_number()` calls now order by every emitted column, so ties resolve only between
      rows identical in the output. `trip_type` and `congestion_surcharge` are excluded
      from the green tiebreak — the external table declares both `INT64` but the
      underlying 2015–2016 Parquet stores `DOUBLE`, so ordering by either throws a type
      error, and neither column is selected downstream. New reproducible count:
      **128,408,323** (373,323 below the old arbitrary draw — expected, since the fix
      changes which tied row wins, not how many trips exist).
- [x] **Known non-issue, documented not fixed:** `dim_monthly_zones_revenue.avg_montly_passenger_count`
      differs between builds at the ~1e-14 relative level. `passenger_count` is cast to
      `INTEGER` upstream, so `AVG` returns `FLOAT64`, and BigQuery's distributed summation
      order is not guaranteed identical across executions — an engine property, not a data
      bug. Verified every other column in that model (all money sums, trip counts,
      `avg_montly_trip_distance`, which averages a `NUMERIC` column) matches exactly
      across all 11,643 groups, dev vs. prod, 0 mismatches.
- [x] **Incident, self-inflicted and repaired:** while checking task logs, an
      `airflow tasks test dbt_build_marts dbt_build` command was run by mistake — that
      subcommand *executes* the task rather than displaying its log, so it ran a second,
      concurrent `dbt build --target prod` against the same tables as the real,
      just-triggered DAG run. Result: `dbt_prod.fact_trips` briefly held 513,633,292 rows
      (~4× duplication, additive, not merged/corrupted — confirmed every `tripid` present
      an exact multiple of the correct row, 5 tripids at 8× from a residual tie-break race
      at the overlap). Fixed by confirming no dbt process remained anywhere (host,
      webserver container, worker container — checked via `/proc` since the containers
      have no `ps`/`kill` binaries) and running exactly one more `dbt build --target prod`
      alone, which returned every `dbt_prod` table to the correct, checksum-verified
      counts. `gs://primary-data-dtc-506916/dbt_prod_restore/` was never touched by any of
      this — reconfirmed at 204 objects, 7.053 GiB, unchanged. All 9 DAGs, including
      `dbt_build_marts`, are re-paused, matching the M2 baseline.
- [x] Measure `fact_trips` logical bytes and record it against the 3.2 assumption.
      **Measured: 58,576,839,052 bytes = 54.56 GiB** (the checksum-verified build).
      3.2 assumed 15–30 GiB — **refuted**, the real figure is ~1.8× the top of that range.
      Storage at one copy: (54.56 − 10) × $0.023 ≈ **$1.02/month**. `dbt_dev` was dropped
      after the reproducibility check completed, so only the one `dbt_prod` copy remains
      (plus the untouched 7.053 GiB GCS restore).
- [ ] Reconnect Looker Studio to `dbt_prod`. **Owner-only** — hand off, not a blocker.
      Step-by-step workflow, traps, and current status: `notes/2026-09-04-looker-studio-reconnect-runbook.md`.
- [ ] Push the branch so CI's `dbt build --target ci` runs once with `GCP_SA_KEY` set
      (the owner pushes — audit item 2), and push the submodule's `3f927a3` to
      `origin/main` so the fix is authoritative remotely. **Owner-only** — hand off, not a
      blocker.
- **Gate:** `pytest tests/` — **0 failed, 301 passed, 1 skipped**, re-measured after all of
      the above. This was already true before M3 (M2 met it); M3 proves no regression, not
      new progress.
- **Rollback:** drop `dbt_prod` tables; reload from `dbt_prod_restore/`, or rebuild is one
      DAG run.
- **Cost:** ~$1.02/month storage on the one surviving copy; queries free (all scans this
      milestone stayed inside the 1 TiB free tier).

</details>

### M4 — Spark on Dataproc Serverless: smoke, then §5c

- [x] Look up and grant the Serverless batch's service-account roles (UNVERIFIED in 2.4).
      **Verified by use:** `prep-m4-20260913-full` (BigQuery read through the connector,
      GCS write) and `m4-smoke-20260915` (GCS read and write) both SUCCEEDED as
      `dataproc-batch@dtc-de-project-506916.iam.gserviceaccount.com`. The role list itself
      was not read from IAM; the two batches show that the roles suffice.
- [x] Upload `sample_work_train.parquet` to `gs://…/ml/samples/`. **Done 2026-09-14.** The
      pre-D-012 object (7,548,053 bytes, CRC32C `N8OPzA==`) was first copied to
      `ml/samples-pre-d012/`, equal in size and CRC32C. `ml/samples/sample_work_train.parquet`
      now holds the D-012 split: 16,406,311 bytes, `1fvgNQ==`, 1,355,641 rows.
- [x] Add a `--input` / `--output` URI pair to `01_mllib_baseline.py` if it does not take
      one (it reads local paths today); test first, in `tests/unit/ml/test_mllib.py`.
      **Done 2026-09-04, `548bfb1`.**
- [x] Submit `01_mllib_baseline.py` as a batch on runtime 3.0. **Done 2026-09-15:
      `m4-smoke-20260915`, SUCCEEDED.** Submitted without `spark.jars.packages`: the
      connector coordinate already resolved in `prep-m4-20260913-full`, and this job reads
      parquet only. Compared to a **new local run on the D-012 split**, not the 2.4
      baseline, whose 612,608 rows D-012 voided.

      | Run | Where | Master | `default_parallelism` | Rows | MAE | RMSE | R² | Elapsed |
      | --- | --- | --- | ---: | ---: | --- | --- | --- | ---: |
      | `mllib_gbt_d012`, 2026-09-14 | dev container, Spark 4.0.1 | `local[8]` | 8 | 1,355,641 | 0.466188 ± 0.005356 | 1.201377 ± 0.021579 | 0.984675 ± 0.000561 | 2,961.4 s |
      | `mllib_cloud_smoke_d012`, 2026-09-15 | Serverless 3.0, Spark 4.0.1 | `dataproc` | 8 | 1,355,641 | 0.466188 ± 0.005356 | 1.201377 ± 0.021579 | 0.984675 ± 0.000561 | 4,301.8 s |

      - **Gate PASS.** Per metric, \|cloud − local\| mean against the local std: MAE
        1.110e-16 against 0.005356; RMSE 2.220e-16 against 0.021579; R² 0 against
        0.000561. The differences are floating-point rounding. Every fold matched to four
        decimals where both logs were compared (folds 0–3), and the stds agree to the same
        precision as the means.
      - **Why they are equal:** the cloud `default_parallelism` is **8, measured**, the same
        as `local[8]`, so the read partitions and the fold membership agree. The two runs
        also cross an architecture: aarch64 locally, x86_64 in the cloud, Python 3.12.13
        against 3.12.12.
      - **Executors:** 2 registered at 14:14 UTC, on hosts 10.128.0.21 and 10.128.0.22;
        no executor was lost or removed. With master `dataproc`, the driver runs no task,
        so every task ran on them. Billed DCU averaged 14.24 while RUNNING, 3.07x a
        driver-only 4.64.
      - **Time:** fits of 711.2–766.6 s per fold, 1.37x the local mean fit; 4,301.8 s for
        the folds, 1.45x local. RUNNING 73.8 min, 74.7 min from submit, under the 100-min
        TTL (6,000 s).
      - **Cost:** 63,101.463 DCU-seconds = **$1.05**; shuffle storage 4,601,135 GB-seconds
        = 1,278.1 GB-hours = **$0.07** at $0.000054795 per GiB-hour; **$1.12** in total,
        under the $1.50 cap. The batch held 14.4 DCU, not the 13.92 that FACTS assumed, and
        1,050 GB of shuffle storage.
      - **Code:** `dependencies/m4smoke-2ad1a8f/`, built from `git show HEAD:` at `2ad1a8f`.
        Outputs: `ml/results/leaderboard_mllib_cloud_smoke_d012.csv` (225 bytes, `fdEUYg==`)
        and `sweep_mllib_cloud_smoke_d012.json` (1,222 bytes, `qj0fLQ==`).
- [x] **`m4-parity-20260905b` recorded** (2026-09-05, unrecorded until 2026-09-14). It ran
      on the pre-D-012 612,608-row split with master `local[*]` and `default_parallelism`
      4, so the driver did all the work. MAE 0.5201744916052814, equal to the local 2.4
      baseline; 2,441.8 s; 12,310.6 DCU-seconds. It tested the stack, not the executors,
      and D-012 voided its rows.
- [x] `00_prep_spark.py` reading `dbt_prod.fact_trips` through the connector instead of
      the local backup. **Done 2026-09-13** on the D-012 rebuild, so the target count is
      M3's re-run, not the retired 128,408,323. Batch `prep-m4-20260913-full`,
      SUCCEEDED on its own in 25.6 min: `raw_rows` **307,339,039** and `guarded_rows`
      **304,766,876**, both exact matches to M3; `master` `dataproc`, 4 executors.
      Caps: Yellow $52.0 / 18.5 mi, Green $44.5 / 13.9 mi, duration 57.5 min. Yellow
      matches M3's BigQuery caps to 0.01; green is 0.5 and 0.25 below them. **Measured
      2026-09-13:** both green caps sit inside the exact p98.9–p99.1 band, so the gap is
      the approximate-quantile method — "Open anomaly" in the prep-cloud-baseline note.
      **Cost: 32,053.8 DCU-seconds = $0.53**; BigQuery `JOBS_BY_PROJECT`
      billed 0 bytes for the batch (Storage Read API bytes UNVERIFIED — reads are not
      jobs). Full record, three-way cap table and submit command:
      `notes/2026-09-06-prep-cloud-baseline.md`.
- [x] **`--cluster` fixed first (`24175b0`).** Every batch before 2026-09-13 ran on the
      driver alone, `--cluster` or not — billed DCU is 1.00–1.03x a driver-only figure
      for all of them, `m4-wiring-195925` included. Serverless runtime 3.0 injects
      `spark.master=local` and `MASTER=local`; a job that sets no master gets one
      thread. `--cluster` now names `dataproc`. Three probes, $0.04 together.
- [x] §5c MLlib arm on `sample_full` — **Done 2026-09-26: `m4scale-20260926`, SUCCEEDED.**
      Row 3's configuration (D-013), `--cluster --drop-corridor`, runtime 3.0 (Spark 4.0.2),
      `maxExecutors=4`, `--ttl=175m`; code in `dependencies/m4scale-8613595/`. Input
      `ml/samples/sample_full_train.parquet`: 22,603,549 rows, 272,456,910 B, `gvslwA==`.
      Master `dataproc`, `default_parallelism` 8, 2 executors (10.128.0.27, 10.128.0.28),
      none lost. MAE 0.465098 ± 0.002346, RMSE 1.153382 ± 0.005396, R² 0.985878 ±
      0.000125; 6,583.6 s; RUNNING 112.5 min. **Cost:** 96,454.390 DCU-s = $1.61, shuffle
      1,953.6 GB-h = $0.11, **$1.71**. CV only; no holdout or temporal metric. Full record
      and the LightGBM arm: modeling plan §6, "The RESCALE results".
- [x] `gcloud dataproc batches list` — nothing running; `gcloud compute instances list` —
      nothing exists. **Measured 2026-09-15**, after `m4-smoke-20260915`: 0 batches
      running or pending, 0 instances.
- **Gate:** smoke run inside fold noise of the local result — **PASS 2026-09-15**, above.
  Prep compared to **M3's
  BigQuery numbers**, not the pre-D-012 local `prep_stats.json` (archived 2026-09-13;
  the tracked file now holds the cloud output): raw rows
  307,339,039 and guarded rows 304,766,876 (yellow 270,075,802, green 34,691,074)
  exactly. Caps inside Q1's exact p98.9–p99.1 band under option A, or equal to Q1's
  exact p99 under option B — "Open anomaly" in `notes/2026-09-06-prep-cloud-baseline.md`.
- **Rollback:** delete the `ml/` prefix. Serverless leaves nothing behind.
- **Cost:** ~$0.20 smoke; ~$3 MLlib at scale; < $1 prep (3.2).

#### The smoke — the owner's decisions and the goal (2026-09-14)

The owner accepted these on 2026-09-14, after the baseline and before any smoke work. The
SMOKE goal below runs every open smoke item in order. It is stored as the owner accepted
it; paste it as written. It is 3,975 characters.

Measured while the goal was written, 2026-09-14, read-only:

- **An unrecorded cloud run exists.** `m4-parity-20260905b` trained on the pre-D-012
  612,608-row split. Its MAE, 0.5201744916052814, equals the local 2.4 baseline exactly.
  Its master was `local[*]` with `default_parallelism` 4, so the driver did all the work,
  in 2,441.8 s for 12,310.6 DCU-seconds. It tests the stack, not the executors.
- **The old GCS split would be overwritten.** `ml/samples/sample_work_train.parquet` is the
  pre-D-012 split: 7,548,053 bytes, CRC32C `N8OPzA==`.
- **Fold membership can now depend on `default_parallelism`.** The script assigns folds per
  read partition. The old split was one row group; the new one has 2. `local[8]` gives 8;
  the cloud value is **UNVERIFIED**.
- `ml/code/spark_ml_src.zip` (2026-09-05) predates the `--cluster` fix, `24175b0`.
- Docker Desktop was not running. The host has pyspark 4.1.2; the dev container pins 4.0.1.

| # | Question | Decision | Why | Consequence |
| --- | --- | --- | --- | --- |
| 1 | One step at a time, or one goal | **One goal**, all smoke items in order | The steps are sequential, and each depends on the one before | The per-step review is suspended for the goal; its FACTS and stop rules replace it |
| 2 | What the smoke compares against | **A new local run on the 1,355,641-row split** | The 2.4 baseline ran on 612,608 rows that D-012 voided | One local run first; about 36 min, **estimated** from 989 s × 2.21 |
| 3 | Where the local run executes | **The dev container**, pyspark 4.0.1 | It matches the cloud runtime (D-011) | Docker Desktop must start first |
| 4 | The old split in GCS | **Copy it to `ml/samples-pre-d012/`, then upload** | Archive, never delete — the rule the baseline used | Both splits stay in the bucket |
| 5 | What "inside fold noise" means | **Per metric, \|cloud − local\| mean within the local std** | It is measured by the local run itself, for MAE, RMSE and R² | A FAIL on any one metric fails the gate |
| 6 | Cloud size and spend | **`maxExecutors=2`; TTL 2x the local elapsed, at most 90 min; cap $1.50** | Driver + 2 executors = 13.92 DCU = $0.84/h; 90 min is about $1.26 | About $0.63 expected, **estimated** |
| 7 | A gate FAIL | **Record it, commit, stop; never re-run** | A FAIL is a result, and a second batch spends to hide it | The owner decides what comes after a FAIL |

Options not taken: one step at a time; comparing against the voided 2.4 baseline; the
host `.venv` for the local run; overwriting the old GCS split; a fixed tolerance such as
$0.002 instead of the measured std; dynamic allocation without a cap.

<details><summary>The SMOKE goal command</summary>

```text
/goal Run the M4 MLlib smoke on the D-012 train split and record it. MET only when the transcript shows a final report headed "SMOKE DONE" with all of:
(a) M4 checklist: --input/--output (548bfb1) and roles (prep-m4-20260913-full) ticked; m4-parity-20260905b recorded;
(b) local run: 1,355,641 rows; mae, rmse, r2 mean and std; spark 4.0.1; master local[8]; default_parallelism; elapsed;
(c) GCS: the old split copied to ml/samples-pre-d012/, size and crc32c equal; the new split in ml/samples/, 16,406,311 bytes, crc32c 1fvgNQ==;
(d) code at HEAD in dependencies/m4smoke-<sha>/;
(e) batch SUCCEEDED; master dataproc; executors used; default_parallelism; rows; metrics with std; DCU-seconds, dollars;
(f) gate: per metric, |cloud - local| mean vs the local std; PASS only if all three are inside; every difference shown;
(g) batches list: none running; instances list: empty;
(h) docs, measured numbers only (D-009): migration plan M4 + Status line; modeling plan Status; next = §5.3;
(i) commits: this goal's note, then docs; gate 358 passed, 1 skipped before and after; git status clean; unpushed.
Judge IMPOSSIBLE if a line starts "SMOKE STOPPED:". Stop after 40 turns.

FACTS
- The owner suspends the CLAUDE.md per-step review for this goal.
- Read first: migration plan M4; prep-cloud-baseline note, Next; notes/decisions.md.
- Split: spark/ml/data/sample_work_train.parquet, 1,355,641 rows, 2 row groups, 16,406,311 bytes, crc32c 1fvgNQ==.
- gs://primary-data-dtc-506916/ml/samples/sample_work_train.parquet is the pre-D-012 split: 7,548,053 bytes, crc32c N8OPzA==.
- m4-parity-20260905b: local[*], parallelism 4, 612,608 rows, mae equal to local 2.4, 2,441.8 s, 12,310.6 DCU-s; driver only.
- Config: the script defaults, §5b row 2.
- Local run: dev container, pyspark 4.0.1 (D-011). Start Docker Desktop. --driver-memory 4g: 0.82 GiB peak at 612,608 rows.
- default_parallelism fixes the read partitions, so fold membership. local[8] gives 8; the cloud value is UNVERIFIED.
- Serverless driver and executor: 4 cores, 22.4 GB, 4.64 DCU each. Driver + 2 executors = 13.92 DCU = $0.84/h at $0.06/DCU-h.
- Zip: spark/ml/src/*.py plus empty __init__.py at each level. ml/code/spark_ml_src.zip predates 24175b0; never use it.
- Submit: the prep note's command, minus spark.jars.packages. gcloud CLI as saggysimmba@gmail.com; Storage client in .venv, GOOGLE_APPLICATION_CREDENTIALS=secrets/gcp-credentials.json.
- Gate: .venv/bin/pytest tests/ --ignore=tests/unit/ml/test_oof_encode.py. Last: 358 passed, 1 skipped.

STEPS
1. Gate. Commit this goal's note.
2. Local run in the container: 01_mllib_baseline.py --driver-memory 4g --tag mllib_gbt_d012.
3. Copy the old GCS split to ml/samples-pre-d012/; verify. Upload the new split; verify.
4. Upload 01_mllib_baseline.py and the zip at HEAD to dependencies/m4smoke-<sha>/.
5. Submit m4-smoke-<yyyymmdd>: --cluster, --input the GCS split, --output gs://primary-data-dtc-506916/ml/results, --tag mllib_cloud_smoke_d012, maxExecutors=2, --ttl 2x the local elapsed, rounded up to 10 min.
6. Read the batch meta and usage. Compute (f).
7. batches list; instances list.
8. Docs (h) with (a). Commit.
9. Final gate. Print "SMOKE DONE" with (a)-(i) and the next step: §5.3.

PRINT "SMOKE STOPPED: <reason>" AND END WHEN
- a row count, byte or crc32c differs from FACTS;
- the batch fails, its master is not dataproc, or no executor ran a task;
- 2x the local elapsed passes 90 min;
- (f) fails: record the numbers in M4 and commit first;
- spend passes $1.50;
- the gate shows a new failure;
- an action conflicts with a LOCKED entry in notes/decisions.md;
- the same auth or tool failure happens twice.

RULES
- Stay on refactor/wire-pipeline. Never push. No Co-Authored-By.
- GCS writes only to the paths in steps 3-5. Delete nothing. No BigQuery queries. One batch; never re-run it.
- Never read, score or print the holdout or the temporal set.
- Out of scope: §5.3, §5c, sample_full, CASE_STUDY.md (D-005), dbt edits, terraform.
```

</details>

#### The smoke stopped at the TTL limit, and the CLOUDSMOKE goal (2026-09-15)

The SMOKE goal ran on 2026-09-14 and stopped at its step 5, before the submit. Its own stop
rule fired: 2x the local elapsed, **98.7 min** (2 × 2,961.4 s), passed the 90-min TTL limit
of decision 6. **No batch ran; Dataproc spend $0.** What it finished, measured:

- **Gate:** 358 passed, 1 skipped. The goal note was committed as `2ad1a8f`.
- **Local run**, `01_mllib_baseline.py --driver-memory 4g --tag mllib_gbt_d012` in the dev
  container: 1,355,641 rows; MAE 0.466188 ± 0.005356, RMSE 1.201377 ± 0.021579,
  R² 0.984675 ± 0.000561; Spark 4.0.1; master `local[8]`; `default_parallelism` 8;
  2,961.4 s for the five folds, 3,070 s wall clock; fits of 487.0–554.3 s per fold. The
  output is in `spark/ml/results/*_mllib_gbt_d012.*`, gitignored.
- **GCS:** the pre-D-012 split was copied to `ml/samples-pre-d012/sample_work_train.parquet`,
  7,548,053 bytes, CRC32C `N8OPzA==`, equal to its source. `ml/samples/sample_work_train.parquet`
  now holds the D-012 split: 16,406,311 bytes, `1fvgNQ==`, 1,355,641 rows, 2 row groups.
- **Code:** `dependencies/m4smoke-2ad1a8f/01_mllib_baseline.py` (21,461 bytes, `Jmqp2w==`)
  and `mllib_deps.zip` (20,439 bytes, `rrH9dQ==`), both built from `git show HEAD:` at
  `2ad1a8f`. The zip holds `spark/ml/src/*.py` and an empty `__init__.py` at each level.

Two statements in the SMOKE record above were wrong, measured 2026-09-14:

- **The dev image carries pyspark 4.1.2, not 4.0.1.** `spark/ml/requirements.txt` pins
  4.0.1, but `nyc-taxi-dev:latest` was built six weeks earlier, before the D-011 pin. The
  local run installed 4.0.1 in its one-off container, as the 2.4 parity test did; the image
  is unchanged. A rebuild would align it.
- **The local run took 49.4 min, not about 36.** The estimate was 989 s × 2.21; the measured
  2,961.4 s is 1.35x that. Docker Desktop also restarted the Airflow stack and `mongodb-new`
  by their restart policies; they ran at about 5% CPU during the run.

| # | Question | Decision | Why | Consequence |
| --- | --- | --- | --- | --- |
| 1 | The TTL limit | **100 min**, the goal's own 2x rule rounded up | 2x the measured local elapsed is 98.7 min | Driver + 2 executors at $0.84/h make 100 min at most $1.39 |
| 2 | The spend cap | **Stays $1.50** | 100 min fits under it | A TTL kill cannot pass the cap |
| 3 | The local comparison | **No re-run**; the 2026-09-14 run is the reference | Its metrics are deterministic at `local[8]` | The cloud batch is the only new compute |
| 4 | The uploads | **Reuse them**, verified again before the submit | They are measured and equal to FACTS | The batch is the only GCS writer |

Options not taken: a 90-min TTL, which risks a TTL kill and permits no second batch;
`maxExecutors=4`, faster but $1.39/h and a `default_parallelism` further from the local 8.

<details><summary>The CLOUDSMOKE goal command</summary>

```text
/goal Finish the M4 MLlib smoke: one Serverless batch on the D-012 train split, gated against the 2026-09-14 local run, then record it. MET only when the transcript shows a final report headed "CLOUDSMOKE DONE" with all of:
(a) inputs re-verified: GCS objects and the local result equal FACTS; spark/ml unchanged since 2ad1a8f;
(b) batch SUCCEEDED; master dataproc; executors used; default_parallelism; 1,355,641 rows; mae, rmse, r2 mean and std; elapsed; DCU-seconds, dollars;
(c) gate: per metric, |cloud - local| mean vs the local std; PASS only if all three are inside; every difference shown;
(d) batches list: none running; instances list: empty;
(e) docs, measured numbers only (D-009): M4 checklist ticked (upload; --input/--output 548bfb1; roles prep-m4-20260913-full; submit); m4-parity-20260905b, the local run and this batch recorded; M4 Status line; modeling plan Status; next = §5.3;
(f) commits: this goal's note, then docs; gate 358 passed, 1 skipped before and after; git status clean; unpushed.
Judge IMPOSSIBLE if a line starts "CLOUDSMOKE STOPPED:". Stop after 30 turns.

FACTS
- The owner suspends the CLAUDE.md per-step review for this goal.
- Read first: migration plan M4; notes/decisions.md.
- SMOKE stopped 2026-09-14 at step 5: 2x the local elapsed, 98.7 min, passed the 90-min TTL limit. No batch ran.
- Owner decision 2026-09-15: TTL limit 100 min; spend cap stays $1.50; no local re-run.
- Local run, spark/ml/results/*_mllib_gbt_d012.*: 1,355,641 rows; mae 0.466188 ± 0.005356; rmse 1.201377 ± 0.021579; r2 0.984675 ± 0.000561; spark 4.0.1; local[8]; default_parallelism 8; 2,961.4 s.
- gs://primary-data-dtc-506916:
  ml/samples/sample_work_train.parquet 16,406,311 B, 1fvgNQ==
  ml/samples-pre-d012/sample_work_train.parquet 7,548,053 B, N8OPzA==
  dependencies/m4smoke-2ad1a8f/01_mllib_baseline.py 21,461 B, Jmqp2w==
  dependencies/m4smoke-2ad1a8f/mllib_deps.zip 20,439 B, rrH9dQ==
  ml/results/ holds no mllib_cloud_smoke_d012 object.
- m4-parity-20260905b: driver only, local[*], 612,608 rows, 2,441.8 s, 12,310.6 DCU-s.
- Driver + 2 executors = 13.92 DCU = $0.84/h; 100 min is at most $1.39. Cloud elapsed ~61 min, ESTIMATED.
- Cloud default_parallelism is UNVERIFIED; it fixes fold membership.
- No smoke batch exists; no instances.
- Submit: the prep note's command, minus spark.jars.packages. gcloud as saggysimmba@gmail.com; Storage client in .venv, GOOGLE_APPLICATION_CREDENTIALS=secrets/gcp-credentials.json.
- Gate: .venv/bin/pytest tests/ --ignore=tests/unit/ml/test_oof_encode.py. Last, 2026-09-15: 358 passed, 1 skipped.

STEPS
1. Gate. Commit this goal's note.
2. Verify (a); git diff 2ad1a8f HEAD -- spark/ml is empty.
3. Submit m4-smoke-<yyyymmdd>: the two m4smoke-2ad1a8f objects, --cluster, --input gs://primary-data-dtc-506916/ml/samples/sample_work_train.parquet, --output gs://primary-data-dtc-506916/ml/results, --tag mllib_cloud_smoke_d012, maxExecutors=2, --ttl=100m.
4. Wait. Read the run's JSON and leaderboard, the batch usage and the executor evidence. Compute (c).
5. batches list; instances list.
6. Docs (e). Commit.
7. Final gate. Print "CLOUDSMOKE DONE" with (a)-(f) and the next step: §5.3.

PRINT "CLOUDSMOKE STOPPED: <reason>" AND END WHEN
- a size, crc32c or row count differs from FACTS, or spark/ml changed;
- the batch fails or hits the TTL, its master is not dataproc, or no executor ran a task;
- (c) fails: record the numbers in M4 and commit first;
- spend passes $1.50;
- the gate shows a new failure;
- an action conflicts with a LOCKED entry in notes/decisions.md;
- the same auth or tool failure happens twice.

RULES
- Stay on refactor/wire-pipeline. Never push. No Co-Authored-By.
- GCS writes only by the batch, to ml/results/*mllib_cloud_smoke_d012*. Delete nothing. No BigQuery queries. One batch; never re-run it.
- Never read, score or print the holdout or the temporal set.
- Out of scope: §5.3, §5c, sample_full, a local re-run, the dev image, CASE_STUDY.md (D-005), dbt edits, terraform.
```

</details>

### M5 — The Airflow VM, and D-006 closed

- [x] Verify the GCE price for the chosen machine on the Compute Engine page (3.1).
      Minimum size: the compose stack runs a Postgres, scheduler, webserver, worker and
      triggerer; **the 1 GB `e2-micro` is ASSUMED too small** — an `e2-standard-2` (8 GB)
      is the first thing to try. ≤ 30 GB boot disk so a stopped VM is free.
- [x] Create the VM with SA `nytaxi-pipeline` attached, `cloud-platform` scope. Clone the
      repo on the VM, `docker compose up --build` **there** (amd64 — never push a Mac
      build). No keyfile is copied; `GOOGLE_APPLICATION_CREDENTIALS` is unset and the
      client libraries use ADC. Confirm `dbt/profiles.yml`'s `method: service-account` +
      `keyfile` lines handle an unset variable — UNVERIFIED; the `oauth` method is the
      fallback for the VM target.
- [x] Trigger one ingest DAG end to end from the VM's Airflow UI; it must reach the
      external-table DAG. Then **stop the VM.**
- **Done 2026-09-26** under the CLOSEOUT goal below, with two changes: the SA is
  `dtc-de-course` (`nytaxi-pipeline` does not exist), and the code went over as a
  `git archive`, not a clone, because the branch is unpushed. The trigger was the CLI,
  not the UI. The record is "The M5 result" below. The `dbt/profiles.yml` question in the
  second bullet stays **UNVERIFIED**: the zone DAGs never call dbt, and no dbt DAG ran.
- **Gate:** the E2E tier in CLAUDE.md's table; VM state `TERMINATED` in
  `gcloud compute instances list`.
- **Rollback:** delete the VM. Nothing else depends on it.
- **Cost:** hours × verified rate while running; $0 stopped.

#### CLOSEOUT — the owner's decisions and the goal (2026-09-26)

The owner set this goal on 2026-09-26, after Phase 5's one holdout score. It closes the
project: the case study, M5 and M6 in one goal. It is stored as accepted; paste it as
written. The decisions:

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | The per-step review in CLAUDE.md | **Suspended for this goal only** | The owner reviews the goal's commits at the end |
| 2 | D-005 | **Reopened: rewrite `CASE_STUDY.md` to fare, move it to `notes/`** — not remove it | Phase 5 landed 2026-09-26, which is D-005's Reopen-if |
| 3 | The VM's identity | **`dtc-de-course@dtc-de-project-506916.iam.gserviceaccount.com`**, attached | SA `nytaxi-pipeline` (M5's first bullet) does not exist; `dtc-de-course` read and wrote the bucket in RESCALE |
| 4 | The keyfile lines in `airflow/docker-compose.yaml` | **A VM compose override, test first**, with no keyfile env var and no `../secrets` mount | The base file needs a keyfile three ways: `GOOGLE_APPLICATION_CREDENTIALS`, `key_path` in `AIRFLOW_CONN_GOOGLE_CLOUD_DEFAULT`, the `../secrets` mount |
| 5 | Getting the code onto the VM | **`git archive` over IAP `scp`**, not a clone | The branch is unpushed |
| 6 | Which DAG | **`nyc_taxi_zone_ingestion_dag`**, which triggers `create_external_table_taxi_zone` | The smallest ingest DAG: one 12,331 B object. dbt reads zones from its seed, so the rewrite changes no mart |
| 7 | The VM | **`e2-standard-2`, us-central1-a, 30 GB, IAP SSH, `--max-run-duration=120m --instance-termination-action=STOP`** | M5's first size to try; the RESCALE lesson — the VM's end must not depend on an interactive step |

Measured at step 1, 2026-09-26: the zone object
`gs://primary-data-dtc-506916/nyc_taxi_data/taxi_lookup_data/taxi_zone_lookup.csv` is
12,331 B, CRC32C `tzgoCw==`, generation 1788407685700341, last written 2026-09-03
03:54:45Z; the instances list and the Dataproc `RUNNING`/`PENDING` batches are empty;
gate host 407 passed, 2 skipped, container 423 passed. `CASE_STUDY.md` is 315 lines, 17,707 B, from `b443a1c`, and tells the duration
story with pre-D-012 numbers.

<details><summary>The CLOSEOUT goal command</summary>

```text
Close out the project: rewrite CASE_STUDY.md to the fare story, run one Airflow DAG end to end on a VM with ADC (M5), and reconcile the documents (M6). MET only when the transcript shows a final report headed "CLOSEOUT DONE" with all of:
(a) note: the CLOSEOUT decisions + this goal in the migration plan, M5;
(b) CASE_STUDY.md rewritten to fare, moved to notes/; every number cites a committed note; audit item 4 and 11 checked; D-005 closed;
(c) TDD: a failing test for a VM compose override with no keyfile env or mount, then code;
(d) M5: price read; VM up; zone DAG and create_external_table_taxi_zone both success; object size + crc32c; VM TERMINATED; minutes, dollars;
(e) M6: CLAUDE.md GCP section replaced; audit items 5, 6, 8 checked; D-006 closed; one next pointer; the migration plan §5c line and M5/M6 checked;
(f) docs measured only (D-009);
(g) commits: note, case study, override, M5 docs, M6 docs; gate before/after; git status clean; unpushed.
Judge IMPOSSIBLE if a line starts "CLOSEOUT STOPPED:". Stop after 45 turns.

FACTS
- The owner suspends the CLAUDE.md per-step review for this goal.
- Owner: reopen D-005 — Phase 5 landed 2026-09-26, its Reopen-if. Rewrite, not remove.
- Read first: migration plan M5, M6; audit note; D-005, D-006; modeling plan §6 "The HOLDOUT result"; D-009.
- CASE_STUDY.md: 315 lines, 17,707 B, from b443a1c; duration story, pre-D-012 numbers (128.8M trips, "59-test suite").
- SA nytaxi-pipeline does NOT exist. Attach dtc-de-course@dtc-de-project-506916.iam.gserviceaccount.com (read/wrote the bucket in RESCALE).
- airflow/docker-compose.yaml needs a keyfile: GOOGLE_APPLICATION_CREDENTIALS, key_path in AIRFLOW_CONN_GOOGLE_CLOUD_DEFAULT, ../secrets mount.
- The branch is unpushed: ship code as git archive over IAP scp, not a clone.
- Zone DAG: nyc_taxi_zone_ingestion_dag -> create_external_table_taxi_zone. Object gs://primary-data-dtc-506916/nyc_taxi_data/taxi_lookup_data/taxi_zone_lookup.csv: 12,331 B, tzgoCw==. dbt reads zones from its seed, not this table.
- VM: e2-standard-2, us-central1-a, 30 GB, IAP SSH, --max-run-duration=120m --instance-termination-action=STOP. e2 rates 2026-09-25: $0.02181159/vCPU-h, $0.00292353/GiB-h = $0.0670/h.
- Gate: host 407 passed, 2 skipped; container 423 passed (2026-09-26).
- Instances list and RUNNING/PENDING batches: empty.

STEPS
1. Gate both. Commit the note.
2. Rewrite CASE_STUDY.md (writeup rules); move to notes/; index it. Commit.
3. TDD the override. Commit.
4. Read the price. VM up; archive + scp; compose up --build with the override.
5. Trigger the zone DAG; wait for both DAG runs; read the object.
6. Stop the VM. Instances list. M5 docs. Commit.
7. M6 docs. Commit.
8. Final gate. Print "CLOSEOUT DONE" with (a)-(g); next = dashboard v3 (owner-built).

PRINT "CLOSEOUT STOPPED: <reason>" AND END WHEN
- a DAG run fails, or the object differs in size from 12,331 B;
- the VM passes 120 minutes, or spend passes $1;
- a keyfile reaches the VM or an image;
- a new test fails, or passes before its code exists;
- a conflict with a LOCKED entry other than D-005;
- the same tool failure happens twice.

RULES
- Stay on refactor/wire-pipeline. Never push. No Co-Authored-By.
- Cloud writes only: the zone object, nyc_taxi_data.taxi_zone_external_table, the one VM.
- Trigger no other DAG. Delete nothing in GCS or BigQuery. No BigQuery queries.
- Case study: starter intro, no test counts, each test names its problem, plain terms; holdout numbers from §6 only.
- Out of scope: the dashboard, dbt edits, terraform, Phase 6, the PR, a push.
```

</details>

#### The M5 result (2026-09-26)

Every number below is measured.

- **Price, read 2026-09-26** from the Cloud Billing Catalog, Compute Engine, us-central1:
  E2 core $0.02181159/h (`CF4E-A0C7-E3BF`), E2 RAM $0.00292353/GiB-h (`F449-33EC-A5EF`),
  so `e2-standard-2` (2 vCPU, 8 GiB) is **$0.06701142/h**, equal to FACTS. Standard PD
  $0.04/GiB-month (`D973-5D65-BAB2`). External IP on a standard VM: $0 for the first 720
  h, then $0.005/h (`C054-7F72-A02E`).
- **The override, test first (`18893f6`).** `airflow/docker-compose.vm.yaml` resets
  `GOOGLE_APPLICATION_CREDENTIALS`, sets `AIRFLOW_CONN_GOOGLE_CLOUD_DEFAULT` to a bare
  `google-cloud-platform://`, and replaces the volumes without `../secrets`, for all seven
  Airflow services. `tests/unit/test_airflow_vm_override.py` failed first on the missing
  file. Its `docker compose config` layer then caught a real defect: a `!reset` reached
  through a YAML alias loses its tag, and Compose 2.31 kept the keyfile variable. The
  resets are written inline per service.
- **The VM.** `closeout-airflow-vm`, `e2-standard-2`, us-central1-a, Debian 12, 30 GB
  `pd-standard`, `dtc-de-course@...` attached with the `cloud-platform` scope,
  `--max-run-duration=120m --instance-termination-action=STOP`. Created 19:21:51Z;
  RUNNING from 19:22:00.820Z.
- **The first IAP `scp` failed** ("Connection closed"): it ran before the guest had
  generated its host keys (19:22:20Z) and started `ssh.service` (19:22:26Z). The retry,
  after the serial console showed `ssh.service` started, succeeded. One failure, not two.
- **Code:** `git archive` of `airflow/` and `dbt/` at `18893f6`, 19,888 B, 31 entries,
  none matching `secret`, `credential` or `.json`. Docker from `get.docker.com`: Docker
  Compose v5.5.1. `.env` held only `AIRFLOW_UID`.
- **No keyfile reached the VM or an image.** A `find` over the VM's disk for
  `gcp-credentials.json` or a `.json` under a `secrets` path returned three Google Cloud
  SDK test fixtures (`fake_client_secrets.json` and two others under
  `/usr/lib/google-cloud-sdk/.../testdata` or `tests/data`), nothing else. The merged
  config on the VM had the three keyfile strings only in the inert `x-airflow-common`
  block; every service had none. No container mounted a `secrets` source. Inside the
  worker: no `GOOGLE_APPLICATION_CREDENTIALS`, no `/.google`, and `google.auth.default()`
  returned compute-engine `Credentials` for `dtc-de-project-506916`; the metadata server
  named `dtc-de-course@dtc-de-project-506916.iam.gserviceaccount.com`.
- **The stack.** `docker compose -f docker-compose.yaml -f docker-compose.vm.yaml up -d
  --build` on the VM (amd64). All containers up by about 19:28Z; the scheduler's Docker
  health check reads `unhealthy`, but `airflow jobs check --job-type SchedulerJob` found
  one alive job, and it scheduled the runs below. The same unhealthy status was seen
  locally in TUNE; its cause is **UNVERIFIED**.
- **The runs.** Both DAGs unpaused; only `nyc_taxi_zone_ingestion_dag` triggered, run
  `closeout_m5_zone`, 19:34:59Z.

  | DAG | run | state | start | end |
  |---|---|---|---|---|
  | `nyc_taxi_zone_ingestion_dag` | `closeout_m5_zone` | **success** | 19:34:59.86Z | 19:35:25.68Z |
  | `create_external_table_taxi_zone` | `manual__2026-09-26T19:35:09.189963+00:00` | **success** | 19:35:25.59Z | 19:35:30.68Z |

  Tasks: `download_dataset_task`, `local_to_gcs_task`, `cleanup_local_file_task`,
  `trigger_external_table`, `create_external_table` — all `success`.
- **The object**, read after the run:
  `gs://primary-data-dtc-506916/nyc_taxi_data/taxi_lookup_data/taxi_zone_lookup.csv`
  **12,331 B, CRC32C `tzgoCw==`**, generation 1790451305011309, written 19:35:05Z —
  equal in size and checksum to the generation of 2026-09-03 it replaced.
- **Stopped.** `gcloud compute instances stop` at 19:36:30Z; `lastStopTimestamp`
  19:37:07.485Z. `gcloud compute instances list`: `closeout-airflow-vm` **TERMINATED**,
  the only instance. Dataproc `RUNNING`/`PENDING` batches: empty.
- **Cost: 15.11 minutes RUNNING (0.2519 h).** VM $0.06701142 x 0.2519 = **$0.0169**;
  external IP at most $0.0013 (inside the SKU's free first 720 h if unused this month);
  disk while running $0.0004 at list price. **About $0.019 in total**, under the $1 cap.
  The stopped 30 GB standard disk sits inside the 30 GB-month Always Free allowance (2.3).
- **Gate** (E2E tier, CLAUDE.md): both DAG runs `success`, VM `TERMINATED`. **M5 is
  complete, and D-006 is decided: keep the Deployment split** (closed in M6).

### M6 — Documents

- [ ] CLAUDE.md: replace the "GCP — not provisioned" section; keep "Deployment split"
      (D-006 resolved: keep). `notes/gcp-reference.md`: live layout, fingerprint table,
      the post-run cost checks. `notes/decisions.md`: D-006 entry closed with this file as
      the reason; §5's D2 reversal recorded as a new entry.
- [ ] `notes/2026-08-22-repo-audit.md`: check off 5, 6, 8; the workstream order (item 6)
      is: **this migration through M3 → modeling §5 encoder + §5c → dashboard v3.**
      Item 8's "one next pointer" is: CLAUDE.md → this file → the modeling plan Status.
- [x] `spark/ml/requirements.txt` to `pyspark==4.0.1` if M0's parity test passed.
      **Done early, in M0 (`cc8b7a9`).** 2.4 pre-registered the pin move as the consequence
      of a result inside fold noise, and the result was bit-identical, so it landed with the
      measurement rather than waiting for M6. The four "we run 4.1.2" statements it made
      stale moved with it.
- **Gate:** `pytest tests/` green; `test_docker_runtime.py` pins still hold.

---

## 5. The PySpark modeling question

### 5.1 What is being reversed

Modeling plan §10 (2026-07-30) adopted **D2: `preprocess.py`, `evaluate.py` and the
Phase-4 sweep stay scikit-learn**, on three grounds: at 765K rows Spark is slower on one
machine; Spark ML lacked a `TargetEncoder`; a third of the model families do not exist in
MLlib. The second ground was wrong (`TargetEncoder` exists since 4.0.0; corrected
2026-08-09). The owner now asks for modeling in PySpark. This plan treats that as a
requirement and records here whether the reversal is justified on evidence. **It is not
yet — and the thing that would justify it is buildable.**

### 5.2 The measured evidence (identical 612,608 rows, 5-fold CV, 2026-09-01)

| Row | Stack | `od_corridor` | MAE | RMSE | R² | s/fold |
|---|---|---|---|---|---|---|
| 1 | sklearn `lightgbm` | target-encoded, **cross-fitted** | **0.3503** ±0.0015 | **1.0516** | **0.9883** | **1.0** |
| 3 | MLlib GBT | dropped | 0.4828 ±0.0038 | 1.2582 | 0.9833 | 51.4 |
| 2 | MLlib GBT | target-encoded, smoothing 5, **not cross-fitted** | 0.5202 ±0.0050 | 1.4502 | 0.9778 | 170.5 |

Seven-arm smoothing sweep (200,000 rows, 3 folds, `maxIter=20`, mean MAE): corridor
dropped **0.6329**; s=5 0.6650; s=1 0.6713; s=0.067 0.6908; s=100 0.7094; s=20 0.7102;
s=500 0.7236. **No smoothing value beats dropping the corridor.** The result replicates.

**Root cause, identified but not yet isolated:** sklearn's `TargetEncoder` cross-fits
inside `fit_transform` (measured 2026-08-09 on a 12-row worst case: encoded values exclude
the row's own target; a spy transformer confirmed `Pipeline.fit` takes that path). Spark's
`TargetEncoder.fit` takes the plain per-category mean, so a training row's own fare enters
its own feature. With 5,373 of 18,668 corridors holding exactly one trip (28.8% of
corridors), the encoding for those rows *is* the label, the booster over-trusts it, and
the feature goes net-negative. §5b says plainly that the run establishes the effect, not
the mechanism, and that isolating the mechanism needs "a cross-fitted encoding computed
outside MLlib and fed in as a plain column".

That is the thing which has to exist before PySpark modeling is defensible. Without it, a
PySpark model is one that cannot use the project's strongest feature.

### 5.3 The out-of-fold target encoder in PySpark — specification

**Re-based to D-012 on 2026-09-15** — "Re-based to D-012" at the end of §5.4 supersedes the
612,608-row figures below and adds the tests the wiring needs.

Module `spark/ml/src/oof_encode.py`, one pure function over DataFrames, unit-tested first
with a local `SparkSession` fixture (these tests will take seconds, not milliseconds; that
is a known cost — §10 said so — and it is confined to one test module).

```
oof_target_encode(df, key_col="od_corridor", target_col="fare_capped",
                  fold_col="oof_fold", k=5, smoothing=5.0) -> DataFrame
```

1. **Fold assignment from a stable key, not `monotonically_increasing_id()`.** That
   function is not stable across re-materialisation, so a bug would be silent. Use
   `F.pmod(F.crc32(F.col(row_key)), k)`. **Finding:** `00_prep_spark.py` drops `tripid`
   (it is in the ID/unused drop list), so `sample_work_train.parquet` carries **no stable
   row key**. The prep's `keep` list needs `tripid` (or a persisted row id) — the same
   prep re-run §4a's deferred temporal split is already waiting on (`pickup_datetime`).
   Do both in one re-run; note the re-run regenerates the samples and therefore the
   holdout partition, so it happens *before* Phase 5, never after.
2. For each fold `f`: `groupBy(key).agg(sum(target), count())` over rows with
   `fold != f`; join onto rows with `fold == f`; encode as
   `(n·mean_cat + smoothing·global_mean) / (n + smoothing)` — the formula
   `self_leakage_weight()` in `mllib.py` already documents — with `global_mean` also taken
   from the complement. Keys absent from the complement → `global_mean`.
3. Union the k frames. Every training row now carries an encoding computed without its
   own label. k group-by + join passes; at 612K rows this is seconds, at 10.2M it is
   minutes.
4. Test-fold / holdout / prediction time: encode with the **full training set's** means,
   no fold exclusion — exactly what sklearn's `transform()` does. Unknown → global mean,
   matching both stacks today (§5b: "unknown categories agree").
5. Feed the column to the existing `mllib.py` pipeline as a plain double in place of the
   `StringIndexer` + `TargetEncoder` stages. Remove the stages, keep everything else — the
   parity rule holds.

**Tests to write first** (`tests/unit/ml/test_oof_encode.py`): the §5b 12-row worst case
(every corridor unique, targets 10–21, mean 15.5) — no encoded value may equal its row's
own target, and each must equal the smoothed complement mean; a repeated corridor across
folds encodes from the other folds only; an unseen key at transform time returns the
global mean; the same input yields the same output across two `SparkSession`s (stability
of the fold key); `smoothing=0` with a key present in every complement returns the plain
complement mean.

**Nested CV cost, stated so it is not discovered later.** The outer harness is 5 folds;
each outer training half needs its own OOF encoding fitted on those rows only, so a
5-fold outer run costs 5 × k inner passes. At k=5 that is 25 group-by/join passes per
model. Fine on Serverless at $0.72/h; slow in the laptop TDD loop. The §10 warning stands:
this is the one place a bug is silent leakage that inflates R². The tests above are the
defence; the second defence is the acceptance criterion below, which cannot be gamed
upward by leakage because the *test* folds are untouched.

### 5.4 Acceptance criterion — what would justify the reversal

**Re-based to D-012 on 2026-09-15** — the rows below name the voided 612,608-row split.
"Re-based to D-012", at the end of this section, gives the current comparison.

Run row 2 again with the OOF column: `mllib_gbt_oof@work612k`, same rows, same folds as
`01_mllib_baseline.py`, same GBT settings.

| Outcome | Reading | Consequence |
|---|---|---|
| MAE materially below row 3 (0.4828) — the corridor turns net-positive | mechanism isolated: it was the uncross-fitted encoder | the reversal of D2 has an evidential basis; §5c runs the MLlib arm with the OOF column; the PySpark stack is a fair comparator |
| MAE between row 2 and row 3 | partial: encoder was part of the story, MLlib's GBT the rest | report it; MLlib arm runs with the OOF column, but "PySpark modeling" stays a demonstrated *baseline*, not the champion track |
| MAE at or above row 2 | the mechanism was not the encoder | negative result, reported; D2 stands and is re-recorded with the new reason |

Either way the mechanism question §5b left open gets closed, which is a result the
write-up needs regardless of the stack decision.

#### Re-based to D-012, and the owner's decisions (2026-09-15)

§5.3 and §5.4 were written on the pre-D-012 612,608-row split. D-012 voided that split and
both comparison rows. Measured 2026-09-15, read-only:

- **Row 2 exists on the D-012 split.** `mllib_gbt@work1355k`, the M4 smoke's local run:
  MAE 0.466188 ± 0.005356, RMSE 1.201377 ± 0.021579, R² 0.984675 ± 0.000561, 2,961.4 s.
  The cloud batch agrees to floating-point precision (M4).
- **Row 3 does not.** The corridor ablation has no D-012 run, so §5.4 cannot be read
  until it runs.
- **`tripid` exists and is unique in `sample_work`** (1,828,181 of 1,828,181 rows), but the
  train split drops it. The split holds 16 columns, no `tripid`, no `pickup_datetime`, and
  no nulls.
- **The feature-crc fold key is sound on this split.** 11 rows share a feature row, and
  the largest group is 2. Rows that share a key land in one fold, so each is still encoded
  from the other folds only.
- **The red test** (`4ce8e2b`, 17 tests) fails at collection, because
  `spark/ml/src/oof_encode.py` does not exist. Its docstring cites the voided 0.5202 and
  0.4828.

The red test covers the encoder function. It does not cover the wiring into the outer CV
loop, and the wiring can break silently in these ways:

| # | Break | Kind | Defence, test first |
| --- | --- | --- | --- |
| 1 | One encoding for the whole split, before the outer fold loop: outer-test fares enter the training features | leakage | **Fare-change test:** change only the outer test fold's fares; the train-half and test-fold encodings must not move |
| 2 | The outer test fold encoded with the fold-exclusion function | train/test mismatch | **Test-fold mapping test:** test-fold values equal `apply_target_encoding(fit_target_encoding(train half))` |
| 3 | The test fold's fallback mean taken from the whole frame | leakage | The fare-change test |
| 4 | `split_column_groups` drops `od_corridor_te` and `oof_fold` as unrecognised and keeps `od_corridor` target-encoded, so an "OOF" run trains the old encoder | column | **Stage/column test:** no `TargetEncoder`; `od_corridor_te` in the assembler; raw `od_corridor` and `oof_fold` out |
| 5 | `model_name` knows only `_nocorr`, so the OOF row takes row 2's label | column | **`_oof` label test**, on a pure helper in `mllib.py` |
| 6 | The crc32 key's column list is not pinned | fold key | **Key-column test:** the key uses the feature columns only — not the fare, `_fold` or the output |

Two conditions keep the comparison fair: every row runs at Spark 4.0.1 with `--cores 8`
(`default_parallelism` 8), and `_fold` is assigned and cached before any OOF join. Whether
4.0.1 and 4.1.2 turn a double into the same string for the key is **UNVERIFIED**; running
every row at 4.0.1 keeps that question out of the comparison.

Decisions 1–3 were accepted on 2026-09-15. Decisions 4 and 5 come with the OOF goal and
are accepted when the owner pastes it.

| # | Question | Decision | Why | Consequence |
| --- | --- | --- | --- | --- |
| 1 | The inner fold key | **crc32 over the feature columns**, as the red test has it | The train split, row 2 and the smoke stay valid; the key is stable and measured sound | No `tripid` in the train split; no new split |
| 2 | Tests before the wiring | **The five tests above** (breaks 1 and 3 share one) | The red test covers the function only | Step 4 lands test first |
| 3 | Where the rows run | **Local, dev container, Spark 4.0.1, `--cores 8`** | Same rows and folds as row 2; D-011 | $0 |
| 4 | What "materially" means in §5.4 | **More than the fold std** of the compared run | The measure the M4 smoke gate used | A difference inside the std reads as no difference |
| 5 | How the work runs | **One goal, OOF**, below | The steps are sequential | The per-step review is suspended for the goal |

Options not taken: `tripid` in the train split; one encoding before the outer loop; a fixed
tolerance for "materially"; the cloud for §5.4.

<details><summary>The OOF goal command</summary>

```text
/goal Build and wire the §5.3 out-of-fold encoder, then run the §5.4 acceptance on the D-012 split. MET only when the transcript shows a final report headed "OOF DONE" with all of:
(a) row 3 local: mllib_gbt_nocorr_d012 on 1,355,641 rows; mae, rmse, r2 mean and std; spark 4.0.1; local[8]; default_parallelism 8; elapsed;
(b) oof_encode.py: the key-column test shown failing, then all of test_oof_encode.py passing; its docstring re-based to D-012;
(c) wiring: the fare-change leakage, test-fold mapping, stage/column and _oof label tests shown failing, then passing;
(d) §5.4 run mllib_gbt_oof_d012: same rows, folds and GBT settings; mae, rmse, r2 mean and std; spark 4.0.1; default_parallelism 8; elapsed; [cols] show od_corridor_te, no TargetEncoder;
(e) §5.4 reading: OOF against row 2 (0.466188) and row 3, each difference against the fold std; the outcome row named;
(f) docs, measured numbers only (D-009): migration plan §5.4 + Status; modeling plan §5b (a D-012 table) + Status;
(g) commits: this goal's note, each TDD change, then docs; gate before 358 passed, 1 skipped; after: tests/ without --ignore, 0 failed; git status clean; unpushed.
Judge IMPOSSIBLE if a line starts "OOF STOPPED:". Stop after 50 turns.

FACTS
- The owner suspends the CLAUDE.md per-step review for this goal.
- Read first: migration plan §5.3-§5.4 and its re-base; tests/unit/ml/test_oof_encode.py; notes/decisions.md.
- Owner decisions 2026-09-15: fold key = crc32 over the feature columns, not tripid; the five added tests; every run Spark 4.0.1, --cores 8; "materially" = more than the fold std; local only, $0.
- Train split: spark/ml/data/sample_work_train.parquet, 1,355,641 rows, 16 columns, no nulls, 16,406,311 B, crc32c 1fvgNQ==; 11 duplicate feature rows, largest group 2.
- Row 2 local 2026-09-14: mae 0.466188 ± 0.005356; rmse 1.201377 ± 0.021579; r2 0.984675 ± 0.000561; 2,961.4 s; mean fit 532.8 s.
- Row 3: ~14 min, ESTIMATED.
- Red test: 17 tests; API add_oof_fold, oof_target_encode, fit_target_encoding, apply_target_encoding; output od_corridor_te; no module yet.
- Column trap: split_column_groups drops od_corridor_te and oof_fold as unrecognised and keeps od_corridor target-encoded; model_name knows only _nocorr.
- Local runs: dev container (image has pyspark 4.1.2): pip install pyspark==4.0.1 first; --driver-memory 4g; start Docker Desktop.
- Gate: .venv/bin/pytest tests/ --ignore=tests/unit/ml/test_oof_encode.py. Last 2026-09-15: 358 passed, 1 skipped.

STEPS
1. Gate. Commit this goal's note.
2. Row 3: 01_mllib_baseline.py --drop-corridor --driver-memory 4g --tag mllib_gbt_nocorr_d012.
3. TDD: the key-column test red; build spark/ml/src/oof_encode.py until test_oof_encode.py is green; re-base its docstring. Commit.
4. TDD: the four wiring tests red; add --oof: an OOF encoding per outer train half; the test fold via fit + apply on the train half; no StringIndexer/TargetEncoder on od_corridor; label _oof. Commit.
5. §5.4: 01_mllib_baseline.py --oof --driver-memory 4g --tag mllib_gbt_oof_d012.
6. Compute (e). Docs (f). Commit.
7. Final gate without --ignore. Print "OOF DONE" with (a)-(g) and the next step: the owner's §5.4 decision.

PRINT "OOF STOPPED: <reason>" AND END WHEN
- a row count, byte or crc32c differs from FACTS;
- a new test passes before its code exists;
- after the code, a leakage test fails or a row reads back its own fare;
- the §5.4 [cols] lines show a TargetEncoder or no od_corridor_te;
- a local run fails;
- the gate shows a new failure;
- an action conflicts with a LOCKED entry in notes/decisions.md;
- the same auth or tool failure happens twice.

RULES
- Stay on refactor/wire-pipeline. Never push. No Co-Authored-By.
- No cloud: no GCS writes, no Dataproc, no BigQuery. Delete nothing.
- Never read, score or print the holdout or the temporal set.
- Assign and cache _fold before any OOF join.
- Out of scope: §5c, sample_full, tripid in the train split, the dev image, a prep re-run, CASE_STUDY.md (D-005), dbt edits, terraform.
```

</details>

#### The §5.4 result on the D-012 split (2026-09-15)

The OOF goal ran. `spark/ml/src/oof_encode.py` (`8f67d17`) and the `--oof` wiring
(`36cedd5`) landed test first. The five tests above failed before their code existed,
and all pass after it. Both leakage tests also fail on two mutants: one encoding before the
outer loop, and a test-fold fallback mean from the whole frame. The key-column test fails on
a key that also reads the fare.

All three rows: the 1,355,641-row train split, 5 folds, seed 42, GBT `maxIter=100`
`maxDepth=5`, smoothing 5, Spark 4.0.1 in the dev container, `local[8]`,
`default_parallelism` 8. The same seed and parallelism give the same fold membership.

| Row | Run | `od_corridor` | MAE | RMSE | R² | Elapsed |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2 | `mllib_gbt@work1355k` (2026-09-14) | `TargetEncoder`, not cross-fitted | 0.466188 ± 0.005356 | 1.201377 ± 0.021579 | 0.984675 ± 0.000561 | 2,961.4 s |
| 3 | `mllib_gbt_nocorr@work1355k` | dropped | **0.453134** ± 0.003220 | **1.138785** ± 0.006900 | **0.986234** ± 0.000206 | 2,055.2 s |
| OOF | `mllib_gbt_oof@work1355k` | out-of-fold, 5 inner folds | 0.462342 ± 0.000505 | 1.161502 ± 0.011404 | 0.985679 ± 0.000304 | 2,886.7 s |

The OOF run's `[cols]` lines show `od_corridor_te` in the numeric group and an empty
target-encoded group. Its stages are three `StringIndexer`, three `OneHotEncoder`, a
`VectorAssembler` and the `GBTRegressor`, with no `TargetEncoder`. The mean fit is 473.8 s,
and the mean encode is 67.5 s per outer fold.

**The reading, by decision 4** (a difference inside the compared run's fold std is no
difference):

| Comparison | MAE | RMSE | R² |
| --- | --- | --- | --- |
| OOF − row 2 | −0.003846, **inside** row 2's std 0.005356 (0.72×) | −0.039876, outside 0.021579 (1.85×) | +0.001003, outside 0.000561 (1.79×) |
| OOF − row 3 | +0.009208, outside row 3's std 0.003220 (2.86×) | +0.022716, outside 0.006900 (3.29×) | −0.000555, outside 0.000206 (2.69×) |

- **Outcome row, on MAE as §5.4 states it: "MAE at or above row 2".** The OOF MAE is
  0.003846 below row 2, inside row 2's fold std, so it reads as equal to row 2. Read that
  way, the uncross-fitted encoder was not the mechanism.
- **RMSE and R² disagree.** On both, the OOF row is materially better than row 2 and
  materially worse than row 3. That is the "between row 2 and row 3" row: the encoder was
  part of the story.
- **Every metric agrees on one point:** the corridor stays net-negative in MLlib. Row 3,
  without the corridor, is materially better than the OOF row on all three metrics. Fold by
  fold, the OOF MAE is higher than row 3 on all 5 folds, by 0.0045 to 0.0134.
- Row 2's per-fold values are not recorded, so a paired comparison with row 2 is
  UNVERIFIED.

**The leakage check on real data.** On outer fold 0 of the train split (host Spark 4.1.2,
`local[4]`, so the fold membership can differ from the runs), `encode_outer_fold` gave:
0 of 1,084,270 train-half rows with an encoding equal to the row's own fare; 5,482 rows in
single-trip corridors, whose encodings span 12.6710 to 12.6886 around the train-half mean
12.6817; a correlation of −0.0103 between those encodings and the fares, against 1.0000
for an in-sample encoding of the same rows; no null encodings in either half. The holdout
and the temporal set were not read.

**Two run notes.** The harness stopped the host `docker compose run` client during the OOF
run, at low host memory; the container ran on, and `docker logs -f` captured its output to
the end (`MLLIB BASELINE OK`). The encode time rose to 139.5 s in fold 3 under that memory
pressure, so the OOF elapsed time is not evidence about cost (D-011). Row 3 took 2,055.2 s
against the ~14 min the goal estimated.

**Next: the owner's §5.4 decision** — which metric names the outcome row, and so whether D2
is re-recorded with the new reason or the MLlib arm runs §5c with the OOF column.

#### The owner's §5.4 decisions and the SWEEP goal (2026-09-15)

Three findings, measured 2026-09-15 before the decisions:

- **§5c's model list is voided.** Its "top 4 by RMSE" comes from `sweep_work.json`, on the
  612,608 pre-D-012 rows. No sklearn sweep exists on the D-012 split, so row 1 of the stack
  comparison is missing too.
- **The container carve equals the saved split in rows and order, not to the last bit.**
  The host carve equals `sample_work_train.parquet` row for row. The dev container carves
  the same 1,355,641 rows in the same order; only `pickup_hour_sin` and `pickup_hour_cos`
  differ, by at most 5.55e-17 (numpy 2.4.6 in the container, 2.3.2 on the host).
- **`sample_full` is in the bucket only**, at 30,482,494 rows — 2.4x the 12.75M rows §5c's
  sizing assumed. §5c needs re-scoping after the sweep.

| # | Question | Decision | Why | Consequence |
| --- | --- | --- | --- | --- |
| 1 | Which metric names the §5.4 outcome row | **MAE, as §5.4 states** | §5.4's table is written on MAE; decision 4 reads a difference inside the fold std as none | Outcome "at or above row 2": the encoder was not the mechanism. D2 stands, re-recorded with the new reason. The §5c MLlib arm is row 3's configuration (corridor dropped), the best measured MLlib row on all three metrics |
| 2 | Which models the D-012 sklearn sweep runs | **All 14** | §5c's "top 4" reads the whole ranking, and the ranking is voided | ~66 min ESTIMATED; stacking UNVERIFIED |
| 3 | The Airflow stack during the sweep | **Stop it, start it after** | It holds ~3.5 GiB of the Docker VM's 11.67 GiB; the OOF run's client was stopped at low host memory | Reversible; nothing deleted |
| 4 | Where the §5.4 outcome is recorded | **D-013 in `notes/decisions.md`**, in the SWEEP goal | Open decision 4 asks for a successor to D2 | The docs commit adds the entry |

Options not taken: RMSE and R² as the §5.4 metric (the "between" row, with the OOF column
in §5c); deferring the §5.4 choice until row 1 exists; the old top 6 only; lightgbm only;
leaving the Airflow stack running; recording the decision in the notes only.

<details><summary>The SWEEP goal command</summary>

```text
/goal Re-run the Phase-4 sklearn sweep on the D-012 split, record the §5.4 decision as D-013, and re-derive §5c's model list. MET only when the transcript shows a final report headed "SWEEP DONE" with all of:
(a) sweep work_d012: 14 models; [split] temporal 133,629, holdout 338,911 SEALED, train 1,355,641; per model mae, rmse, r2 mean and std, fit_time_s; python, numpy, sklearn, lightgbm, xgboost, catboost versions; elapsed;
(b) row 1 (lightgbm) against row 2 (0.466188) and row 3 (0.453134): each MAE difference against the fold std; the accuracy and s/fold ratios;
(c) the new top 4 by RMSE, against the voided list lightgbm, catboost, stacking, extra_trees;
(d) D-013 in notes/decisions.md: the §5.4 outcome on MAE, "at or above row 2", with Decision, Why, Reopen-if, Status;
(e) docs, measured numbers only (D-009): modeling plan §5b D-012 table + row 1, §5c scope, Status; migration plan Status;
(f) commits: this goal's note, then D-013 + docs; gate before and after 385 passed, 1 skipped; git status clean; unpushed; the Airflow stack running again.
Judge IMPOSSIBLE if a line starts "SWEEP STOPPED:". Stop after 40 turns.

FACTS
- The owner suspends the CLAUDE.md per-step review for this goal.
- Read first: migration plan §5.4 "The §5.4 result on the D-012 split"; modeling plan §5, §5b, §5c; notes/decisions.md.
- Owner decisions 2026-09-15: §5.4 is read on MAE, outcome "at or above row 2"; D2 stands; the §5c MLlib arm is row 3's configuration; all 14 models; stop the Airflow stack for the sweep, start it after; D-013 in this goal.
- sample_work.parquet: 1,828,181 rows; part-00000 55,157,841 B crc32c xRpTjg==; part-00001 55,316,815 B r4pp0Q==.
- Container carve, measured 2026-09-15: the same 1,355,641 rows in the same order as sample_work_train.parquet; pickup_hour_sin/cos differ by at most 5.55e-17 (numpy 2.4.6 vs host 2.3.2).
- Container: python 3.12.13, pandas 2.3.1, numpy 2.4.6, sklearn 1.7.1, lightgbm 4.7.0, xgboost 3.3.0, catboost 1.2.10; 11 CPUs, 11 GiB.
- Row 2 mllib_gbt@work1355k: mae 0.466188 ± 0.005356, rmse 1.201377 ± 0.021579, mean fit 532.8 s. Row 3: mae 0.453134 ± 0.003220. OOF: mae 0.462342 ± 0.000505.
- Time: ~66 min, ESTIMATED linear from sweep_work (1,805.9 s at 612,608 rows); stacking ~32 min of it, UNVERIFIED.
- The default tag "work" overwrites the voided sweep_work.json; use --tag work_d012.
- Airflow: 6 services running (airflow/docker-compose.yaml).
- Gate: .venv/bin/pytest tests/. Last 2026-09-15: 385 passed, 1 skipped.

STEPS
1. Gate. Commit this goal's note.
2. docker compose -f airflow/docker-compose.yaml stop.
3. Dev container: python spark/ml/01_run_sweep.py --tag work_d012 > log; capture with docker logs -f if the client stops.
4. docker compose -f airflow/docker-compose.yaml start; ps shows 6 running.
5. Compute (b) and (c).
6. D-013 and docs (e). Commit.
7. Final gate. Print "SWEEP DONE" with (a)-(f) and the next step: re-scope §5c for the 30,482,494-row sample_full.

PRINT "SWEEP STOPPED: <reason>" AND END WHEN
- a row count, byte or crc32c differs from FACTS;
- the sweep lists fewer than 14 models;
- the sweep fails or the container is killed;
- any holdout or temporal metric is printed;
- the gate shows a new failure;
- an action conflicts with a LOCKED entry in notes/decisions.md;
- the same tool failure happens twice.

RULES
- Stay on refactor/wire-pipeline. Never push. No Co-Authored-By.
- No cloud at all. Delete nothing. Never --write-train.
- Never read, score or print the holdout or the temporal set.
- No code changes; if one seems needed, stop and say why.
- Out of scope: §5c runs, sample_full, MLlib runs, the dev image, CASE_STUDY.md (D-005), dbt edits, terraform.
```

</details>

### 5.5 When PySpark is the right tool for the *model*, and what the owner gives up

**Memory is the boundary, not row count as such.**

| Frame | Rows (train) | Feature frame | Fits where |
|---|---|---|---|
| `sample_work` | 612,608 | ~0.24 GB (scaled from the 500K measurement) | anywhere |
| `sample_full` | 10.2M | **3.9 GB** measured extrapolation; ~1.3 GB after the §5c category-dtype enabler (four string columns are ~80% of the footprint) | the 18 GiB laptop, comfortably, after the enabler |
| all of `fact_trips` | ~103M | ~39 GB; ~13 GB with category dtype (ASSUMED — linear scaling of the same measurement) | a single 64 GB VM; not the laptop |
| the full TLC yellow archive 2009–2026 | roughly an order of magnitude more (ASSUMED; not counted in the research) | past any single machine the trial buys |

sklearn `lightgbm` is ~1.4 min per fold at 10.2M rows by §5c's extrapolation and would be
~15 min at 103M. So: **sklearn holds through the full 12.75M sample on the laptop, and
through all 128.78M rows on one 64 GB VM. PySpark modeling becomes necessary — not merely
possible — beyond about 10⁸ rows at the current 15-column width, or when the training
data becomes the multi-year archive (§6).** The one unknown §5c names still stands:
sklearn's cross-fitted `TargetEncoder` over 19,953 levels at 10.2M rows has never been
measured — probe it with `--only lightgbm --sample full` before sizing anything.

**What a full move to PySpark gives up at today's sizes** (measured unless marked):

- 1.5× accuracy and 50–167× wall time on identical rows, until §5.4 shows the OOF column
  recovers the accuracy half. The wall-time half does not recover on one machine — JVM,
  serialisation and shuffle are overhead with no cluster to amortise (§10).
- `ExtraTrees`, `BaggingRegressor`, `StackingRegressor` — three of the top-four-by-RMSE
  scope for §5c has `stacking` and `extra_trees` in it. XGBoost via `xgboost.spark` and
  LightGBM via SynapseML exist but are heavier dependencies (ASSUMED untested here).
- `cross_validate`'s multi-metric, per-fold arrays. `CrossValidator.evaluator` takes one
  `Evaluator` (VERIFIED, Spark JavaDoc); `avgMetrics` is per param-map for that one
  metric; `subModels` only with `collectSubModels=True`; `stdMetrics` since 3.3.0
  (CORROBORATED). Three metrics means three runs or the hand-rolled loop `mllib.py`
  already has — so `evaluate.py`'s contract is kept by *keeping* the hand-rolled loop,
  not by adopting `CrossValidator`, which is a hyperparameter-search tool.
- Millisecond unit tests. Every `SparkSession` fixture costs seconds; the TDD loop that
  carried Phases 1–3 gets slower in proportion to how much moves.

**Recommendation, stated the way the question was asked.** PySpark for the pipeline —
prep, feature engineering (D1, already adopted), the OOF encoder, the MLlib arm of every
comparison, and the §5c MLlib run on Serverless. sklearn for the champion model until the
training frame exceeds a single machine's memory, which at this width is the full
128.78M-row table on a 64 GB VM. Build the OOF encoder now regardless: it is the
prerequisite for the MLlib arm to be a fair comparison, it closes the open mechanism
question, and it is the deliverable that makes "modeling in PySpark" a defensible portfolio
claim rather than a weaker model with a Spark logo on it. Record this as a new entry in
`notes/decisions.md` that supersedes D2 in the modeling plan, with §5.4's table as the
condition.

---

## 6. Data growth

### 6.1 The TLC schema question — resolved, with a dependency

The widely repeated claim that TLC switched from pickup/dropoff latitude-longitude to
`PULocationID`/`DOLocationID` in the July 2016 files, leaving this project's 2015-01–2016-12
window straddling a schema change, **does not apply to the archive that is downloadable
today.** Checked 2026-09-02 by downloading from TLC's own CDN
(`d37ci6vzurychx.cloudfront.net/trip-data/`):

- `yellow_tripdata_2015-06`, `2016-06`, `2016-07`, `2016-12` and `green_tripdata_2015-06`
  all carry `PULocationID`/`DOLocationID` and **no** lat/long columns.
- The full `yellow_tripdata_2015-06.parquet` (172 MB, 12.3M rows) was inspected with
  pyarrow: zero nulls in either ID column, values 1–265, realistic frequencies (zone 79:
  393,995 rows). Real backfilled zone IDs, not placeholders.
- The current data dictionaries (dated 2025-03-18) list only the ID columns. Wayback
  snapshots show the lat/long-era dictionary still being served in July 2017, so the PDFs
  lagged the data by a year and are not a source for the cutover date.
- The CSV→Parquet migration happened between 2022-03-03 and 2022-05-31 (Wayback bracket);
  the retroactive zone-ID backfill evidently rode along with it. The exact date is
  UNVERIFIED and does not matter here.

**Status: risk closed for the 2015–2016 window.** The dbt staging models and the
external-table wildcards see one schema.

**The dependency it creates.** The project ingests TLC's *reprocessed* archive. TLC has
rewritten history once; nothing guarantees a later download is byte-identical to today's
(sizes, row counts, or a further backfill). Hence the fingerprint in M2 (per-file bytes
and row counts) and the M3 check (`COUNT(*) = 128,781,646`). If a re-ingest ever disagrees
with the fingerprint, the modeling samples are regenerated from the new archive and the
change is reported, not absorbed.

### 6.2 Later years

Availability (VERIFIED, TLC page 2026-09-02): yellow 2009-01 → 2026-05; green 2013-08 →
2026-05; FHV 2015-01 → 2026-04 (base-license, pickup time, zone — no fare); FHVHV
(Uber/Lyft/Via/Juno) 2019-02 → 2026-05, richer schema, "separate (and more detailed)
dataset".

What extending `INGEST_END_DATE` past 2016-12 costs and risks:

- Storage and query stay free or near-free at any span the trial would see: ~170 MB per
  yellow month, ~2 GB a year; BigQuery logical growth of perhaps 15 GiB a year (ASSUMED
  from the 3.2 estimate) at $0.023/GiB — dollars a month at a decade.
- **Column drift in later years** — the external tables are defined by wildcard over all
  files in a prefix, and additional or retyped columns in newer files can break the
  external table or the staging view. Which columns and when (congestion surcharge, airport
  fee, `passenger_count` nullability) is **ASSUMED from general knowledge, not checked in
  the research**; inspect the Parquet schema of the first file of each new year before
  ingesting it, the same way 6.1 was settled.
- **Rate-card changes.** A metered fare is near-deterministic in distance and time under
  one rate card; the 2015–16 window sits under one (§4a). Later years do not (dates
  UNVERIFIED here). A model trained across a rate change without a `year`/rate feature
  or a temporal split is wrong by construction — so extending the window is the moment
  §4a's deferred temporal test set stops being deferrable.
- This is also the point at which §5.5's memory boundary moves: three or four years of
  yellow alone is the ~10⁸-row regime where the MLlib arm stops being a baseline and
  starts being the only stack that runs.

### 6.3 OSM / OSRM routing integration — scoped, not scheduled

README Future Scope already names the design: a 263×263 zone-centroid route
distance/duration matrix from one OSRM `/table` call, joined into the feature contract as a
corridor-level feature. Facts gathered (all CORROBORATED from OSRM issues/community
sources, none from an official OSRM doc fetch):

- Image `ghcr.io/project-osrm/osrm-backend`; pipeline `osrm-extract` (car profile) →
  `osrm-partition` → `osrm-customize` → `osrm-routed --algorithm mld`.
- `--max-table-size` defaults to 100 coordinates; **263 needs it raised explicitly.** A GET
  URL-length ceiling around 350–400 pairs is well above 263.
- Geofabrik `new-york-latest.osm.pbf` is **471 MB and covers New York State**, not the
  city. Clip with `osmium extract` and a five-borough polygon first, or accept the
  superset. Rule-of-thumb RAM ≈ 5× the `.pbf` ⇒ ~2.5 GB; disk single-digit GB.

**Where it runs: locally, in Docker, once.** The output is one small table (69,169 rows),
which goes to GCS and becomes a dbt seed or a `dim_zone_routes` model upstream, joined in
`fact_trips` on the two zone IDs. Nothing about it belongs on the Airflow VM or on
Serverless. Centroids come from the zone geometry noted in the project's reference-data
memory (NYC Taxi Zones GeoParquet), not from the CSV lookup, which has no coordinates.

**Modeling consequence.** A route feature is a new version of the 15-feature contract:
new sweep on CV, new leaderboard, the sealed holdout still untouched. It is a post-Phase-5
item by the modeling plan's own order and this plan does not move it.

---

## 7. Risks, and what would make each choice wrong

| # | Risk | Mitigation | Abandon the choice if… |
|---|---|---|---|
| R1 | An always-on VM or a forgotten cluster burns the credit (3.3) | budget alerts; VM stopped by rule; `instances list` / `batches list` in every post-run step | the alerts fire at $150 with M4 unfinished — then the remaining work moves back to local and the VM is deleted, not stopped |
| R2 | Spark 4.0.1 behaves differently from 4.1.2 in `TargetEncoder` metadata or `targetType` | the M0 parity test; both quirks are already pinned in `mllib.py` and its tests | the parity run lands outside fold noise and the cause is a 4.0.x defect — reopen the custom-image option with that as the reason |
| R3 | The Serverless batch cannot authenticate or lacks roles (runtime 3.0 uses end-user credentials by default) | look up the role set before M4; run as `nytaxi-pipeline` | never a reason to abandon Serverless; a reason to fix IAM |
| R4 | Dataset location vs bucket region mismatch makes external tables fail (2.1) | resolve before `terraform apply`; locations are immutable | — (a one-time check) |
| R5 | `COUNT(*)` in M3 ≠ 128,781,646 — the archive changed under the project (6.1) | fingerprint at M2; report the delta; regenerate samples from the cloud build | the delta is large enough to change the p99 caps — then the local backup is retired as the modeling input, and `paths.py` points at the new samples |
| R6 | OOF encoder has a silent leak | TDD tests in 5.3; the 12-row worst case; the 5.4 criterion cannot be inflated by training-half leakage | a leak is found *after* an MLlib arm has been reported — the row is withdrawn and the sweep re-run, exactly as the 2026-08-01 corrupt-odometer row was handled |
| R7 | sklearn §5c scope does not fit the Serverless driver (16 GB) or the laptop | category-dtype enabler first; probe `--only lightgbm --sample full`; short-lived VM as the fallback (§8 decision 3) | — |
| R8 | Submodule fix upstream stalls (owner's other repo, CI there) | it is a two-line YAML change; the pointer bump is the only change here | never — no workaround inside the submodule directory is acceptable |
| R9 | Terraform `apply` collides with something that already exists in the project | `plan` shows creates only, or `import` first (standing rule) | — |
| R10 | Trial clock (90 days) runs out before M5 | start the clock after M0; M1–M3 are days, not weeks; M4/M5 are hours of runtime each | the clock runs out with M3 done: the pipeline is proven, the §5c run moves to a paid few dollars — acceptable |
| R11 | Documentation URLs in this file rot (§0) | product names recorded as search terms | — |

---

## 8. Open decisions for the owner

Each with options and a recommendation. None is taken by this document.

1. **Provision, substitute, or archive** (audit item 5). *Options:* (a) provision under
   `$GCP_PROJECT_ID` on the trial — this plan; (b) local substitute (MinIO + DuckDB or
   similar) — proves nothing about the BigQuery/dbt/Airflow path and still leaves §5c with
   no target; (c) archive the GCP half as reference — the repository's portfolio claim
   becomes "designed", not "ran". **Recommend (a).** Settles D-006 as *keep*.

2. **Spark version on the cloud. TAKEN — (a), 2026-09-02.** *Options were:* (a)
   Serverless runtime 3.0, Spark 4.0.1, `requirements.txt` moved to 4.0.1 after the parity
   test; (b) custom image with 4.1.2; (c) GCE VM with the local install. The M0 parity test
   (2.4) measured 4.0.1 and 4.1.2 **bit-identical** on Phase 4b row 2 — every metric and
   every standard deviation equal to full float64 precision, per fold as well as in the
   mean. (a) is taken and `requirements.txt` is on 4.0.1. (b) needed a measured reason from
   that test and has none. (c) stays rejected on credit exposure. Recorded as **D-011**.

3. **Where the sklearn half of §5c runs.** *Options:* (a) the laptop after the
   category-dtype enabler, model by model (`--only`), `stacking` overnight; (b) a
   short-lived 8-vCPU / 32 GB GCE VM, deleted at the end (price UNVERIFIED — check first);
   (c) the Serverless driver, 16 GB, sklearn on the driver only — a misuse of the service.
   **Recommend (a) first**, with a single `--only lightgbm --sample full` probe as the
   decider; (b) if it OOMs or the encoder probe exceeds an hour.

4. **How far modeling moves to PySpark** (§5). **TAKEN — 2026-09-15, recorded as D-013.**
   *Options were:* (a) the OOF encoder + MLlib arm of every comparison, sklearn champion
   until the memory boundary; (b) full port of `preprocess.py` / `evaluate.py` / the sweep,
   losing three model families and the multi-metric harness. The §5.4 run built the OOF
   encoder and measured it: read on MAE, the OOF row is inside row 2's fold std, and row 3
   beats it on all three metrics. So **D2 stands**, PySpark modeling stays a demonstrated
   baseline, and the §5c MLlib arm drops the corridor (row 3's configuration). (b) was never
   in reach on this evidence.

5. **Terraform state.** *Options:* (a) local state, gitignored — one operator, one
   laptop; (b) GCS backend in the pipeline bucket under `tfstate/` — survives the laptop,
   costs a chicken-and-egg (the bucket must exist before the backend does). **Recommend
   (a) now**, (b) if a second machine ever runs Terraform. The three tracked files leave
   git either way.

6. **When to start the 90-day clock.** *Options:* sign up now, or after M0. **Recommend
   after M0** — the parity test, the submodule fix and the Terraform rewrite spend no
   cloud time and should not spend trial days.

7. **VM size and the `e2-micro` question.** The Always Free `e2-micro` (1 vCPU shared,
   1 GB) is ASSUMED unable to run the five-container compose stack. *Options:* try it
   first because it is free, or start at `e2-standard-2`. **Recommend `e2-standard-2`**
   and stop it by rule; the free tier is a bonus, not a design constraint.

8. **Ingest window.** *Options:* keep 2015-01–2016-12 for the migration; extend later per
   §6.2. **Recommend keep** — the migration's job is reproducing 128,781,646 rows, and
   every later-year risk in §6.2 is a separate decision with its own gate.

---

## Status

- [ ] Plan reviewed by Sagar
- [~] Decisions 1–8 taken; D-006 closed; D2 successor entry added to `notes/decisions.md`.
      **Decisions 2 and 4 are taken** — D-011 (Spark 4.0.1) and **D-013** (2026-09-15, the
      D2 successor: §5.4 read on MAE, D2 stands, the §5c MLlib arm drops the corridor).
      Decisions 1, 3, 5–8 stay open; D-006 stays open on audit item 5.
- [x] M0 — hygiene (tests first; tfstate out of git; `main.tf` rewrite; submodule fix
      upstream + pointer bump; Spark 4.0.1 parity test). **Complete 2026-09-02.** The
      submodule fix landed last, as upstream `305868f`; the pointer moved off `d11219d`,
      both `xfail(strict=True)` guards reported `XPASS(strict)` on the bump, and the marker
      was deleted. Gate: 292 passed, 1 skipped, 0 xfailed, the 9 integration failures
      unchanged — they need M1's provisioned project.
- [x] M1 — provision via Terraform; budget alerts first. **Complete 2026-09-02** — the
      M1 section already recorded this; the box was left unticked.
- [x] M2 — DAGs proven from the laptop; archive fingerprint recorded. **Complete
      2026-09-03.** All 4 ingest and all 4 external-table DAGs ran green from the laptop:
      50 objects, 4.10 GiB, 312,790,342 trip rows across yellow and green. Gate met at
      **0 failed, 301 passed, 1 skipped** — all 7 integration failures flipped. Per-file
      sizes in `notes/gcp-reference.md`. `dbt_prod_restore/` untouched at 204 objects.
- [x] M3 — **re-run 2026-09-12 on the D-012 fix. Complete.** Submodule `25f3186`
      (PR #11), one `dbt build --target prod`, `PASS=21 ERROR=0`. `fact_trips` =
      **307,339,039** rows, 130.81 GiB. Yellow `dist_p99` **18.51**, `dur_p99` 57.75;
      yellow guarded rows 2.70x the voided build. Build $1.56, session $1.66. Gate
      340 passed, 1 skipped, before and after. Full record in the M3 section.
- M3's first run was **VOIDED by D-012 on 2026-09-07.** That rebuild is reproducible and
      counts the wrong rows: the staging dedup key discards 63.09% of yellow trips, and M3's
      own tiebreak biases the survivors short. 128,408,323 is retired. The record below
      stands as history; do not quote its numbers.
  <details><summary>Original M3 record, superseded</summary>

  - [x] M3 — `dbt_prod` rebuilt from the cloud, four ways, all in agreement. **Complete
      2026-09-03.** The planned check against 128,781,646 turned out to target an
      unreproducible number — a missing `ORDER BY` in two staging models let BigQuery pick
      an arbitrary tiebreak winner on every build. Fixed upstream (submodule `3f927a3`);
      new reproducible count is **128,408,323**, checksum-verified identical across two
      `dbt_dev` builds, a `dbt_prod` CLI build, and the `dbt_build_marts` DAG. A
      self-inflicted accidental concurrent build briefly 4×-duplicated `dbt_prod.fact_trips`;
      caught, diagnosed, and repaired with one clean rebuild — full detail in the M3
      section. `fact_trips` measured at 54.56 GiB, refuting the 3.2 assumption of 15-30
      GiB. Gate: **0 failed, 301 passed, 1 skipped** — unchanged from M2, so this proves no
      regression, not new progress. `dbt_dev` dropped and all 9 DAGs re-paused after.
  </details>
- [x] M4 — Serverless smoke inside fold noise; prep from BigQuery matches M3's BigQuery numbers
      **Complete 2026-09-15.** **Smoke:** `m4-smoke-20260915` SUCCEEDED on the 1,355,641-row
      D-012 split, master `dataproc`, 2 executors, `default_parallelism` 8. MAE 0.466188 ±
      0.005356, RMSE 1.201377 ± 0.021579, R² 0.984675 ± 0.000561, equal to the 2026-09-14
      local `local[8]` run to floating-point precision — gate PASS. 4,301.8 s of folds,
      73.8 min RUNNING, $1.12 (DCU $1.05 + shuffle $0.07). Gate 358 passed, 1 skipped.
      The §5c item in M4 stays open on its own Status line below. **Next: §5.3.**
      **Prep half done 2026-09-13:** `--cluster` fixed (`24175b0`), then
      `prep-m4-20260913-full` read 307,339,039 rows and guarded 304,766,876, both exact to
      M3, in 25.6 min for $0.53. **Green cap anomaly:** measured 2026-09-13, both green
      caps IN BAND, RESOLVED-as-method; the owner chose A (keep the Spark caps) — "Open anomaly"
      in `notes/2026-09-06-prep-cloud-baseline.md`. **Baseline set 2026-09-13:**
      `sample_work` (1,828,181 rows) and `prep_stats.json` came down from `ml/prep`, equal
      to the objects; every cap inside Q1's band; temporal 133,629 / holdout 338,911 /
      train 1,355,641; pre-D-012 samples archived — Next item 4 of that note.
- [x] §5.3 OOF encoder (TDD) and §5.4 acceptance run — result recorded in the modeling plan §5b.
      **Run 2026-09-15, local, $0.** `oof_encode.py` (`8f67d17`) and `--oof` (`36cedd5`),
      test first. On the 1,355,641-row D-012 split, Spark 4.0.1, `local[8]`: row 3
      `mllib_gbt_nocorr@work1355k` MAE 0.453134 ± 0.003220; OOF `mllib_gbt_oof@work1355k`
      MAE 0.462342 ± 0.000505, RMSE 1.161502 ± 0.011404, R² 0.985679 ± 0.000304; row 2 MAE
      0.466188 ± 0.005356. On MAE the OOF row is inside row 2's fold std, so the outcome row
      is "at or above row 2"; RMSE and R² read "between row 2 and row 3". The corridor stays
      net-negative in MLlib on all three metrics. **Next: the owner's §5.4 decision** — see
      "The §5.4 result on the D-012 split" in §5.4.
- [~] §5c run scoped per decision 3. **Model list re-derived 2026-09-15**, because D-012
      voided the sweep it came from: `01_run_sweep.py --tag work_d012` ran 14 models on the
      1,355,641-row split in 7,468.0 s, local, $0. Top 4 by RMSE unchanged as a set —
      `catboost` 0.928765, `lightgbm` 0.936259, `stacking` 0.937680, `extra_trees` 0.947680
      — with the first two swapped inside the fold std. Row 1 `lightgbm` MAE 0.331289 ±
      0.001310, 1.41x more accurate and 105x faster than row 2. The MLlib arm drops the
      corridor (D-013). **Still open:** the machine size, because `sample_full` holds
      30,482,494 rows, 2.4x the figure §5c sized against, and it is in the bucket only.
      Next: the `--sample full --only lightgbm` probe (decision 3).
- [x] M5 — Airflow VM on ADC, one DAG end to end, VM stopped. **Complete 2026-09-26.**
      `closeout-airflow-vm`, `e2-standard-2`, `dtc-de-course` attached, no keyfile, the
      test-first override `airflow/docker-compose.vm.yaml` (`18893f6`).
      `nyc_taxi_zone_ingestion_dag` and `create_external_table_taxi_zone` both `success`;
      the object 12,331 B, `tzgoCw==`. 15.11 min RUNNING, about $0.019; `TERMINATED`.
      Record: M5, "The M5 result".
- [ ] M6 — documents reconciled; audit items 5, 6, 8 checked off
