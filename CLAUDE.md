# CLAUDE.md

@~/.claude/global_rules.md

Guidance for AI agents (and humans) working in this repository.

## Working Agreement — one step at a time, reviewed before the next

**Stop after every step and wait for the repository owner to review it.** A "step" is one
reviewable unit of work: one file written or edited, one script run, one commit, one
analysis. Not a whole phase, not a batch of related edits.

After each step, report what changed and what the next step would be — then **stop and
end the turn.** Do not chain the next step onto the same turn, and do not treat silence,
a general goal, or an approved plan as approval for the step after the current one.
Approval is per step and is not durable: "yes, do that" applies to the step just proposed,
never to the rest of the sequence.

This is deliberate friction. The owner is learning this stack and reviews each output as
it lands; work that runs ahead has to be unpicked, which costs more than it saves. It
applies even when the remaining steps look obvious or mechanical.

Read-only investigation — searching, reading files, `git status`, running the test suite —
does not need a gate. Gate anything that **writes, commits, pushes, deletes, or spends
real time**.

### Show the changes every turn

**Any turn that changed a file ends by showing what changed, for review.** Not a summary
of the intent — the actual diff. Run `git diff` (add `--staged` if anything is staged, and
show new files explicitly since an untracked file has no diff) and put the result in the
reply, alongside a one-line note per file saying why it changed.

Small diffs go inline in full. For a large diff, lead with `git diff --stat`, show the
substantive hunks, and say plainly which files were summarized rather than shown. Never
report a file as changed without showing it, and never let a change ride along unmentioned
because it was incidental to the main edit.

If a turn changed nothing, say so — silence is not the same as "no changes".

## Project: NYC Taxi Fare Prediction

End-to-end data engineering pipeline (DataTalksClub DE Zoomcamp): TLC trip data → GCS →
BigQuery external tables → dbt marts → fare model (`spark/ml/`) + Looker Studio.
The modeling target is **fare** (`fare_capped`), locked (D-001). The repository folder name
still says "duration" and stays that way; the README carries a note explaining the pivot.
The case study, rewritten to fare after Phase 5, is `notes/CASE_STUDY.md` (D-005 closed).

## GCP — `dtc-de-project-506916`, live on the free-trial credit

**The project is provisioned and billed.** Bucket `primary-data-dtc-506916`; layout in
**`notes/gcp-reference.md`**. Locally, auth is the keyfile `secrets/gcp-credentials.json`
(SA `dtc-de-course@…`) via `GOOGLE_APPLICATION_CREDENTIALS` — the gcloud user account has
no bucket access. On a VM, attach that SA and use `airflow/docker-compose.vm.yaml`.
**Nothing may bill idle:** `gcloud compute instances list` shows only TERMINATED VMs and no
Dataproc batch is RUNNING/PENDING, before and after any cloud work. Read the migration
plan before spending. `spark/ml/` still runs locally from the backup.

The dbt project is a **git submodule** at `dbt/ny_taxi_analytics` (remote:
github.com/sinhasagar507/ny_taxi_analytics, authoritative — the submodule pins a commit).
Verified stack: dbt-core 1.11 + dbt-bigquery 1.11 on Python 3.12.

## Directory layout

The Phase-3 restructure is done. Top level: `terraform/` (infra — import before apply),
`airflow/` (DAGs + compose stack), `dbt/` (submodule + profiles.yml + requirements),
`docker/dev/` (dev container), `spark/` (batch + ML, `spark/ml/` is the fare model),
`bigquery/` (two exported CSVs; the SQL reference it once held is gone), `tests/`,
`secrets/` (gitignored), `notes/` (docs),
`.github/workflows/dbt.yml` (CI).

**Artifacts to retire** (learning leftovers, not pipeline components). Most of the
original list is already gone — `03_data_warehouse_bigquery/`, `04_analytics_engineering/`,
`docker_nana_tutorial/`, `gcs_storage/`, `spark_data/`, `google-cloud-sdk/` (vendored SDK)
and `bigquery/queries/*.sql` no longer exist. Audit item 9 (2026-09-01) pruned the course
leftovers out of `spark/`: the `03_test`…`09_spark_gcs` notebooks, `Demo Spark notebook`,
`head.csv`, `tmp/`, `lib/` (39 MB GCS connector jar) and `local_spark/`. What is left:

- `project_architecture/` — parked by **D-003**. Do not raise it as a next step.
- `spark/nyc_taxi_duration_prediction*.ipynb` and `spark/pyspark_r_equivalent_toolkit.ipynb`
  — pre-pivot duration/XGBoost work and an R-equivalence toolkit. Kept on purpose; the
  audit never listed them, and the owner decides their fate.

## Environments — dev container vs Airflow vs host venv

Three environments with non-overlapping jobs. Two Docker stacks; they are never merged.

- **Dev container (`docker/dev/`) — the default for anything that needs a library.**
  One image for the ML stack (pandas/scikit-learn/XGBoost/LightGBM/CatBoost/SHAP/Optuna,
  later torch), PySpark batch prep, the pytest suite, the dbt CLI, and JupyterLab. ML
  libraries are installed **here only — never in the host `.venv`.** Run from the repo root:

  ```bash
  docker compose -f docker/dev/docker-compose.yml run --rm dev pytest tests/
  docker compose -f docker/dev/docker-compose.yml run --rm dev bash
  docker compose -f docker/dev/docker-compose.yml up   # JupyterLab, 127.0.0.1:8888
  ```

  The repo is bind-mounted at `/workspace` and nothing is copied into the image, so host
  edits are live in the container. New deps go in `spark/ml/requirements.txt` (libraries)
  or `docker/dev/requirements-dev.txt` (harness) — never on the host. dbt lives in an
  isolated `/opt/dbt-venv` inside this image, symlinked to `/usr/local/bin/dbt`.
- **Airflow (`airflow/docker-compose.yaml` + `dockerfile`) — a separate stack, untouched
  by the above.** Anything a DAG imports at runtime goes in the Airflow image, never on
  the host. It builds its own isolated `/opt/dbt-venv`. Do not merge the two stacks or
  share images between them.
- **Host `.venv` (repo root, Python 3.13) — still present, no longer the default.** Fast
  path for `.venv/bin/pytest tests/`, `.venv/bin/dbt`, and ad-hoc GCP client scripts.
  Invoke it explicitly — never rely on shell activation, never use system python/pip,
  don't create additional venvs. It deliberately lacks xgboost/lightgbm/catboost; tests
  needing them use `pytest.importorskip` and skip cleanly here.

### Deployment split (target: GCP, single GCE VM running the compose stack)

- **Deploys:** Airflow image + DAGs (ingest → external tables → `dbt_build_marts`),
  dbt project via the submodule pin, Terraform-managed infra (bucket, BQ datasets).
- **Stays local:** notebooks, EDA artifacts, ML experiments, and the 7.1 GB
  `fact_trips` backup — which since audit item 10 (2026-09-01) lives **outside the
  repository**, in the sibling `nyc_taxi_migration_backup/`. `spark/ml/src/paths.py`
  resolves it; set `MIGRATION_BACKUP_DIR` to point elsewhere. Tests run locally + in
  CI, not on the VM.
- **Auth:** keyfile via `GOOGLE_APPLICATION_CREDENTIALS` locally/CI; on the VM use an
  attached service account (ADC) via `airflow/docker-compose.vm.yaml` — no keyfiles
  inside images or git, ever.
- **Prod images are built on the VM itself** (decided 2026-07-10): clone the repo on the
  VM and `docker compose up --build` there. This sidesteps the Apple-Silicon/amd64
  mismatch entirely — local Mac builds are dev-only and must never be pushed to the VM.

## Testing

Each development phase must pass its verification gate before the next phase begins.
Run `pytest tests/` from the repo root — **always with the explicit `tests/` path.**
A bare `pytest` also collects the legacy `airflow/tests/` stubs and dies with import
errors (`pytest.ini` declares its options under `[tool:pytest]`, the setup.cfg section
name, which pytest ignores in a `pytest.ini`, so `testpaths` never takes effect).

Unit tests run without credentials; integration tests auto-skip if credentials are
unavailable. Either environment works:

```bash
.venv/bin/pytest tests/                                                  # host
docker compose -f docker/dev/docker-compose.yml run --rm dev pytest tests/   # container
```

ML tests that need xgboost/lightgbm/catboost skip on the host (container-only libs)
and run in the container.

### Tiers

| Tier | Requires | Run command | Gate for |
| --- | --- | --- | --- |
| **Unit** | nothing | `pytest tests/unit/` | every commit |
| **Integration** | `GOOGLE_APPLICATION_CREDENTIALS` | `pytest tests/integration/` | Phase 1, Phase 3, Phase 4 |
| **dbt** | credentials + dbt Core | `dbt compile --project-dir dbt/ny_taxi_analytics --profiles-dir dbt` | Phase 1 |
| **E2E** | `docker compose up` | trigger DAGs in Airflow UI, check task states | Phase 1, Phase 4 |

Test layout + E2E smoke steps: `notes/gcp-reference.md`. `airflow/tests/` holds legacy
TDD stubs needing local Airflow — not part of the standard run; treat as documentation.

## Development Plan — sequenced for safety

Each phase ends with a test-suite verification gate and its own commit. Do not mix phases.

Phases 0–4 are **done** (each verified `pytest tests/` green); full detail in
`notes/gcp-reference.md`:

- ✅ Phase 0 — checkpoint branch + `.gitignore` hardening.
- ✅ Phase 1 — config reconciled with live infra; DAGs chained; dbt + CI/CD wired.
- ✅ Phase 2 — pruned unreferenced learning artifacts.
- ✅ Phase 3 — restructure: `05_batch_processing/` → `spark/`, mechanical moves only.
- ✅ Phase 4 — DRY'd config to env vars; stable credentials path; parametrized ingest window.
- ✅ Phase 5 — docs reconciled to the final structure and the live project (M6, 2026-09-26).

## Current work + the Decision Register

**The one next pointer:** this file → the Status list of
`notes/2026-09-02-gcp-cloud-migration-plan.md` → the modeling plan's Status.

**Notes are pulled, not read wholesale (D-010).** Every project document carries an
`Invoke when:` line, and `notes/README.md` indexes all of them by trigger. Scan that index;
read only what fires. New notes get the line when they are created.

**Settled and parked decisions** live in `notes/decisions.md`, each with its reason.
**Enforce it.** When the user proposes anything that contradicts a **LOCKED** entry, or
starts a **DEFERRED** item without an explicit ask, **stop before acting** — even
mid-task. Quote that entry's Decision and Why, then ask one question: "Reopen D-NNN?"
Proceed only on the explicit words "reopen D-NNN" plus a new reason, and record the
reopening in the entry. A repeated thought is not a new reason. Never raise a DEFERRED
item as a blocker or a next step.

## Commit Conventions

- **Never add `Co-Authored-By` trailers to commit messages.** The repository owner does
  not want AI model names to appear as contributors or co-authors on GitHub. Commit solely
  under the git user identity configured in the repo (`Sagar Sinha`). This applies to all
  commits on all branches, including automated or agent-driven commits.

## Conventions & Gotchas
- **Single source of truth** for project/bucket = Airflow `docker-compose.yaml` env vars
  (`GCP_PROJECT_ID`, `GCP_GCS_BUCKET`); DAGs read them via `os.environ`. All ingest *and*
  external-table DAGs now follow this — keep new DAGs on env vars too.
- dbt prod dataset is **`dbt_prod`**, not `dbt_production`. Keep PySpark in sync.
- dbt runs via **dbt Core** locally (no dbt Cloud). Three execution contexts share one
  `dbt/profiles.yml` (profile `default`, keyed off `GOOGLE_APPLICATION_CREDENTIALS` so no
  secret is committed): local CLI, Airflow (`dbt_build_marts` DAG → isolated
  `/opt/dbt-venv` built in the dockerfile, target `prod`), and CI (`.github/workflows/dbt.yml`,
  target `ci` → `dbt_ci` dataset). Local run:
  `GOOGLE_APPLICATION_CREDENTIALS=secrets/<your-keyfile>.json dbt build --project-dir dbt/ny_taxi_analytics --profiles-dir dbt`.
  CI needs the `GCP_SA_KEY` repo secret. dbt artifacts (`target/`, `dbt_packages/`) land
  inside the submodule and are gitignored there. Edit dbt models in the **remote repo**,
  then bump the submodule pointer — don't edit inside the submodule here.
- Taxi-zone data reaches dbt via a **seed** (`seeds/taxi_zone_lookup.csv` →
  `ref('taxi_zone_lookup')`), NOT `taxi_zone_external_table`. The zone ingest +
  external-table DAGs are a parallel path dbt does not depend on.
- **Never `terraform apply` against the live resources without `terraform import` first.**
  TF state is empty, so an apply would try to recreate the existing bucket/datasets and
  conflict. Treat Terraform as reference until state is reconciled.
- The Google Cloud SDK is vendored at `google-cloud-sdk/`; `gsutil` needs Python ≤3.12,
  but the BigQuery/Storage **client libraries** work fine under the repo `.venv` (3.13).
- The Airflow **dockerfile no longer installs gcloud** — the old step pulled a *macOS*
  tarball (`…darwin-arm.tar.gz`) into a Linux image (would fail to build) and no DAG uses
  the gcloud CLI (they use the Python client libs + Airflow operators).
