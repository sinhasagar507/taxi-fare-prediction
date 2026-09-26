# Notes

This directory holds two unrelated sets of documents. The **project documents**
below are this repository's own working notes. Everything under *Course notes* is
the inherited DataTalksClub DE-Zoomcamp lesson material.

## Project documents

Each note carries an **`Invoke when:`** line at its top, per **D-010**. Scan the triggers
below; read only the notes whose trigger fires. Do not read them all.

| Document | Invoke when |
| --- | --- |
| [Case study](CASE_STUDY.md) | Writing the project up for a resume, portfolio or interview, or checking a public claim about it. |
| [Decision Register](decisions.md) | **Always.** Before proposing anything. A LOCKED entry stops contradicting work; a DEFERRED one is never a next step. |
| [GCP cloud migration plan](2026-09-02-gcp-cloud-migration-plan.md) | Any cloud, Terraform, Dataproc, BigQuery or cost question. Before spending anything. Its Status list is the current next-step pointer. |
| [Repository audit, 2026-08-22](2026-08-22-repo-audit.md) | Starting new work, or looking for what is still open. |
| [Cap divergence root cause, 2026-09-06](2026-09-06-prep-cloud-baseline.md) | **Before rebuilding `dbt_prod`, or trusting any row count or p99 cap from it.** The staging dedup discards 62.9% of yellow trips and biases the survivors short. Also any question about the caps, `tripid`, or the M4 prep run. |
| [Cloud cost baseline, 2026-09-04](2026-09-04-cloud-cost-baseline.md) | Any cost question, any optimisation of the BigQuery or dbt workflow, or writing that work up. It is the **before** number and the method to re-measure the after. |
| [GCP reference](gcp-reference.md) | You need the bucket or dataset layout, the test layout, or the E2E smoke steps. |
| [GCP project setup runbook](gcp-setup-runbook.md) | The owner is provisioning a project, or credentials are missing. |
| [Dashboard development plan v3](2026-05-24-Dashboard-development-plan-v3.md) | Working on the Looker Studio dashboard. |
| [Looker Studio reconnect runbook](2026-09-04-looker-studio-reconnect-runbook.md) | Repointing the dashboard at a different GCP project or BigQuery dataset — including finishing the reconnect to `dtc-de-project-506916`. |
| `project-status-phase5.pdf` | A point-in-time status export. **Gitignored on purpose**, so it exists only on the owner's machine; a fresh clone will not have it. |

The modeling plan lives with the code it describes, at
`spark/2026-07-10-fare-prediction-modeling-plan.md`, next to its companion
`spark/2026-07-04-ml-handoff-context.md`. **Invoke when:** any question about the fare
model, the feature contract, the sweep, or the sealed holdout.

## Course notes

Below you will find links to the notes for each lesson.

* [Lesson 1: Introduction to Data Engineering](1_intro.md)
* [Lesson 2: Data Ingestion](2_data_ingestion.md)
* [Lesson 3: Data Warehouse](3_data_warehouse.md)
* [Lesson 4: Analytics Engineering](4_analytics.md)
* [Lesson 5: Batch Processing](5_batch_processing.md)
* [Extra: Preparing data for Spark](extra1_preparing_data.md)
* [Lesson 6: Streaming](6_streaming.md)

Additionally, the following gists with cheatsheets are available.

* [Virtualization and containerization](https://gist.github.com/ziritrion/1842c8a4c4851602a8733bba19ab6050)
* [Python environment management](https://gist.github.com/ziritrion/8024025672ea92b8bdeb320d6015aa0d)
* [Git cheatsheet](https://gist.github.com/ziritrion/d73ca65bf4d19c79ca842a55853cb962)
* [Create a VM instance for the DE zoomcamp](https://gist.github.com/ziritrion/3214aa570e15ae09bf72c4587cb9d686)