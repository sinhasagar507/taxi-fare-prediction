# Notes

This directory holds the repository's own working notes.

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
