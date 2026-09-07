# Looker Studio reconnect runbook: pointing the dashboard at a new GCP project

**Invoke when:** the Looker Studio dashboard (`dtc-de_DEProject_DataEngineeringReport`)
needs to be repointed at a different GCP project or BigQuery dataset — including finishing
the reconnect to `dtc-de-project-506916` if it is still open, or any future account/project
migration. Owner-only; nothing here needs Claude's execution, since there is no Looker
Studio API for this and the browser-automation extension was not connected when this was
written (see the *Tooling checked* note at the end).

## Why this exists

The dashboard's two data sources — `fact_trips` (reusable, ~44 charts) and
`feature_correlations_v` (embedded, 1 chart) — were built against the old, dead project
(`dtc-de-project-492321`). The pipeline's actual `dbt_prod.fact_trips` now lives in
`dtc-de-project-506916` (see the GCP cloud migration plan, M3). This note is the concrete
click-by-click path from one to the other, and the two traps that make it non-obvious.

## Trap 1 — Reusable vs. Embedded data sources have different sharing rules

Open the report's **Resource → Manage added data sources**. Two source types appear:

- **Embedded** — lives inside the report. Anyone with edit access to the *report* gets
  edit access to it automatically. Its EDIT link is always active for a report editor.
- **Reusable** — an independent asset with its own sharing list. Sharing the report does
  **not** grant edit access to it. This is `fact_trips` (`ds0`) here. If you can edit the
  report but the EDIT link on a Reusable row is greyed out, this is why — it is not a bug.

**Fix:** the account that owns/edits the Reusable data source must share *the data source
itself* (not the report) with Can-edit access:

1. Sign into `lookerstudio.google.com` as the account that owns `ds0` (the *older* account
   that built this report — check the report's own Share dialog to confirm which one).
2. Click **Data sources** in the top nav (the homepage list, not inside the report).
3. Find `fact_trips`, open its **⋮** menu → **Share** → add the new account → **Can edit**.

Once that lands, an editor can also open the data source's **Data credentials** and click
**Make me the owner** — this transfers who Looker Studio authenticates as when *report
viewers* (not editors) load the charts. Ownership can move between editors more than once;
it is not a one-way lock, so this step is reversible from the new account back to the old
one if needed, as long as the old account is still listed as an editor.

## Trap 2 — the project picker is scoped to whoever is *signed into the browser*, not to Data Credentials

"Data credentials" (who Looker Studio authenticates as for viewers) and "which Google
account is currently browsing the connection wizard" are two separate things. The
**Edit Connection** project picker (including "Enter Project Id manually") always searches
under the second one. Becoming the credential owner does **not** let the *currently signed
in* browser session see a project it has no IAM access to.

**Fix:** before clicking Edit Connection, switch the browser's active Google account
(avatar, top right → Switch account) to the one that actually owns the target GCP project
(here, `saggysimmba@gmail.com`, which owns `dtc-de-project-506916`). Only then will that
project resolve, whether typed manually or picked from the list.

## The full sequence, in order

1. Share `ds0` (`fact_trips`) with Can-edit access to the new account (Trap 1).
2. Use **Data credentials → Make me the owner**, run from the new account, so report
   viewers authenticate as the account that owns the new project.
3. Switch the browser's signed-in Google account to the new account (Trap 2).
4. Open `ds0` → **Edit Connection** → select project `dtc-de-project-506916` → dataset
   `dbt_prod` → table `fact_trips`.
5. Click **Reconnect**. Looker Studio shows an **Apply Connection Changes?** dialog listing
   what changed before it commits — read it:
   - A **New fields** / **Missing fields** section means an actual schema mismatch. There
     should be none here, since the new table comes from the same dbt model.
   - A **Changed Semantic Configuration** note (seen once, on `dropoff_borough`) is benign:
     it means Looker's auto-detected semantic/geo classification for that field changed,
     not that the field itself is missing or renamed. Field renames, calculated fields,
     and chart bindings are explicitly *not* affected by reconnecting, per Google's own
     documentation — only raw schema changes and semantic-type flags are.
6. Click **Apply**.
7. Spot-check the report in View mode across several pages, not just one.
8. Remove any throwaway test data source created while diagnosing this (e.g. an `ds48`created
   via "Add a data source" instead of editing `ds0` in place — it has 0 charts and is
   redundant once `ds0` itself is reconnected).
9. Handle `feature_correlations_v` (`ds42`) separately — **it is not a dbt model.** It is a
   hand-built BigQuery view that, as of this writing, exists only in the old project. It
   needs its defining SQL recreated as a view in `dtc-de-project-506916` before repeating
   steps 3–7 on it.
10. Only after both data sources are verified: it becomes safe to disable billing on, or
    retire, the old project (`dtc-de-project-492321`). Not before — the old project is the
    rollback path (step 11) until then, and `feature_correlations_v` likely still depends
    on it directly.

## Rollback, at any point

Re-open **Edit Connection** on the affected data source and point Project/Dataset/Table
back at the old ones. This only works as long as the old table has not been deleted and
the old project's billing has not been disabled — see step 10.

## Status as of 2026-09-04 (update this section as you go)

- [x] `ds0` shared with edit access to the new account; new account made credential owner.
- [ ] `ds0` reconnected to `dtc-de-project-506916.dbt_prod.fact_trips` and verified across
      all report pages.
- [ ] Redundant test data source (`ds48`) removed.
- [ ] `feature_correlations_v` (`ds42`) recreated in the new project and reconnected.
- [ ] Old project (`dtc-de-project-492321`) billing disabled / project retired.
- [ ] `notes/2026-09-02-gcp-cloud-migration-plan.md` line 999 ("Reconnect Looker Studio to
      `dbt_prod`") ticked off once all of the above is done.

## Tooling checked, for next time

No Looker Studio API covers editing an existing report's data source connection — the
public Looker Studio Linking API only creates new linked reports, so this cannot be
automated from here. The Claude in Chrome browser extension, if connected, could navigate
and read the live report to speed up future verification, but was not connected when this
runbook was written. Computer-use (OS-level screen control) explicitly cannot click or type
inside a browser window; it defers to Claude in Chrome for that.
