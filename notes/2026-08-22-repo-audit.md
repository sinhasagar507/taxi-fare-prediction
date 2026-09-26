# Repository Audit — 2026-08-22

**Invoke when:** starting new work, or looking for what is still open.

Full-repo audit run 2026-08-22 (session: audit only, no files changed by the audit itself).
Tests at audit time: 258 passed, 1 skipped. Work each item in a **fresh session**; check it
off here when done. The three standing Deferred items in CLAUDE.md (PR,
`project_architecture/` move, `pytest.ini` header) are excluded — they stay parked.

## A. Broken repository state — fix first

- [x] 1. Stale cherry-pick sequencer in `.git` — `git status` says "cherry-pick in
      progress"; the todo picks (`e5f29bd`, `38283a2`) are already on the branch.
      Fix with `git cherry-pick --quit` (NOT `--abort`).
      **Verified clear 2026-09-01:** no `.git/sequencer`, no `.git/CHERRY_PICK_HEAD`,
      `git status` clean. Cleared in an earlier session; the box was never ticked.
- [ ] 2. 14 commits unpushed on `refactor/wire-pipeline` — the whole Phase-4/4b ML
      program exists only locally. Push after clearing item 1.
      **Partly overtaken, still open 2026-09-01.** The branch reached `origin` at some
      point after the audit, so the backlog is no longer 14. It stands at 7, six of them
      from the 2026-09-01 session. **The owner pushes** — that session was explicitly
      instructed not to touch any remote.
- [x] 3. Uncommitted edit in `spark/2026-07-10-fare-prediction-modeling-plan.md` —
      the 2026-08-09 leakage-verification write-up. Commit it.
      **Done before 2026-09-01** — landed as `d7e5ea8`, "docs(ml): measure the sklearn
      target-encoding half instead of asserting it". The box was never ticked.

## B. Vision gaps

- [x] 4. Identity mismatch: repo name + README say **duration**; the locked modeling
      target is **fare** (`fare_capped`); `CASE_STUDY.md` (Jun 3) predates the pivot.
      **Done 2026-08-22:** the CLAUDE.md "Project" section and the README are reoriented
      to fare (title, intro, diagram, `spark/` tree, plus a note explaining the pivot).
      **Remaining:** `CASE_STUDY.md` waits for modeling Phase 5 — see D-005 in
      `notes/decisions.md`. The folder name stays; the README pivot note carries it.
      **Done 2026-09-26:** Phase 5 scored the holdout, D-005 was reopened and closed, and
      the case study is rewritten to fare at `notes/CASE_STUDY.md`.
- [x] 5. GCP target decision missing: no project is provisioned, so Airflow/dbt/
      Terraform/BQ cannot run end-to-end and the §5c cloud run has no target. Record a
      decision: provision a project, build a local substitute, or archive-as-reference.
      *Rider (2026-08-22):* the decision settles the "Deployment split" subsection in
      CLAUDE.md — keep it if the GCP design continues, else move it to
      `notes/gcp-reference.md` and keep only the no-keyfiles auth rule.
      **Done 2026-09-26.** Decided "provision" in `notes/2026-09-02-gcp-cloud-migration-plan.md`
      (2026-09-02) and carried through M5: the Airflow stack ran the zone DAG and its
      external-table DAG on a GCE VM with ADC. The design continues, so the Deployment split
      stays in CLAUDE.md (D-006 closed), and CLAUDE.md's "not provisioned" section is replaced.
- [x] 6. Three workstreams (pipeline docs, ML sweep, dashboard v3) with no priority
      order. Write the order down.
      **Done 2026-09-26.** The order, per the migration plan M6: the migration through M3 →
      modeling §5 encoder + §5c → dashboard v3. The first two are complete (M3 2026-09-12,
      §5c RESCALE 2026-09-26, Phase 5's holdout 2026-09-26, M5 2026-09-26). **Next: the
      dashboard v3, owner-built.**

## C. Plan-of-action gaps

- [ ] 7. Branch scope: `refactor/wire-pipeline` now carries the entire ML program.
      Decide how to split future work into smaller branches.
- [x] 8. Two plans, no single next-step list: CLAUDE.md's phase plan and the spark
      modeling plan don't point at each other. Add cross-links / one "next" pointer.
      **Done 2026-09-26.** One pointer: CLAUDE.md → the migration plan Status → the modeling
      plan Status.

## D. Implementation drawbacks

- [x] 9. `spark/` mixes course notebooks (`03_test.ipynb` … `09_spark_gcs.ipynb`,
      `Demo Spark notebook.ipynb`, `head.csv`, `tmp/`, `lib/` 39 MB, `local_spark/`)
      with the production `spark/ml/` code. Extend the retire list and prune.
      **Done 2026-09-01.** Pruned exactly that list (39 MB freed, all of it `lib/`).
      `spark/local_spark/pyspark_bigquery_debug.py` also left `SCANNED_FILES` in
      `tests/unit/test_credential_decoupling.py`. The CLAUDE.md retire list is rewritten
      to what actually remains — most of the original entries no longer exist on disk.
      **Left in place, deliberately:** `nyc_taxi_duration_prediction.ipynb`,
      `nyc_taxi_duration_prediction_1.ipynb`, `pyspark_r_equivalent_toolkit.ipynb`. The
      audit never named them, and the first two are the pre-pivot duration work the owner
      still has plans for. Decide them separately.
- [x] 10. `migration_backup/` (7.1 GB) sits inside the working tree. Move it outside
      the repo.
      **Done 2026-09-01.** Moved to the repo's sibling
      `/Applications/saggydev/projects_learning/nyc_taxi_migration_backup/` — same disk,
      so the move was a rename. All 204 `fact_trips` parquet files verified in place.
      `spark/ml/00_prep_spark.py` hard-coded `REPO_ROOT / "migration_backup"`, so the
      path became policy in the new `spark/ml/src/paths.py` (TDD, 7 tests): the
      `MIGRATION_BACKUP_DIR` env var wins, else the sibling directory. The `.dockerignore`
      guard in `test_docker_runtime.py` stays — it costs nothing and still catches a
      restore. Docs updated in CLAUDE.md, `notes/gcp-reference.md`, the runbook (since
      renamed `notes/gcp-setup-runbook.md`) and the modeling plan.
- [x] 11. Loose root documents (`2026-05-24-Dashboard-development-plan-v3.md`,
      `project-status-phase5.pdf`, `MIGRATION_RUNBOOK.md`, `CASE_STUDY.md`) have no
      home. Relocate to `notes/`.
      **Three of four done 2026-09-01.** The dashboard plan, the runbook and the status
      PDF moved to `notes/`. `notes/README.md` was the inherited course-notes index and
      indexed none of the project's own documents; it now opens with a **Project
      documents** section that lists them, so "no home" is fixed for the index as well as
      the directory. The handoff note's pointer at the dashboard plan was repathed.
      **`CASE_STUDY.md` stays at the repository root.** D-005 is LOCKED and says to leave
      it untouched until modeling Phase 5 scores the sealed holdout. Moving it is not a
      rewrite, but the entry says *untouched*, and the file's fate — rewrite or delete —
      is decided by that same phase. Move it then, in the change that settles it.
      **Done 2026-09-26:** moved to `notes/CASE_STUDY.md` in the rewrite commit, and
      indexed in `notes/README.md`.
- [x] 12. Known model defect unassigned: design matrix rank 25/27 (dummy trap,
      cond ≈ 4e15), coefficients unstable. Assign it a phase in the modeling plan.
      **Closed 2026-09-01 — the defect was already fixed.** `fc87020` (2026-08-01) added
      `drop="first"` to the scaled variant's OneHotEncoder, three weeks before this audit
      recorded the item. The audit copied the modeling plan's stale "known, untouched"
      line instead of re-measuring. Re-measured on the 612,608-row train split: the
      `scaled` matrix is **rank 23/23**, 24/24 with an intercept, cond **6.97e+03**. The
      `tree` matrix stays rank 24/26 by design — trees split rather than invert, so full
      one-hot costs them nothing. So no phase needed assigning; the plan's Status now
      carries the measurement instead of the stale claim.
      **The transferable lesson:** an audit that copies a status line inherits its age.
      Numbers in a checklist need re-measuring at the moment they are written down.

## Attack order

1. ✅ Items 1–3 (git state) — one short cleanup session. Item 2 is the owner's to finish.
2. ✅ Items 5–6 + 8 — one decision document, the migration plan. Closed 2026-09-26 (M6).
3. ✅ Item 4 — docs reconciliation. `CASE_STUDY.md` waits on D-005.
4. ✅ Items 9–11 — prune + relocate. Done 2026-09-01. `CASE_STUDY.md` held by D-005.
5. ✅ Resume the modeling plan at MLlib row 2 (Phase 4b §5b); item 12 rides along there.
   Done 2026-09-01. Phase 4b is complete; item 12 turned out to be already fixed.

**What is left after 2026-09-01:** items 2, 5, 6, 7, and the `CASE_STUDY.md` half of 11.
Items 5, 6 and 8 are still one decision document, and it is now the front of the queue —
item 5 (the GCP successor) also unblocks D-006 and the §5c cloud run.

## Session log — 2026-08-22

- Audited the repository; wrote this file. Tests green: 258 passed, 1 skipped.
- Trimmed `CLAUDE.md` from 350 lines to ~200, against Anthropic's under-200-line target.
  Moved the GCS/BigQuery layout, the test-structure tree, the E2E smoke steps, and the
  Phase 0–4 history to `notes/gcp-reference.md`.
- Reoriented `CLAUDE.md` and `README.md` from duration to fare (part of item 4).
- Created `notes/decisions.md`, the Decision Register, and put its guard rule in
  `CLAUDE.md`. Eight seed entries: D-001 … D-008.
- Nothing committed. The git-state items (1–3) remain the cleanup session's job.

## Session log — 2026-09-01

Attack-order steps 4 and 5, run end to end in one session at the owner's request (the
per-step review gate was suspended for that session only; nothing was pushed).

- **Item 9** — pruned the course leftovers out of `spark/`; 39 MB freed, all of it the
  vendored GCS connector jar. Rewrote the CLAUDE.md retire list, six of whose eight
  entries no longer existed on disk.
- **Item 10** — moved the 7.1 GB `fact_trips` backup to the repo's sibling
  `nyc_taxi_migration_backup/`. The hard-coded path in `00_prep_spark.py` became policy
  in the new `spark/ml/src/paths.py` (TDD).
- **Item 11** — three of four loose root documents moved to `notes/`, and
  `notes/README.md` grew a Project documents index. `CASE_STUDY.md` stayed put under D-005.
- **Item 12** — closed without work. The defect was fixed on 2026-08-01 by `fc87020`;
  this audit had copied a stale status line rather than re-measuring.
- **Phase 4b row 2** — the MLlib GBT baseline with `od_corridor` target-encoded. It scored
  *worse* than the corridor-dropped ablation, and no smoothing value recovered it. Written
  up in the modeling plan §5b as a negative result with the seven-arm sweep behind it.

Tests: 257 unit passed, 1 skipped. The 9 integration failures are pre-existing — they
need a provisioned GCP project, and fail identically before and after this session.
