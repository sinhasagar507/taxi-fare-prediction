# Decision Register

**Invoke when:** always — before proposing anything. This file is enforced by the guard rule in `CLAUDE.md`.

Settled decisions for this repository. `CLAUDE.md` carries the guard rule that enforces
this file; this file carries the entries.

**Status vocabulary**

- **LOCKED** — a settled decision. Anything that contradicts it stops work until the
  entry is reopened.
- **DEFERRED** — parked work. Do not start it, and do not raise it as a blocker or a
  next step, until the owner asks explicitly.

**Reopening.** A LOCKED entry reopens only on the explicit words "reopen D-NNN" plus a
new reason. A repeated thought is not a new reason. Record every reopening in the entry
itself, with the date and the new reason.

**Format.** Each entry states the Decision, the Why, the Reopen-if condition, and the
Status. The Why is the important field — it is what a future session quotes back.

---

## D-001 — The modeling target is fare, not duration

- **Decision:** the ML model predicts **fare** — target `fare_capped`, the p99-capped
  `fare_amount`. It does not predict trip duration. `trip_duration_min` is a candidate
  *feature*, wired as an on/off ablation, not the target.
- **Why:** the owner confirmed it explicitly ("Its fare prediction") on 2026-07-04. The
  repository folder name (`nyc_taxi_durationprediction`) and the legacy notebooks predate
  the pivot and must not pull the target back.
- **Reopen if:** never expected. The whole `spark/ml/` program assumes this target.
- **Status:** LOCKED (2026-07-04)

## D-002 — The PR into `main` waits for the owner

- **Decision:** the repository owner opens the pull request themselves, on their own
  timing. Compare link when wanted:
  `https://github.com/sinhasagar507/taxi-fare-prediction/compare/main...refactor/wire-pipeline`
- **Why:** nothing in any plan gates on it. Surfacing it as a pending next step reads as
  pressure and misrepresents what the plan requires.
- **Reopen if:** the owner asks for the PR.
- **Status:** DEFERRED (2026-07-30)

## D-003 — `project_architecture/` move stays parked

- **Decision:** leave `project_architecture/` at the repository top level for now. It is
  listed under "Artifacts to retire" in `CLAUDE.md`.
- **Why:** deferred, not forgotten. The owner sets the sequencing.
- **Reopen if:** the owner asks for the move.
- **Status:** DEFERRED (2026-07-30)

## D-004 — The `pytest.ini` header fix stays its own change

- **Decision:** do not fix `[tool:pytest]` → `[pytest]` as part of other work.
- **Why:** the fix makes `testpaths`, `--strict-markers`, and `filterwarnings` take
  effect for the first time. That behavioural change must land alone, so its fallout is
  attributable.
- **Reopen if:** the owner schedules it as its own change.
- **Status:** DEFERRED (2026-07-30)

## D-005 — `CASE_STUDY.md` waits for modeling Phase 5

- **Decision:** leave `CASE_STUDY.md` untouched. Rewrite it to the fare story, or remove
  it, only after **modeling Phase 5** (tune + diagnose) scores the sealed holdout.
- **Why:** the case study's headline number does not exist until the holdout is scored
  once, per the §4a split policy. A rewrite now goes stale the day Phase 5 lands.
- **Reopen if:** modeling Phase 5 lands, or a job-application deadline needs a partial
  write-up sooner. The `ds-writeup` skill generates it when the time comes.
- **Reopened 2026-09-26** by the owner: "reopen D-005 — Phase 5 landed 2026-09-26, its
  Reopen-if. Rewrite, not remove." Phase 5 scored the sealed holdout once (modeling plan
  §6, "The HOLDOUT result"), so the headline number exists.
- **Status:** CLOSED (2026-09-26) — rewritten to the fare story with the `ds-writeup`
  skill and moved to `notes/CASE_STUDY.md`. Every number in it cites a committed note.

## D-006 — The "Deployment split" subsection rides audit item 5

- **Decision:** keep the "Deployment split" subsection in `CLAUDE.md` unchanged until
  audit item 5 records the GCP successor decision. If the GCP design continues, it stays.
  Otherwise it moves to `notes/gcp-reference.md`, and only the no-keyfiles auth rule
  stays in `CLAUDE.md`.
- **Why:** three of its four bullets are safety and build rules that must load
  unconditionally. Moving them early would hide them exactly when they matter. Cutting
  them early would pre-empt a decision that is not made yet.
- **Reopen if:** audit item 5 is decided.
- **Status:** LOCKED (2026-08-22)

## D-007 — Sequential sessions, not worktrees

- **Decision:** run one session per concern, in sequence. Do not create git worktrees for
  parallel work until audit items 1–3 are done.
- **Why:** the stale cherry-pick sequencer lives in `.git`, which every worktree shares.
  Broken repository state leaks into all of them. Fix it once, first.
- **Reopen if:** audit items 1–3 are done and genuinely parallel work appears.
- **Status:** LOCKED (2026-08-22)

## D-008 — Rules live in `CLAUDE.md`, state lives in `notes/`

- **Decision:** `CLAUDE.md` holds only rules that must load in every session. Task state,
  reference material, history, and checklists go to `notes/`, reached by a pointer.
  `CLAUDE.md` must not grow; new state goes to `notes/`.
- **Why:** Anthropic's memory documentation targets under 200 lines per `CLAUDE.md`;
  larger files consume context and reduce adherence. The 2026-08-22 trim took the file
  from 350 lines to about 200 on this principle. A rule must push, because Claude cannot
  fetch a rule it has never seen; state can pull, because a task knows when it needs it.
- **Reopen if:** the official guidance changes.
- **Status:** LOCKED (2026-08-22)

## D-009 — Measure before you assert; label what you could not measure

- **Decision:** when a document, plan, or test depends on a number or on a library's
  behaviour, **measure it at the moment you write it down.** Do not copy a number out of an
  older document. Do not extrapolate across an order of magnitude. Before recording a
  result, test the obvious objection to it. If you cannot measure something, write
  **UNVERIFIED** beside it and state what you tried — never quietly guess, and never
  silently drop the claim either.
- **Why:** this repository has paid for the opposite four times.
  - `d83141b` — acted on a believed statement and kept the raw `trip_distance`, which
    disabled the p99 cap and let one corrupt odometer row produce a $9.29M prediction.
  - §5b's corridor drop — "MLlib has no `TargetEncoder`" was false for the installed
    version and was never checked against it. A whole baseline shipped without a feature it
    could have had.
  - Audit item 12 — the 2026-08-22 audit copied a stale "known, untouched" status line. The
    defect had been fixed three weeks earlier by `fc87020`. A solved problem sat on two open
    lists for a month.
  - `DEFAULT_SMOOTHING` — chosen at 20 from a leakage bound, then measured worst-but-one
    across seven arms. 5 won. The bound was an argument; the sweep was evidence.
  The failure mode is always the same shape: a plausible statement, never checked, that
  later work builds on. Measuring costs minutes. Each of the four cost days.
- **How to apply:** a number in a checklist is re-measured when the checklist is written. A
  library behaviour is verified against the *installed* version, not the documented one. A
  hyperparameter chosen from theory is swept before it is reported. A negative result is
  recorded, not discarded — see the `SELECT *` projection note in `01_mllib_baseline.py`.
- **Reopen if:** never expected.
- **Status:** LOCKED (2026-09-02)

## D-010 — Every note declares when to invoke it

- **Decision:** each project document in `notes/` carries an **`Invoke when:`** line near
  the top, naming the trigger that makes it relevant. `notes/README.md` indexes every
  project document together with that trigger. A session scans the index, then reads only
  the notes whose trigger fires. New notes get the line when they are created.
- **Why:** D-008 puts state in `notes/` and says state is *pulled*, not pushed. A pull only
  works if the puller can tell what to pull without reading everything first. `notes/` now
  holds eight project documents, one of them 819 lines. Reading them all every session
  defeats the point of moving them out of `CLAUDE.md`. A trigger line is the smallest thing
  that makes a pull decidable.
- **How to apply:** write the trigger as a condition, not a description — "before any
  `terraform apply`" beats "about Terraform". If a note has no trigger worth writing, that
  is a signal the note is finished and belongs in history rather than in `notes/`.
- **Reopen if:** the index stops being maintained, at which point the mechanism has failed
  and something simpler should replace it.
- **Status:** LOCKED (2026-09-02)

## D-011 — Spark is pinned to the cloud runtime's version, 4.0.1, on measured parity

- **Decision:** `spark/ml/requirements.txt` pins `pyspark==4.0.1`, the version Dataproc
  Serverless runtime 3.0 ships. The pin tracks the **cloud runtime**, not the newest
  release, so a local run reproduces what the cloud will execute. A custom Dataproc image
  carrying 4.1.2 is rejected, and a GCE VM running Spark is rejected. This takes open
  decision 2 of the GCP cloud migration plan.
- **Why:** measured, not argued. The migration plan 2.4 re-ran the Phase 4b row 2
  configuration — 612,608 rows, 5 folds, seed 42, `maxIter=100`, `maxDepth=5`, smoothing 5
  — on both versions and got **bit-identical** results: `mae_mean` 0.5201744916052814,
  `rmse_mean` 1.4501613318146291, `r2_mean` 0.977753809523173, every standard deviation
  equal too, and the five per-fold figures equal individually. The acceptance condition was
  "inside the ~$0.002 fold noise floor"; there is no difference at all to place inside it.
  - The recorded baseline ran on the host, so the control ran 4.1.2 **in the same
    container** as 4.0.1. Without it, one comparison would have moved the Spark version and
    the environment together and proved nothing about either.
  - The 4.1.2-only worry was specific and was tested directly: `targetType` still defaults
    to `"binary"` on 4.0.1, and `TargetEncoderModel` still copies nominal `ml_attr`
    metadata that `* 1.0` clears. Both guards in `01_mllib_baseline.py` stay mandatory.
  - A custom image buys a version nothing depends on, and costs a build on the
    Apple-silicon/amd64 boundary. A VM bills while it *exists*; Serverless bills per second
    and scales to zero, which is the whole argument on a fixed credit.
- **How to apply:** bump this pin only to follow the runtime, and re-run the 2.4 parity
  test when you do. Timing is **not** evidence here: the two container runs differed by 10%
  in seconds per fold, but the control ran second on a warm laptop, so run order and
  thermal state are not separated and the gap is UNVERIFIED. Metrics closed this, not
  seconds.
- **Reopen if:** Dataproc Serverless moves its default runtime, or a model the project
  actually uses needs a class that 4.0.1 lacks — a measured need, not a newer release.
- **Status:** LOCKED (2026-09-02)

## D-012 — The staging dedup key is the emitted row, not `(vendorid, pickup_datetime)`

- **Decision:** `stg_yellow_taxi_data.sql` and `stg_green_taxi_data.sql` deduplicate on the
  **whole projected row**, not on `(vendorid, pickup_datetime)`. `tripid` becomes a
  surrogate key over every emitted source column, because the two-column key was unique
  only as a side effect of the broken dedup. This **reverses the approach** M3 took while
  keeping the thing M3 wanted — a build that reproduces. It also **voids M3's completion**:
  M3 is un-ticked in the migration plan Status, and its 128,408,323 is retired.
- **Why:** measured 2026-09-06, yellow, 2015-01-01 to 2016-12-31.
  - **270.** That is the number of genuinely duplicated rows in 277,171,036 source rows —
    0.0001%. The full-row key finds 277,170,766 distinct trips.
  - **174,868,364.** That is the number of real trips the current key discards — 63.09%.
    It groups 277.17M rows into 102,302,402 buckets, a mean of 2.69 rows each, which is
    simply NYC's yellow trip rate per vendor per second. `(vendorid, pickup_datetime)`
    names a vendor-second, not a trip. The model's own comment concedes it: 74.4M groups
    "hold rows that disagree on the zone pair", and identical duplicates cannot disagree.
  - The p99 caps prove the bias, with the §4 guards applied:

    | Source | Rows | `dist_p99` | `dur_p99` | `fare_p99` |
    | --- | ---: | ---: | ---: | ---: |
    | Raw, no dedup | 274,556,460 | 18.67 | 58.32 | 52.00 |
    | Local backup — pre-M3, arbitrary winner | 100,357,273 | 18.80 | 58.15 | 52.03 |
    | `dbt_prod` — post-M3, `dropoff ASC` | 99,983,155 | 15.47 | 38.45 | 51.00 |

    An arbitrary winner is an unbiased sample of its group, so the pre-M3 build reproduced
    the raw tail to 0.7%. M3's `order by tpep_dropoff_datetime` makes the survivor the
    **shortest trip** in each group, and the tail collapses. M3 was right about determinism
    and wrong about the statistic. It did not create this defect; it stopped it hiding.
  - `fare_p99` held at 52 across all three row sets because it sits on a mass point, so it
    could not have flagged this. Distance and duration did. The JFK flat-fare explanation
    for that mass point is **INFERRED, not measured.**
- **How to apply:** project first, then take `DISTINCT`. That is deterministic by
  construction, so no `order by` chooses a winner and none can be biased. Projecting first
  also sidesteps the type problem the green model records, where the external table
  declares `INT64` for `trip_type` and `congestion_surcharge` while the parquet stores
  `DOUBLE` — neither column is emitted, so neither reaches the comparison. The dbt project
  is a **submodule**: the edit lands in `github.com/sinhasagar507/ny_taxi_analytics`, then
  the pointer moves. Prepared models and the full evidence are in
  `notes/2026-09-06-prep-cloud-baseline.md`.
- **What this invalidates.** Accept these before starting, because they are the cost:
  - M3's completion, its 128,408,323, and its checksum agreement. All four builds agreed
    with each other and all four dropped the same 63%.
  - The M4 prep run of 2026-09-06 and its caps. `spark/ml/data/prep_stats.json` holds the
    local values meanwhile; they match the raw source to 0.7% and are still provisional.
  - Every modeling result, because `fare_capped` is the target and its cap moves. The §5b
    table does not compare across this line. The sealed holdout partition goes with it.
  - About 44 Looker Studio charts, which read `dbt_prod.fact_trips` directly.
  - The trip counts in `README.md` and `CASE_STUDY.md`. Both understate the pipeline by
    roughly 2.7x on the yellow side — the defect makes the project look smaller than it is.
  - Storage roughly triples, from 54.56 GiB toward 150 GiB, so about $3/month rather than
    $1.02. Recurring, not a one-off.
  - Growth factors, **now measured** on a 2016-01 `dbt_dev` build of the fix
    (PR sinhasagar507/ny_taxi_analytics#11): **yellow 2.65x, green 1.24x.** Green is lower
    volume, so fewer trips collide per vendor-second, as expected. The source-level yellow
    figure over the full window is 2.71x; 2.65x is one month of the built mart, and the two
    agree.
- **Validated 2026-09-07**, before any `dbt_prod` rebuild. `dbt build` on the fix returned
  `PASS=21 WARN=0 ERROR=0 SKIP=0`, and the `unique` test on `tripid` passes on its merits
  rather than by collapsing real trips. The caps return to the raw distribution:

  | Service | Source | Rows | `fare_p99` | `dist_p99` | `dur_p99` |
  | --- | --- | ---: | ---: | ---: | ---: |
  | Yellow | `dbt_prod`, current | 4,032,716 | 52.00 | 15.46 | 35.77 |
  | Yellow | `dbt_dev`, fixed | 10,693,594 | 52.00 | **18.51** | **53.22** |
  | Green | `dbt_prod`, current | 1,156,751 | 43.00 | 13.37 | 57.30 |
  | Green | `dbt_dev`, fixed | 1,437,324 | 44.00 | **13.96** | **69.32** |

  Yellow's distance p99 returns to 18.51 against 18.67 measured on the raw source. That is
  the decision's prediction meeting a build.
- **Rebuilt 2026-09-12.** `dbt_prod` on submodule `25f3186` (PR #11), one
  `dbt build --target prod`, `PASS=21 WARN=0 ERROR=0`. `fact_trips` = 307,339,039 rows
  (yellow 271,905,544, green 35,433,495), 130.81 GiB. Full window, §4 guards: yellow
  270,075,802 guarded, `fare_p99` 52.00, `dist_p99` **18.51**, `dur_p99` **57.75**; green
  34,691,074 guarded, 45.00, 14.15, 59.97. Yellow guarded rows are 2.70x the voided
  99,983,155. Build $1.56, session $1.66, measured from `JOBS_BY_PROJECT`.
- **Reopen if:** a measurement shows the source really does carry duplicate trips that a
  full-row key fails to catch. A suspicion that duplicates exist is not a reason — that
  suspicion is what produced this defect, and 270 rows is what it was worth.
- **Status:** LOCKED (2026-09-07)

## D-013 — D2 stands: the out-of-fold encoder does not rescue the corridor in MLlib

- **Decision:** the successor to D2 (modeling plan §10) and open decision 4 of the GCP
  cloud migration plan. The §5.4 acceptance is read on **MAE**, as §5.4 states it, and its
  outcome row is **"MAE at or above row 2"**: the uncross-fitted `TargetEncoder` was not the
  mechanism behind the corridor's net-negative effect in MLlib. D2 stands — preprocessing,
  the CV harness and the sweep stay scikit-learn, and PySpark modeling stays a demonstrated
  baseline, not the champion track. The §5c MLlib arm uses **row 3's configuration**, the
  corridor dropped (`--drop-corridor`).
- **Why:** measured 2026-09-15 on the 1,355,641-row D-012 train split, 5 folds, seed 42,
  GBT `maxIter=100` `maxDepth=5`, smoothing 5, Spark 4.0.1, `local[8]`, the same folds for
  all three rows (migration plan §5.4, "The §5.4 result on the D-012 split"):

  | Row | `od_corridor` | MAE | RMSE | R² |
  | --- | --- | ---: | ---: | ---: |
  | 2, `mllib_gbt@work1355k` | `TargetEncoder`, not cross-fitted | 0.466188 ± 0.005356 | 1.201377 ± 0.021579 | 0.984675 ± 0.000561 |
  | OOF, `mllib_gbt_oof@work1355k` | cross-fitted out-of-fold | 0.462342 ± 0.000505 | 1.161502 ± 0.011404 | 0.985679 ± 0.000304 |
  | 3, `mllib_gbt_nocorr@work1355k` | dropped | 0.453134 ± 0.003220 | 1.138785 ± 0.006900 | 0.986234 ± 0.000206 |

  - The OOF MAE is 0.003846 below row 2, inside row 2's fold std of 0.005356. Under the
    owner's rule, a difference inside the compared run's fold std is no difference.
  - Row 3 beats the OOF row on all three metrics, by 2.7x to 3.3x row 3's fold std, and on
    MAE in all 5 folds. So the corridor feature stays net-negative in MLlib even when
    cross-fitted, which is why the §5c MLlib arm drops it.
  - The leakage defence was measured, not assumed: the encoder's tests fail on two leaky
    mutants, and on real data 0 of 1,084,270 train-half rows read back their own fare.
- **Not taken:** reading §5.4 on RMSE and R², where the OOF row sits materially between
  rows 2 and 3 (the "partial" row). Its consequence, §5c with the OOF column, would run an
  arm that row 3 beats on every metric.
- **Reopen if:** a measurement shows the OOF corridor column beating row 3 on MAE by more
  than row 3's fold std — for example on `sample_full`, or with a different booster. A
  belief that more data will help is not a reason; measure it.
- **Status:** LOCKED (2026-09-15)

## D-014 — Phase 6, neural nets, waits until after the end of the project

- **Decision:** modeling Phase 6 — the PyTorch model with corridor embeddings — is out of
  the project. The project ends at the tuned tree champion, scored once on the holdout and
  the temporal test set, with M5 done, the public documents true, and nothing billing.
- **Why:** the owner chose the fastest path to the end on 2026-09-16. Phase 6 is the
  largest single block of the remaining work, and nothing in the finish line depends on it.
  The decisions table is in the modeling plan §6, "The finish plan".
- **Reopen if:** the owner asks for Phase 6.
- **Status:** DEFERRED (2026-09-16)
