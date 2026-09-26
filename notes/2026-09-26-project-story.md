# The Project Story: What Sets This Pipeline Apart

**Invoke when:** writing the LinkedIn post or any public story about the project, or
answering "how is this different from the other NYC taxi projects?". Numbers come from
`CASE_STUDY.md` and the notes it cites; the field survey was run on 2026-09-26 and is
summarized in section 6. Re-check a number there before you publish it (D-009).

This is the story of a fare model for New York City taxi trips, and of the data platform
built underneath it. Public trip records flow from the city's open data site into Google
Cloud Storage, BigQuery and dbt, then out to a LightGBM model. It began as a course
capstone. It ended as something the other public versions of this project do not do.

---

## 1. It started like everyone else's

Hundreds of people have built this pipeline. The city publishes every taxi trip, and a
popular data engineering course uses those files as its teaching set. The shape is always
the same: download the files, land them in a cloud bucket, load them into a warehouse,
model them with dbt, schedule it with Airflow, and draw a dashboard.

I built that shape too. Two years of yellow and green trips, 2015 to 2016: 50 files and
312,790,342 trip rows in the bucket, and 307,339,039 rows in the final dbt fact table.

At that point my project was not different from the others. The course grades exactly that
shape, so the shape is the entry ticket, not the achievement.

## 2. The question changed

The first model predicted trip duration. I moved it to fare, because a fare is what a
rider, a dispatcher or a pricing analyst actually wants to know before the trip starts.
That decision also changed the standard of proof: a fare estimate is judged in cents.

## 3. The number that did not add up

Before I trusted the data, I compared the warehouse against the raw files. The 99th
percentile cap on yellow trip distance came out short of the raw source. A small gap, easy
to explain away.

It was not small. The staging models removed "duplicates" keyed on vendor and pickup
second. That key does not name a trip; it names a busy second in Manhattan. It discarded
**63.09%** of real yellow trips. Only **270** rows in 277,171,036 were true duplicates. The
trips that survived were biased short, so every model trained on them learned the wrong
city.

Fixing it meant a full rebuild of the warehouse and re-running every model result. After
the fix, the yellow distance cap came back to 18.51 miles, against 18.67 in the raw source.

The lesson I took from it: write a number down only after you measure it. Four defects in
this project came from statements that sounded right and were never checked.

## 4. Judging the model honestly

A model that picks its winner on the same rows used to judge it will always look better
than it is. So before comparing any model, I set 20% of the rows aside and sealed them. A
guard in the code refuses to score them a second time. I also held back the last two months
as a temporal set, to see how the model does on trips from the future.

Fourteen models ran on identical folds, from linear models to stacking. LightGBM won after
tuning, ahead of CatBoost by more than CatBoost's own fold-to-fold spread, and it fits 4.7x
faster.

The most dangerous feature was the average fare of each pickup-to-dropoff zone pair. Done
naively, a trip sees its own fare inside its feature and the score becomes a lie. I computed
it out-of-fold, and wrote a test that fails on two deliberately leaky versions of the
encoder, so that mistake cannot come back quietly.

Scored once on the sealed rows:

| Set | Rows | MAE |
|---|---:|---:|
| Random holdout, 2015-01 to 2016-10 | 338,911 | $0.311 |
| Temporal set, 2016-11 to 2016-12 | 133,629 | $0.316 |

The model did about half a cent worse on the future months than on its own period. The
cross-validation estimate was not optimistic.

I also ran Spark MLlib against LightGBM on the same 22.6 million rows. LightGBM's error was
1.53x lower. Spark kept the job it is good at, data preparation, and lost the modeling job.

## 5. Running it like it costs money

It does cost money, so I measured it. The first 30 days of cloud work cost about **$20.7**,
and 99% of that was BigQuery queries. That number is the "before" for any saving I claim
later. The full-scale Spark-versus-LightGBM comparison cost $2.97.

One cloud VM once ran 204 minutes for a 43-minute job. Every VM since is created with a hard
run limit, and nothing is allowed to bill while idle.

Security got the same treatment. A leaked key was purged from the git history. The Airflow
stack then ran on a cloud VM with no keyfile at all, using an attached service account: the
zone ingest DAG and its external-table DAG both succeeded, in 15 minutes of VM time for
about two cents.

dbt runs in three places from one profile: my laptop, Airflow, and GitHub Actions. CI builds
into its own `dbt_ci` dataset, so a pull request can never write to production.

Every settled choice sits in a decision register with its reason, so a question answered
once stays answered.

## 6. Then I checked what everyone else built

I surveyed 24 public NYC taxi pipeline repositories on GitHub, reading each README, file
tree, CI workflow and Terraform file, plus the Kaggle fare and duration competitions, an
arXiv paper, and the fare-prediction repositories built on the same city data. Most of the
pipeline repositories have 0 to 12 stars, so this shows common practice, not a ranking.

**What almost everyone has** — and so what does not set this project apart:
- The same pipeline shape, run from docker-compose.
- Terraform copied from the course template, with local state.
- Service-account keyfiles for Terraform, dbt and Airflow.
- A data window of one to three months.
- No CI in about 16 of the 24, and no model at all in the Google Cloud capstones.

**What none of the 24 had:**
1. A fare model trained on BigQuery dbt marts. The one fare pipeline found trains on a
   Kaggle CSV with a random split.
2. Measured cloud cost. Others limit scope "to save cost"; nobody reports dollars spent.
3. A dbt CI target that builds into its own dataset.
4. A pipeline run on a VM with an attached service account instead of a keyfile.
5. A data-quality defect of this size found, measured and fixed. Two repositories document
   a finding; none found one that removed most of a service's trips.
6. Spark MLlib and a gradient-boosting library compared on the same tens of millions of
   rows.

**What at most two others had:**
- A sealed holdout or a time-based evaluation. The two that have one forecast demand. No
  fare project found tests on later months.
- A leakage test. None of them encodes zone pairs out-of-fold.
- A decision record.
- A dev container: none.

## 7. The honest caveat

One feature needs a straight answer before the $0.31 goes on a headline.

`trip_duration_min` is the recorded duration of the trip. The modeling plan treats it as
the estimate a booking app would show, but the value in the data is the real one, known
only after the trip ends. Without it, the error rises by $0.91, to about $1.23.

The published fare projects that use the same feature report similar numbers: one gets an
MAE of $0.27 and an R² of 0.995 on the same kind of data. So the $0.31 is in good company,
but it is the company of models that know how long the trip took.

Three honest ways to tell it:
- Replace the feature with a booking-time duration estimate, such as the out-of-fold median
  duration for each zone pair and hour, and score it on a fresh sealed split.
- Keep the model and describe it as "fare given the trip", with both ablation results shown.
- Lead with the $1.23 result, reported with RMSE on the uncapped fare so it lines up with
  the Kaggle numbers (strong entries there sit near $2.85 RMSE).

Whichever is chosen, report MAE and RMSE on the uncapped fare too. The 99th-percentile cap
makes both look better than the raw fare would.

## 8. What the story is really about

The pipeline is the part everyone has. The story is what happened on top of it:

- A silent defect that threw away most of the yellow trips, found by distrusting a small gap.
- A model judged once, on sealed rows and on future months.
- Every cloud dollar measured, and no key left on a cloud machine.
- Every settled decision written down with its reason.

Still open: the dashboard on the corrected tables, a Terraform state that matches the live
resources, and the duration question above.

---

## Sources

Project numbers: `notes/CASE_STUDY.md`, `notes/decisions.md` (D-001, D-009, D-012, D-013),
`notes/2026-09-02-gcp-cloud-migration-plan.md`, `notes/2026-09-04-cloud-cost-baseline.md`,
`spark/2026-07-10-fare-prediction-modeling-plan.md` (§6, the duration ablation).

Field survey, 2026-09-26 (a sample of the repositories read):
- Course project rubric: https://github.com/DataTalksClub/data-engineering-zoomcamp/tree/main/projects
- Largest pipeline found (157 stars): https://github.com/trannhatnguyen2/NYC_Taxi_Data_Pipeline
- Keyless CI and ADRs: https://github.com/Nhut-Data/nyc-taxi-analytics
- Documented TLC duplicates: https://github.com/jumma786/nyc_taxi_dbt
- Sealed time-based test, demand: https://github.com/marc4data/portfolio-nyc-taxi
- Rolling backtest and leakage test, demand: https://github.com/pratri/nyc-taxi-demand-forecasting
- Fare model on a Kaggle CSV: https://github.com/amruth1181/dsp-taxifareprediction-project
- Fare model using recorded duration (MAE $0.27, R² 0.995): https://github.com/dishayayyy/nyc-taxi-fare-xai
- Fare prediction paper (R² 0.991): https://arxiv.org/html/2507.20008
- Kaggle fare competition, strong entries near $2.85 RMSE:
  https://github.com/sbavon/Kaggle-NYC-Taxi-Fare-Prediction

Survey limits: some findings come from a README or file tree only. The Kaggle fare
leaderboard did not load, so its top score is unconfirmed.
