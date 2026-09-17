# Fare Prediction — Classical ML Modeling Plan

**Invoke when:** any question about the fare model, the feature contract, the sweep, or the sealed holdout.

**Created:** 2026-07-10
**Author:** Sagar Sinha (with Claude Code)
**Companion doc:** `spark/2026-07-04-ml-handoff-context.md` (data provenance, baselines, anomalies, feature correlations)
**Goal:** Refresh the full classical ML toolkit (linear → trees → bagging → boosting → stacking → voting) on a real regression problem, done with correct preprocessing and fair comparison — **then** move to neural nets.

---

## 0. Framing

- **Target:** `fare_capped` (continuous → **regression**)
- **Data flow:** `<backup>/fact_trips/` (128.8M rows, 204 parquet, **Spark local**;
  the backup moved out of the repo on 2026-09-01 — see `spark/ml/src/paths.py`) → guards + stratified sample → **pandas / scikit-learn** from here on. Spark ends before modeling starts. **Torch never runs inside Spark.**
- **No GCP needed.** The dbt-built `fact_trips` backup on local disk is the complete model input. No cloud project, billing, or credential is required for any of this work.
- **Metrics:** MAE, RMSE, R² — overall + sliced by `pickup_borough`, `temp_band`, `pickup_hour`.

### Locked decisions (2026-07-10)
1. **Duration framing:** treat `trip_duration_min` like an app's *estimated-duration* input — available at prediction time as an estimate, therefore a **legitimate candidate feature, not leakage**. Final include/exclude is deferred and wired as an on/off **ablation** (run the sweep both ways, report the gap).
2. **EWR (Newark):** keep EWR trips in training; `is_airport_trip` captures the signal. Do not drop them.

---

## 1. Code structure

```
spark/ml/
├── 00_prep_spark.py            # Spark: fact_trips → cleaned → stratified sample parquet
├── data/
│   ├── sample_full.parquet     # 10% stratified (30,482,494 rows since 2026-09-13; stays in GCS) — final refit
│   └── sample_work.parquet     # 1,828,181 rows since 2026-09-13 — fast sweep iteration
├── src/
│   ├── features.py             # feature engineering (pure fns, unit-tested)
│   ├── preprocess.py           # ColumnTransformer builders, scaled + tree variants
│   └── evaluate.py             # shared CV harness + leaderboard (unit-tested)
├── notebooks/                  # one per phase, thin — calls into src/
└── models/                     # persisted champion estimators + fitted preprocessors
```
`src/` holds pure functions (feature math, cap logic, metrics) → **unit-tested per TDD**. Model fitting lives in notebooks (exploration, not TDD-appropriate).

---

## 2. Phase 1 — Data prep (Spark, one script)

Read the 204-file backup, apply §4 guards, derive caps from the data:
- Filter `fare_amount >= 0`, `trip_distance > 0`; exclude negative-duration rows (`dropoff < pickup`)
- `passenger_count` → valid 1–6, impute NULL with mode = 1
- Re-derive **p99 caps** on `fare_amount` / `trip_distance` from the data itself (defensible paper trail, don't hard-code); cap `trip_duration_min` at p99 (~45 min)
- **Stratified 10% sample** on `service_type` × `temp_band` → `sample_full.parquet`
- Also cut `sample_work.parquet` (~500K–1M rows) for fast iteration
- **Sanity gate:** `COUNT(*)` ≈ 128,781,646 pre-sample; §3 correlations survive the sample

**Output:** two clean sample parquets. Spark's job is done.

---

## 3. Phase 2 — Feature engineering + leakage-safe preprocessing

| Type | Fields | Encoding |
|---|---|---|
| Numeric | `trip_distance`, `temperature`, `passenger_count` | scale (linear/NN only) |
| Numeric (ablation) | `trip_duration_min` | on/off flag; scale (linear/NN only) |
| Cyclic | `pickup_hour` | sin/cos |
| Ordinal | `temp_band` | Freezing < Cold < Mild < Warm < Hot |
| One-hot | `pickup_borough`, `dropoff_borough`, `service_type` | OHE |
| Binary | `is_airport_trip` = `ratecodeid IN (2,3)` | flag (keeps EWR signal) |
| High-card | OD corridor (pickup zone × dropoff zone) | **target encoding, cross-fitted** |

**Leakage discipline (the make-or-break part):**
- **Split first** (70/15/15, stratified), then fit every transformer on **train only** — inside sklearn `Pipeline` / `ColumnTransformer` so leakage is structurally impossible.
- **Target encoding must be cross-fitted** (out-of-fold, `category_encoders` or manual KFold). A naive group-mean leaks the target.
- **Hard exclusions (§5 leakage list):** `tip_amount`, `total_amount`, `mta_tax`, `tolls_amount`, `improvement_surcharge`, `extra`, `revenue_per_mile`, `has_tip`, `tip_pct_of_fare`. `payment_type` excluded (not known pre-trip).
- **Scaling only where needed:** build two preprocessor variants — scaled (linear/SVM/NN) and raw (trees).

> **Partly superseded — 2026-08-09.** This section was written 2026-07-10, before Phase 1
> existed, so two parts of it describe columns and a split that never shipped. The
> *Encoding* column is still current; the rest reads as history.
>
> - **Column names.** The prep emits `distance_capped` and `temp_band_ord`. The raw halves
>   named in the table — `trip_distance` and `temperature` — are model exclusions, not
>   inputs (`d83141b`). Keeping the raw distance disabled the §2 p99 cap and let one
>   corrupt 8,003,318-mile odometer row drive a linear fold to a $9.29M prediction.
> - **The split.** 70/15/15 was never implemented. §4a supersedes it: a sealed 20% holdout
>   carved before the sweep, plus 5-fold CV on the remainder (`f07eb4c`).
>
> Both defects came from the same habit — building on what this section asserted instead of
> on what the prep actually emitted. Read §4a and the live column lists in
> `src/features.py` and `src/preprocess.py` as the authority.

---

## 4. Phase 3 — Evaluation harness (build BEFORE any model)

One `evaluate(model, X, y, cv)` that every model plugs into: **identical KFold folds, identical metrics**, returns a row for a running **leaderboard** DataFrame. This is what turns the sweep into a fair comparison instead of anecdotes. Unit-test the metric functions.

---

## 4a. Data splitting policy (decided 2026-08-01)

**Supersedes the 70/15/15 split named in §3.** That was written before the CV harness
existed; a fixed validation slice is strictly worse than the 5-fold CV we now have, which
validates against all 612K training rows instead of 115K and exposes fold-to-fold spread.

**Adopted — stratified 80/20 holdout + 5-fold CV inside the 80%:**

- `evaluate.make_holdout(X, y, test_size=0.2)` carves the split **before** the sweep sees
  the data. On the D-012 `sample_work` (2026-09-13): **1,355,641 train / 338,911 sealed
  test**, drawn from the rows before the temporal cutoff below. The pre-D-012 sample gave
  612,609 / 153,152; D-012 voided it.
- Stratified on `service_type` × `temp_band_ord` — the same key `00_prep_spark.py`
  sampled with, so the smallest stratum (Green/Freezing, ~1.5%) can't skew.
- Deterministic under `RANDOM_STATE=42`; the sealed rows are re-derivable from the
  sample file + seed + fraction (recorded in `results/sweep_<tag>.json`), so no second
  copy of the data is persisted.
- `01_run_sweep.py` seals it by default and **never scores or prints it**. `--no-holdout`
  exists for wiring smoke runs only.

**Holdout discipline — the part no code enforces.** The test split is scored **exactly
once**, in Phase 5, after the champion is chosen on CV evidence alone. Every decision
informed by a test score makes the final number optimistic: with 15 models separated by
~$0.005 MAE and SE ≈ $0.002, picking the winner *on the holdout* would bias it by roughly
the largest of 15 noise draws (~$0.005) — the same size as the real gap between the top
two models. Tune, ablate and compare freely on CV; touch the holdout at the end.

### Temporal test set — landed 2026-09-13 (was deferred)

**Landed with the baseline, 2026-09-13** (owner decision 3). `evaluate.make_temporal_test`
carves every trip with `pickup_datetime >= 2016-11-01` off the **raw** sample, before
`build_features` drops the timestamp and before `make_holdout` draws the 80/20.
`01_run_sweep.py` seals it by default, never scores or prints it, and records
`temporal_cutoff` and `temporal_rows` in `results/sweep_<tag>.json` (`a00311c`).

Measured on the D-012 `sample_work`, 1,828,181 rows:

| Set | Rows | Share of the sample |
| --- | ---: | ---: |
| Temporal test set (2016-11: 65,459; 2016-12: 68,170) | 133,629 | 7.31% |
| Random holdout — 20% of the 1,694,552 rows before the cutoff | 338,911 | 18.54% |
| Train — the sweep's rows, `sample_work_train.parquet` | 1,355,641 | 74.15% |

The share is 7.31%, not the ≈8% estimated below: 2 of 24 months is 8.33%, but the last
two months hold fewer trips than the 24-month mean of 76,174 per month. The random
holdout therefore draws from 2015-01 to 2016-10 only. The history below is kept as it
was written.

The 80/20 above is a **random** split, but the deployment case is predicting a fare for a
trip happening *now* from a model trained on *past* trips. A random split lets 2016-06
inform a 2016-03 prediction, which is optimistic relative to reality by an unmeasured
amount. Rate cards didn't change across 2015–16 and a metered fare is close to
deterministic in distance and time, so drift is *probably* small — "probably" being
exactly what a temporal holdout would replace with a number.

**Blocked on data, not effort:** `select_model_columns` (`00_prep_spark.py`) consumes
`pickup_datetime` into `pickup_hour` / `pickup_dow` and drops it, so the samples carry no
date to split on.

When the prep is next re-run:

1. Add `pickup_datetime` (or just a `pickup_month` key) to the `keep` list.
   **Done** — `4985c03`, in the 2026-09-13 samples.
2. Carve a second test set: the **last 2 of 24 months (2016-11, 2016-12, ≈8%)**.
   **Done 2026-09-13** — 133,629 rows, 7.31%, measured; see above.
3. Score the Phase-5 champion on **both** test sets and report the pair. **Open — Phase 5.**
   - **Agreement** → the random split was safe, and you can say so with evidence rather
     than assertion.
   - **Divergence** → real temporal drift, quantified.

Either outcome is a reportable result. This is additive — it does not replace the random
holdout, and doing the random holdout now costs nothing against it.

---

## 5. Phase 4 — The model sweep

All models wrapped in `Pipeline(preprocessor, model)`, all scored through the Phase-3 harness, run on `sample_work` for speed:

| Family | Models |
|---|---|
| **Linear** | OLS, Ridge, Lasso, ElasticNet |
| **Single tree** | DecisionTreeRegressor (interpretable baseline) |
| **Bagging** | RandomForest, ExtraTrees, `BaggingRegressor` |
| **Boosting** | sklearn GradientBoosting, XGBoost, LightGBM, CatBoost (native categoricals) |
| **Stacking** | `StackingRegressor`: bases = {RF, XGB, Ridge}, meta = Ridge |
| **Voting** | `VotingRegressor` over top 2–3 |

**Predict the result first, then verify:** linear underfits the airport non-linearity; single tree overfits; RF/boosting lead; stacking/voting eke out a small gain. Confirming that narrative *is* the exercise.

---

## 5b. Phase 4b — Spark MLlib baseline (D3, adopted 2026-07-30)

One Spark-native model — the DE-portfolio counterpart to the sklearn sweep. Runs **after**
Phase 4 so there is a locked champion to compare against.

**Where it runs — revised 2026-08-04.** Locally, the baseline trains on the **`sample_work`
train split: the same 612,608 rows the sklearn sweep used**, not on `sample_full`. Full-scale
training on `sample_full` (12.75M rows) — for **both** MLlib *and* sklearn — moves to the
**Cloud** run and is not attempted on the local machine. See §5c.

Two things fall out of running local MLlib on the sweep's own train split, both good:

- The comparison is **like-for-like on rows.** Same pool, same feature set, same 15-feature
  contract — the only remaining difference is the stack itself, which is the thing being
  measured.
- The sealed 153,153-row holdout stays **genuinely untouched by every model in the project**,
  MLlib included. Training MLlib on `sample_full` would have put 1.2% of its training data
  inside the sealed test set, which is harmless for CV-vs-CV but poisons any write-up that
  places an MLlib number beside the Phase-5 holdout score.

The split cannot be reproduced inside Spark — `make_holdout` is `sklearn.train_test_split`
and seeds do not transfer across libraries. So pandas carves it once and persists
`sample_work_train.parquet` (regenerable from sample + seed + frac; gitignored), and the
Spark script reads that file.

**Model:** `pyspark.ml.regression.GBTRegressor` — the strongest thing MLlib offers, so the
head-to-head is meaningful rather than a strawman. (`LinearRegression` is a one-line
addition if a linear floor is wanted too; not required.)

**Script:** `spark/ml/01_mllib_baseline.py`, following the `00_prep_spark.py` convention —
a parameterizable Spark script, not a notebook. Pure helpers (column-group builders, the
metrics→leaderboard-row adapter) go in `src/` and are **unit-tested per TDD**; the fit
itself is a script run, like Phase 1.

**Pipeline stages** (`pyspark.ml.Pipeline`):
1. `StringIndexer` + `OneHotEncoder` on `service_type`, `pickup_borough`, `dropoff_borough`
2. `StringIndexer` + `TargetEncoder` on `od_corridor`
3. numerics passed through raw — GBT is a tree, no scaling needed
4. `VectorAssembler` → single `features` vector
5. `GBTRegressor(labelCol="fare_capped", seed=42)`

**Correction — 2026-08-09: `od_corridor` stays in.** An earlier draft of this section
dropped the corridor and justified the drop with "MLlib has no `TargetEncoder`". That is
false for this project. `pyspark.ml.feature.TargetEncoder` was added in **Spark 4.0.0** and
we run **4.0.1**. The claim came from Spark 3.x and was never checked against the installed
version, so the 2026-08-08 baseline shipped without a feature it could have had. Same
failure mode as `d83141b`: acting on a believed statement instead of a verified fact.

**Parity rule — decided 2026-08-09.** Whatever configuration the sklearn sweep runs, the
MLlib baseline runs too. Same rows, same 15-feature contract, same encoding families. What
is left over is the stack, and the stack is the only thing this section exists to measure.
A gap that comes from a missing feature measures nothing anyone wants to know.

One difference survives the parity rule and cannot be removed:

- sklearn's `TargetEncoder` **cross-fits** inside `fit_transform`, so a training row's
  encoding is built from the other inner folds and excludes its own target.
- Spark's `TargetEncoder` does not. `fit` takes the plain per-category mean, so a training
  row's own fare enters its own feature.

**The sklearn half is measured, not assumed — 2026-08-09.** This section has now cost the
project twice by asserting library behaviour it never checked, so the claim was tested on
the real `build_preprocessor`. Worst case, 12 rows, every corridor unique, targets 10–21,
global mean 15.5. `pre.fit_transform(X, y)` returned values near 15.5 with the row's own
target excluded; `pre.fit(X, y).transform(X)` returned 10, 11, 12 … — the target itself. A
spy transformer inserted between `pre` and the model inside a real `Pipeline` confirmed the
model receives the cross-fitted values, so `Pipeline.fit` does take the `fit_transform`
path. sklearn 1.7.1. Three conditions hold together: the encoder sits inside the
`Pipeline`, `cross_validate` refits it per fold, and `make_holdout` runs before the sweep.
Had any one failed, every sklearn score in `sweep_work.json` would be inflated.

Both stacks are safe against **test-fold** leakage, because the pipeline is fitted on each
fold's train half only. Neither CV score is inflated. The Spark defect stays inside the
training half: the model over-trusts the encoding, which biases its result **downward**, not
upward. It is not hypothetical — 5,373 of the 18,668 corridors in the train split hold
exactly one trip (28.8% of corridors, 0.88% of rows), and for those a plain group mean *is*
the row's own fare. Spark's `smoothing` parameter shrinks small categories toward the global
mean ($12.69) and is the mitigation; choose the value deliberately and record it. Report the
residual difference as a §10 Tier-2 finding — real, and far smaller than a missing feature.

**Unknown categories agree across the stacks — one fewer difference to report.** Spark's
`TargetEncoder` with `handleInvalid="keep"` maps an unseen value to the dataset overall
statistics. sklearn's maps it to the global target mean. Same fallback, so no adjustment is
needed for parity. This path is live rather than theoretical: `make_cv` is a plain
`KFold(shuffle=True, seed=42)` and is **not** stratified, and no split stratifies on
`od_corridor` — with 5,373 single-trip corridors, none could. Measured over the 612,608
train rows, **1.03%–1.10% of every CV test fold** carries a corridor absent from that fold's
train half (6,491 rows across the five folds, stable fold to fold). Those rows fall back to
the global mean and lean on `distance_capped` and `trip_duration_min` instead. That is the
production case — a route with no history — priced into the CV score at its true rate.

The `OneHotEncoder` mismatch does remain: Spark's `handleInvalid="keep"` gives an unseen
category its own bucket, sklearn's `handle_unknown="ignore"` gives all zeros. It should
never fire, because boroughs and `service_type` are closed sets from the zone lookup.

**The rows to report:**

| Row | Stack | Rows | `od_corridor` | Status |
|---|---|---|---|---|
| 1. Phase-4 champion (`lightgbm`) | sklearn | 612,608 (`sample_work` train) | ✅ target-encoded, cross-fitted | done — `sweep_work` |
| 2. MLlib GBT | Spark | 612,608 (`sample_work` train) | ✅ target-encoded, smoothing 5 | **done 2026-09-01 — `sweep_mllib_gbt`** |
| 3. MLlib GBT, corridor dropped | Spark | 612,608 (`sample_work` train) | ❌ | done — `sweep_mllib_gbt_nocorr` |

Row 1 against row 2 is the like-for-like stack comparison, and it is the headline. Row 3
against row 2 quantifies what the corridor is worth inside Spark. Row 3 is already paid
for: the 2026-08-08 run produced it before this correction, so the ablation costs nothing
extra and the earlier run is not wasted.

**The `--drop-corridor` flag is no longer required.** It existed to supply a
corridor-dropped *sklearn* row, which was only necessary while MLlib could not hold the
corridor. Row 3 now supplies that ablation on the Spark side. The flag stays available as
optional work if the sklearn-side corridor value is wanted as well.

**Naming, not schema.** All rows share the `evaluate()` key set, so the pool goes in the
`model` string (`mllib_gbt@work612k`) rather than a new column. Adding a field would break
the shared-leaderboard contract that `test_row_keys_match_the_evaluate_contract` exists to
protect. Rows 2 and 3 differ only by a feature, so the model string must separate them —
the 2026-08-08 board already claims `mllib_gbt@work612k` for the corridor-dropped run, and
that run is now the ablation, not the baseline. Rename it to `mllib_gbt_nocorr@work612k`
when row 2 lands, so the unqualified name means the full-feature baseline.

**Metrics + folds:** `RegressionEvaluator` for mae/rmse/r2 — the same three metrics — with
**the same k (5) and the same train pool as `make_cv()`, but different fold membership.**
Seeds do not transfer across libraries: `KFold(seed=42)` and a Spark splitter at `seed=42`
partition differently. Observed fold-to-fold SE in the work sweep was ~$0.002, so this is
noise rather than a confound — but it is not the equivalence an earlier draft of this
section claimed. Results adapt into the existing `leaderboard()` as ordinary rows, so
everything still lands in one table.

**Expected outcome, stated up front (2026-08-09):** with the corridor present on both
sides, the MLlib GBT lands close to but behind the sklearn boosting champion — from MLlib's
weaker GBT implementation and its uncross-fitted encoder, no longer from a missing feature
— and costs materially more wall time for the same 612,608 rows, because Spark's scheduling
and shuffle overhead buys nothing at a size that fits in memory. Confirming *that* is the
point locally: it establishes the per-row cost of each stack cleanly, and §5c is where the
scale argument gets made instead.

### Measured outcome — 2026-09-01. Half of that prediction was wrong.

| Row | Model | MAE | RMSE | R² | s/fold |
|---|---|---|---|---|---|
| 1 | `lightgbm` (sklearn champion) | **0.3503** ±0.0015 | **1.0516** | **0.9883** | **1.0** |
| 3 | `mllib_gbt_nocorr@work612k` (corridor dropped) | 0.4828 ±0.0038 | 1.2582 | 0.9833 | 51.4 |
| 2 | `mllib_gbt@work612k` (corridor, smoothing 5) | 0.5202 ±0.0050 | 1.4502 | 0.9778 | 170.5 |
| — | same, smoothing 20 (first attempt) | 0.5292 ±0.0096 | 1.5258 | 0.9753 | 220.9 |

**The wall-time half held, and then some.** Row 2 costs **170.5s per fold** against
lightgbm's **1.0s** — a 167x gap on identical rows, where row 3's was 50x. The encoder
stage more than tripled the per-fold cost, and the whole 5-fold run took 989s.

**The accuracy half was wrong, and interestingly so.** Row 2 was supposed to close most of
the gap to row 1 by restoring the corridor. It did the opposite: **adding the corridor made
MLlib worse than dropping it**, on all three metrics, and roughly doubled the fold-to-fold
spread. The same feature that carries the sklearn champion is a liability in Spark.

**It is not an artefact of the smoothing value.** The obvious objection — a badly chosen
knob — was tested before the result was written down. Seven arms, identical rows (200,000),
3 folds, `maxIter=20`, mean MAE:

| corridor dropped | s=5 | s=1 | s=0.067 | s=100 | s=20 | s=500 |
|---|---|---|---|---|---|---|
| **0.6329** | 0.6650 | 0.6713 | 0.6908 | 0.7094 | 0.7102 | 0.7236 |

**No smoothing beats dropping the corridor.** The best encoded arm (s=5) still loses by
0.03 MAE, and the curve has no useful minimum — it is flat-ish and bad on both sides. So
row 2 is reported at s=5, the measured best, rather than at the s=20 the leakage bound
originally argued for. The first attempt is kept as `sweep_mllib_gbt_s20`, because a
hyperparameter chosen from data and then reported is worth showing the search for.

**What the difference is, and what it is not.** The two stacks now differ in exactly one
respect: sklearn's `TargetEncoder` cross-fits inside `fit_transform`, Spark's does not.
Both are safe against test-fold leakage — each pipeline is fitted on its own fold's train
half — so neither CV score is inflated. The residual sits in the training half, and §5b
predicted it would bias Spark **downward**. It does. What was not predicted is the size:
enough to make the feature net-negative. The mechanism is consistent with over-trust rather
than with an inflated score — during training the encoding partly contains each row's own
fare, so the booster leans on a signal that is weaker at prediction time — but this run
establishes the *effect*, not the mechanism. Isolating the mechanism would need a
cross-fitted encoding computed outside MLlib and fed in as a plain column. That is not
scheduled; it is written here so the claim is not quietly upgraded later.

**The honest portfolio statement** is therefore not "Spark's GBT is a bit behind sklearn's".
It is: at 612,608 rows on one machine, the sklearn stack is **1.5x more accurate and 167x
faster**, and the one preprocessing capability MLlib lacks — a cross-fitted target encoder —
is worth more than the feature it encodes. §5c is where the scale argument gets made
instead, and it is the only place the Spark side can win.

### Measured on the D-012 split — 2026-09-15, with the mechanism test

D-012 voided the table above: its 612,608 rows came from the pre-fix build. The rows below
are on the 1,355,641-row D-012 train split, and they do not compare with the rows above.
The three MLlib rows: 5 folds, seed 42, GBT `maxIter=100` `maxDepth=5`, smoothing 5,
Spark 4.0.1 in the dev container, `local[8]`, `default_parallelism` 8, so the fold
membership is the same. Row 1 comes from the `work_d012` sweep of 2026-09-15, the same
1,355,641 rows and `KFold(5, seed=42)`. No holdout or temporal metric was computed.

| Row | Model | `od_corridor` | MAE | RMSE | R² | Mean fit |
|---|---|---|---|---|---|---|
| 1 | `lightgbm` (sklearn) | target-encoded, **cross-fitted** | **0.331289** ±0.001310 | **0.936259** ±0.029962 | **0.990686** ±0.000610 | **5.1 s** |
| 3 | `mllib_gbt_nocorr@work1355k` | dropped | 0.453134 ±0.003220 | 1.138785 ±0.006900 | 0.986234 ±0.000206 | 389.0 s |
| OOF | `mllib_gbt_oof@work1355k` | cross-fitted out-of-fold, outside MLlib | 0.462342 ±0.000505 | 1.161502 ±0.011404 | 0.985679 ±0.000304 | 473.8 s |
| 2 | `mllib_gbt@work1355k` | `TargetEncoder`, not cross-fitted | 0.466188 ±0.005356 | 1.201377 ±0.021579 | 0.984675 ±0.000561 | 532.8 s |

**This is the test the 2026-09-01 reading asked for:** a cross-fitted encoding computed
outside MLlib and fed in as a plain column (migration plan §5.3, `src/oof_encode.py`). A
training row's encoding now comes from the other inner folds only, and the test fold is
encoded from the whole train half.

**What it shows**, with "material" meaning more than the compared run's fold std:

- **The corridor is still net-negative in MLlib.** Dropping it beats the cross-fitted
  encoding on all three metrics, by 2.7x to 3.3x row 3's fold std, and on MAE in all 5 folds.
  So the uncross-fitted encoder is not the whole story.
- **On MAE the OOF row equals row 2.** It is 0.003846 below, inside row 2's std of
  0.005356. Read on MAE, cross-fitting did not help.
- **On RMSE and R² the OOF row is materially better than row 2** (1.85x and 1.79x row 2's
  std). Read on those, the encoder was part of the story and MLlib's GBT is the rest.
- The OOF row's MAE spread is a tenth of row 2's (0.000505 against 0.005356).

- **Row 1 against the Spark rows holds the 2026-09-01 shape, at a smaller accuracy gap.**
  lightgbm's MAE is 0.134898 below row 2 (25.2x row 2's fold std) and 0.121845 below row 3
  (37.8x row 3's). On identical rows the sklearn stack is **1.41x more accurate and 105x
  faster** than row 2, against 1.5x and 167x on the voided 612,608 rows. Against row 3 it
  is 1.37x and 77x.

The migration plan §5.4 carries the differences and the leakage check on real data. The
owner read §5.4 on MAE on 2026-09-15, so the outcome row is "at or above row 2" and D2
stands — recorded as **D-013** in `notes/decisions.md`.

---

## 5c. Full-scale training — deferred to Cloud (decided 2026-08-04)

**Both stacks train on `sample_full` (12,748,027 rows) in the Cloud, not locally.** This
covers the sklearn champion's final numbers *and* the MLlib at-scale run. Locally we stay on
the `sample_work` train split (§5b).

Measured on the local machine (Apple M3 Pro, 18 GiB RAM, 11 cores) on 2026-08-04:

| | value | source |
|---|---|---|
| feature frame, full sample | **4.8 GB** | extrapolated from a 500K-row slice |
| 80% train split | **3.9 GB** | same |
| dominated by | `od_corridor` 66 MB / 500K rows, + 3 more object-dtype string columns (80% of footprint) | per-column `memory_usage(deep=True)` |
| work sweep, 14 models × 5 folds | **30.1 min** wall (`elapsed_s` 1805.9) | `results/sweep_work.json` |
| scale factor 612,608 → 10.2M train | **16.6×** | — |
| whole sweep at full scale, linear floor | **~8.2 h** | Σ fit_time × 5 × 16.6 |

Three things this table says:

1. **The sweep runs sequentially.** Σ(per-fold fit) × 5 folds = 1778.5s ≈ the 1805.9s
   elapsed, so `n_jobs` is effectively 1 and ten cores idle. Cloud sizing should fix that
   before it buys more RAM.
2. **Cost is wildly uneven across models.** At full scale: lightgbm and xgboost ~1.4 min
   each, catboost ~13 min — but `stacking` ~4 h+ (it refits bases under internal CV, so it
   scales superlinearly), `gradient_boosting` ~83 min, `random_forest` ~50 min with a real
   memory risk from fully-grown trees on 10.2M rows.
3. **Re-running all 14 at scale answers nothing.** Selection is already settled — `lasso`
   and `elasticnet` posted RMSE ~3.0 against lightgbm's 1.05, and 20× the data does not
   rehabilitate a model that is three times worse. Full scale exists to produce the
   champion's final number and §5b row 2, which needs 3–4 models, not 14.

**Cloud run scope, when it happens:** top 4 by RMSE (`lightgbm`, `catboost`, `stacking`,
`extra_trees`) + the corridor-dropped champion + MLlib GBT. Include `stacking` despite its
cost — the top four sit within ~$0.005 MAE of each other, and bagging/stacking families
typically gain more from data than boosters do, so the ranking may legitimately move at
scale. That is a result worth having rather than assuming.

**Cheap enabler, do it first:** cast `od_corridor`, `service_type`, `pickup_borough`,
`dropoff_borough` to `category` dtype. Four string columns are ~80% of the 4.8 GB;
`od_corridor`'s 19,953 levels become int16 codes plus a small dictionary. Expect the frame
to drop to roughly a third. Verify `TargetEncoder` and `OneHotEncoder` still behave on
categorical dtype before relying on it.

**Unknown, not estimated:** `TargetEncoder` cross-fitting over 19,953 corridor levels at
10.2M rows. It cannot be extrapolated from a 500K slice — measure it with a single-model
probe (`--sample full --only lightgbm`) before committing to a machine size.

### Re-scoped to D-012 — 2026-09-15

The figures above are pre-D-012 and the scope below replaces the model list. Measured
2026-09-15:

- **`sample_full` holds 30,482,494 rows**, 2.4x the 12,748,027 above, and it exists in
  `gs://primary-data-dtc-506916/ml/prep/` only — not on the laptop. Every size and time
  figure above is therefore a **floor**, not an estimate, and the machine-size question is
  **UNVERIFIED** until the lightgbm probe runs.
- **The model list survives, with a new order.** The `work_d012` sweep (14 models,
  1,355,641 rows, 5 folds, 7,468.0 s in the dev container) gives the same top four by RMSE:
  `catboost` 0.928765, `lightgbm` 0.936259, `stacking` 0.937680, `extra_trees` 0.947680.
  The first two swapped places, and their gap of 0.007494 is inside the fold std of
  0.030278, so the order between them is noise. On MAE `catboost` beats `lightgbm` by 8.1x
  its fold std.
- **The champion is metric-dependent on this split.** By RMSE it is `catboost`; by MAE it
  is `stacking` (0.319527), with `extra_trees` 0.319730 and `catboost` 0.320963 inside one
  fold std of it. `leaderboard()` sorts by RMSE.
- **`stacking` costs 675.7 s per fold here**, 133x `lightgbm`'s 5.1 s, so it dominates any
  full-scale budget.
- **The MLlib arm is row 3's configuration** — the corridor dropped (`--drop-corridor`) —
  per **D-013**, because the corridor stays net-negative in MLlib even cross-fitted.

---

## 6. Phase 5 — Tune + diagnose

- Hyperparameter search (`RandomizedSearchCV` / Optuna) on top 2–3 only
- **Refit champion on `sample_full`** (the 12.8M-row sample)
- Residual plots, learning curves, feature importance + **SHAP** on champion
- Slice metrics by borough / temp_band / hour → final leaderboard
- Run the **duration on/off ablation** and report the gap

### The finish plan — the owner's decisions and the TUNE goal (2026-09-16)

The owner took these decisions on 2026-09-16, after M4, §5.4 and D-013, and before any
Phase 5 work. They set the fastest path to the end of the project. The TUNE goal below runs
stage 1 of that path. It is stored as the owner accepted it; paste it as written. It is
3,945 characters.

Measured while the goal was written, 2026-09-16, read-only:

- **`sample_full` contains the sealed rows.** `00_prep_spark.py:334` draws `sample_work`
  from `sample_full` (`stratified_sample(full, ...)`). So the holdout (338,911 rows) and the
  2016-11 and 2016-12 months of the temporal test set sit inside `sample_full`. The bullet
  above, "Refit champion on `sample_full`", would train on the rows the holdout later
  scores. §5b recorded the same overlap for MLlib only. No exclusion step exists.
- **No tuning code exists.** The dev container has `optuna` 4.9.0 and `shap` 0.52.0; the
  host `.venv` has neither, nor `lightgbm` or `catboost`.
- **CatBoost leads on MAE.** `work_d012`: `catboost` 0.320963 ± 0.001271, fit 30.5 s;
  `lightgbm` 0.331289 ± 0.001310, fit 5.1 s.
- **The gate is the full suite again.** Host `.venv/bin/pytest tests/`: 385 passed,
  1 skipped. Container `pytest tests/`: 386 passed. The `--ignore` of
  `test_oof_encode.py` hid 21 tests.

| # | Question | Decision | Why | Consequence |
| --- | --- | --- | --- | --- |
| 1 | The end of the project | **Four results:** a tuned champion scored once on the holdout and the temporal set; M5; a true README, `CASE_STUDY.md` and dashboard; nothing running or billing | A finish line that can be checked | Four stages: tune, scale, score once, close |
| 2 | Which rows train the holdout champion | **The `sample_work` train split**, 1,355,641 rows | The rows are clean today; no exclusion code, no VM | The "Refit champion on `sample_full`" bullet above is superseded |
| 3 | Phase 6, neural nets | **After the end** | It is the largest single block of work | D-014, DEFERRED |
| 4 | §5c at full scale | **The champion + MLlib row 3** | `stacking` was a 4 h floor at 12.75M rows | Both train on `sample_full`, so their numbers are **CV only** and never sit beside the holdout score |
| 5 | The dashboard | **Reconnect `ds0` + pages 1–2** | Pages 3–6 and the polish were 5 of 10 planned days | The Day 1 baseline numbers predate D-012 and are refreshed |
| 6 | The champion rule | **Lower tuned CV MAE; a gap inside the larger fold std goes to the faster fit** | MAE is the owner's metric (D-013); speed decides the §5c cost | A close result picks `lightgbm` |
| 7 | The tuning budget | **45 min per model**, Optuna `TPESampler(seed=42)`, trial 0 the defaults | Trial 0 proves the harness equals the sweep | About 2 h, ESTIMATED |
| 8 | Speed | **One goal per stage** | The per-step review moves to the goal boundary | The owner's UI work runs beside the compute |

Options not taken: refitting on `sample_full` with the sealed rows removed; Phase 6 inside
the project; the top four at full scale, or MLlib only; the full v3 dashboard.

<details><summary>The TUNE goal command</summary>

```text
/goal Tune and diagnose the Phase 5 champion on the D-012 work train split, with the holdout still sealed. MET only when the transcript shows a final report headed "TUNE DONE" with all of:
(a) notes: the 2026-09-16 decisions table and this goal in the modeling plan; D-014 DEFERRED (Phase 6) in notes/decisions.md;
(b) TDD src/tune.py: failing test shown in the container, then code; its tests importorskip optuna;
(c) trial 0 = defaults, per model: CV MAE against work_d012, the difference shown;
(d) catboost and lightgbm tuned: trials, best params, mae, rmse, r2 mean and std, elapsed;
(e) the champion by the FACTS rule, both MAEs and stds shown;
(f) champion diagnostics on out-of-fold predictions: MAE by pickup_borough, temp_band_ord and hour; mean |SHAP| per feature; duration on/off MAE gap;
(g) docs, measured numbers only (D-009): modeling plan §6 and Status; next = §5c at scale;
(h) commits: this goal's note + D-014, the tune code, then docs; gate before: host 385 passed 1 skipped, container 386 passed; after: 0 failed, only new tests added; git status clean; unpushed.
Judge IMPOSSIBLE if a line starts "TUNE STOPPED:". Stop after 40 turns.

FACTS
- The owner suspends the CLAUDE.md per-step review for this goal.
- Read first: modeling plan §4a, §6, §8; notes/decisions.md (D-005, D-009, D-013).
- Owner decisions 2026-09-16: the holdout champion trains on the work split; Phase 6 after the end; §5c at scale = champion + MLlib row 3, CV only; dashboard = reconnect + pages 1-2.
- Champion rule, owner 2026-09-16: lower tuned CV MAE; a gap inside the larger fold std goes to the faster fit.
- sample_full contains sample_work (00_prep_spark.py:334), so it holds the sealed rows. Nothing here reads it.
- sample_work.parquet: 2 files, 110,474,656 B, 1,828,181 rows. 01_run_sweep.py carves temporal 133,629, holdout 338,911, train 1,355,641.
- work_d012, 5 folds, seed 42: catboost mae 0.320963 ± 0.001271, fit 30.5 s; lightgbm mae 0.331289 ± 0.001310, fit 5.1 s.
- No tuning code exists. Container: optuna 4.9.0, shap 0.52.0, catboost 1.2.10, lightgbm 4.7.0, sklearn 1.7.1; 11 CPUs, 11.67 GiB; Airflow 6 services up. The host lacks optuna, shap, lightgbm, catboost.
- Tuning: Optuna TPESampler(seed=42); objective = evaluate() on make_cv(5); metric MAE; timeout 45 min per model. About 2 h, ESTIMATED.
- Hour comes from pickup_hour_sin/cos; the frame has no hour column.
- Gate: .venv/bin/pytest tests/ -> 385 passed, 1 skipped. Container: pytest tests/ -> 386 passed.

STEPS
1. Gate both. Commit this goal's note and D-014.
2. TDD src/tune.py: search spaces; the objective reuses sweep.build_pipeline and evaluate; trial 0 enqueues the defaults.
3. Write spark/ml/02_tune.py: carve exactly as 01_run_sweep.py, tune both, write leaderboard_tune_d012.csv and tune_d012.json (best params, trials, elapsed). Commit 2-3.
4. Pick the champion by the rule.
5. Diagnose it: out-of-fold predictions on make_cv(5); slice MAE; SHAP on 20,000 train rows, seed 42; duration off via build_features(include_duration=False).
6. Docs (g). Commit.
7. Final gate both. Print "TUNE DONE" with (a)-(h) and the next step: §5c at scale.

PRINT "TUNE STOPPED: <reason>" AND END WHEN
- a row count differs from FACTS;
- trial 0's MAE differs from work_d012 by more than that model's fold std;
- a new test passes before its code exists, or a gate shows a new failure;
- a model's tuning passes 2x its timeout, or the run passes 4 h;
- anything computes a holdout or temporal metric;
- an action conflicts with a LOCKED entry in notes/decisions.md;
- the same tool failure happens twice.

RULES
- Stay on refactor/wire-pipeline. Never push. No Co-Authored-By.
- No cloud calls: no GCS, BigQuery or Dataproc. Delete nothing.
- Run ML in the dev container. Never read sample_full.
- Never score, print or select on the holdout or the temporal set.
- Out of scope: §5c, holdout scoring, Phase 6, CASE_STUDY.md (D-005), the dashboard, dbt edits, terraform.
```

</details>

#### The TUNE results (2026-09-16 to 2026-09-17)

Ran to completion. LightGBM tuned locally in the dev container; CatBoost's tuning needed
three attempts on this laptop before it ran on a short-lived GCE VM. Full measured record:

- **Trial 0 reproduces the sweep exactly, across every attempt.** Both models' trial 0
  equals `work_d012` to full float precision — LightGBM 0.3312894770867774 (difference 0,
  inside its 0.001310 std) and CatBoost 0.3209632524909859 (difference 0, inside its
  0.001271 std) — on the two aborted local attempts and the completed VM run alike.
- **LightGBM, 28 trials, 5,278.4 s in the dev container.** Best is trial 21: MAE
  **0.312976 ± 0.001181**, RMSE 0.924104 ± 0.029779, R² 0.990926 ± 0.000596, fit 19.1 s;
  `n_estimators` 617, `learning_rate` 0.031875, `num_leaves` 174, `min_child_samples` 13,
  `colsample_bytree` 0.921348, `reg_lambda` 4.121935.
- **CatBoost, 7 trials, 2,886.6 s on the cloud VM.** Best is trial 5: MAE
  **0.319282 ± 0.001334**, RMSE 0.928017 ± 0.031592, R² 0.990848 ± 0.000634, fit 89.7 s;
  `depth` 8, `random_strength` 2.123391, `subsample` 0.590912. Its local attempts reached
  only 1 and then 6 trials before freezing — see "The laptop failed three times" below.
- **Champion, by the owner's rule: LightGBM.** The gap over CatBoost's tuned MAE is
  0.006306, which exceeds CatBoost's own fold std of 0.001334 — LightGBM wins outright, not
  on the speed tie-break. It is also 4.7x faster to fit: 19.1 s against 89.7 s.
- **Champion diagnostics**, on out-of-fold predictions from the tuning folds:
  - Overall: MAE 0.312976, RMSE 0.924583, R² 0.990927 — equal to the tuned CV mean, as
    expected on the same folds.
  - By `pickup_borough`: Manhattan (1,153,561 rows) is best-served at 0.281533; EWR
    (11 rows) at 19.648423 and Staten Island (40 rows) at 1.307828 are worst, both too thin
    to read as anything but noise.
  - By `temp_band_ord` (0=Freezing…4=Hot): a mild upward slope, 0.300806 to 0.321046,
    flattening at band 4 (0.315272).
  - By hour: a trough at 6h (0.290058), peaks at 4h (0.365363) and 16h (0.355205) —
    consistent with thin overnight and evening-peak volume, not an obvious signal.
  - Mean |SHAP| ranks `distance_capped` (4.018100) and `trip_duration_min` (2.887503) far
    above everything else; `is_airport_trip` (0.429733) and `od_corridor` (0.222188) are a
    distant third and fourth. Every borough one-hot column sits below 0.01.
  - **Duration on/off ablation**, tuned params, same rows and folds: on 0.312976 ± 0.001181
    against off 1.225862 ± 0.001602 — a **+0.912886** MAE gap. Duration carries most of the
    model's accuracy, matching the SHAP ranking. The params were tuned with duration on, so
    this is the cost of removing the feature from the tuned model, not a re-optimized
    off-model's ceiling.

**The laptop failed three times**, all on the CatBoost re-run, never on LightGBM or the
diagnostics:
1. The lid closed on battery power mid-study; the container froze with no CPU ticks
   advancing, caught via `/proc/1/stat`.
2. A second attempt, on AC power, was killed by the harness's own low-memory guard — host
   swap was 91% full. The idle Airflow stack (up 20 h, 3.7 GiB, its scheduler unhealthy)
   was stopped to free it; restart with `docker compose -f airflow/docker-compose.yaml
   start` when the stack is needed again.
3. A third attempt froze again despite `caffeinate`, coinciding with a silent switch back
   to battery power.

The fourth attempt moved to a short-lived `e2-standard-4` GCE VM (`tune-catboost-vm`,
us-central1-a, reached over an IAP SSH tunnel; no external GCP calls, since the job only
reads local files), which ran cleanly with no freeze. **Cost: 1.6551 hours × $0.161/hour ≈
$0.266.** The VM was deleted immediately after; `gcloud compute instances list` returns
empty.

---

## 7. Phase 6 — Switch to neural nets (only after classical champion locked)

- PyTorch feedforward, with **embedding layers for high-cardinality categoricals** (OD corridor) instead of target encoding — the one place a net genuinely adds value
- Head-to-head vs the tree champion on the same held-out test set
- Single-machine training on the in-memory sample — no distributed Spark/torch

---

## 8. Cross-cutting principles

1. **Compute tiers:** sweep on `sample_work` for fast iteration; refit winner on `sample_full`. Avoids minutes-per-fit × dozens of models.
   **Sizes since the 2026-09-13 baseline, measured:** `sample_work` **1,828,181** rows, of
   which the sweep trains on **1,355,641** (§4a); `sample_full` **30,482,494** rows. The
   owner kept the emitted sizes rather than re-run or subsample to the original tiers
   (~500K–1M and ~12.8M), so the local work sweep runs on about 2.2x the old 612,609
   training rows. `sample_full` stays in GCS — §5c trains in the cloud — so a local
   `--sample full` run fails until someone fetches it.
2. **Reproducibility:** fixed `random_state` everywhere; persist samples + fitted preprocessors so the leaderboard is regenerable.
3. **Leakage-safe by construction:** all preprocessing inside Pipelines fit on train folds only; target encoding cross-fitted.
4. **Fair comparison:** every model through the one harness, same folds, same metrics.

---

## 9. Prerequisite: clear the disk barrier

Currently 92% full / ~36 GB free. Reclaim ~10 GB of dead FHVHV homework (untracked, unrelated to the 2015–16 fare pipeline) before Spark needs spill headroom:

| Path | Size | Action |
|---|---|---|
| `spark/test_data/` (fhvhv 2023) | 3.5 G | delete |
| `spark/data/` (old raw/pq) | 2.4 G | verify then delete |
| `spark/fhvhv_tripdata_2021-01.csv` + `.gz` | 0.8 G | delete |
| `spark/fhvhv/`, `spark/fhvhv_susbet/` | 0.6 G | delete |
| **`migration_backup/`** | 7.1 G | **KEEP — crown jewels.** Moved to `../nyc_taxi_migration_backup/` on 2026-09-01 (audit item 10); same disk, outside the repo. |

---

## 10. Spark vs scikit-learn — scope decision (2026-07-30)

Revisited after Phases 1–3 landed: *how much of this pipeline should be PySpark?* Audit of
what exists today — only `00_prep_spark.py` imports `pyspark`; `features.py` is pandas,
`preprocess.py` and `evaluate.py` are sklearn.

### The three tiers

**Tier 1 — converts cleanly, 1:1.** Every function in `features.py` is a column expression
with no cross-row state:

| pandas | PySpark |
|---|---|
| `pd.to_numeric(...).astype(float64)` | `F.col(c).cast("double")` |
| `np.sin(2π·hour/24)` | `F.sin(2*pi*F.col("pickup_hour")/24)` |
| `.map({band: i})` | `F.when(...).otherwise(...)` chain |
| `pickup_zone + "→" + dropoff_zone` | `F.concat_ws("→", ...)` |
| `.drop(columns=[...])` | `.drop(*cols)` |

`cast_decimal_columns` becomes unnecessary in Spark — Decimal-as-object is a *pandas*
artifact of reading `decimal(38,9)`; Spark reads it as native `DecimalType`.

**Tier 2 — converts with real rework.** `StandardScaler`/`OneHotEncoder` exist in
`pyspark.ml.feature`, but the shape changes: `ColumnTransformer` → `VectorAssembler` + a
stage chain, and OHE becomes two stages (`StringIndexer` → `OneHotEncoder`).
**The blocker is `TargetEncoder` — Spark ML has no equivalent.** Cross-fitting 19,953 OD
corridors by hand (fold ids → group means over each fold's complement → join back with
smoothing) is the one place a bug is *silent leakage* that inflates R² unnoticed. In
`evaluate.py`, metrics map to `RegressionEvaluator`, but `cross_validate`'s per-fold score
arrays and `fit_time` need hand-rolling, and the identical-folds-for-every-model guarantee
(the entire point of Phase 3) gets **harder** to enforce, not easier.

**Tier 3 — doesn't convert.** MLlib coverage against the §5 sweep: ✅ LinearRegression
(`elasticNetParam` covers OLS/Ridge/Lasso/ElasticNet), DecisionTree, RandomForest;
⚠️ `GBTRegressor` (weaker than sklearn GB), XGBoost via `xgboost.spark`, LightGBM via
SynapseML (heavy JVM dep), CatBoost (fragile Spark package); ❌ **ExtraTrees,
BaggingRegressor, StackingRegressor** — roughly a third of the sweep lost or hand-built.

### The cost nobody mentions

`sample_work.parquet` is 765K rows. On a single laptop, Spark on 765K rows is **slower**
than pandas — JVM startup, serialization and shuffle are pure overhead with no cluster to
amortize against. And the 110 unit tests currently run in milliseconds because they are
pure functions over small frames; every one would need a `SparkSession` fixture and go to
seconds. The TDD loop that carried Phases 1–3 gets materially worse.

### Decisions

- **D1 — ADOPTED: move feature engineering into the Spark prep.** Port the `features.py`
  column logic into `00_prep_spark.py` so it runs on all 128.78M rows **before** sampling,
  not on the sample after. This is the one conversion that plays to Spark's strength: the
  samples then carry engineered features, and training on far more than 765K rows stays
  open. Phase 2's split/encoding discipline is unaffected — only the deterministic column
  math moves upstream.
- **D2 — ADOPTED: `preprocess.py`, `evaluate.py` and the Phase-4 sweep stay scikit-learn.**
  Not because Spark can't, but because at 765K rows it is slower, it costs `TargetEncoder`
  and a third of the model families, and it buys nothing demonstrable. §0's "Spark ends
  before modeling starts" stands.
- **D3 — ADOPTED (2026-07-30): one MLlib baseline on `sample_full`.** A single Spark-native
  model alongside the sklearn sweep — the skill demonstrated at a data size where Spark is
  genuinely the right tool, without rewriting three working modules. Specified as **Phase 4b**
  in §5b below.

**Naming note:** `spark/` is the batch-processing + ML home (inherited from Phase 3's
rename of `05_batch_processing/`), not a claim that everything inside is Spark. Renaming
now would churn Docker mounts and import paths for cosmetics — not worth it.

---

## Status
- [x] Plan reviewed by Sagar
- [x] **Spark-vs-sklearn scope decision — §10, 2026-07-30.** D1 adopted (feature
      engineering moves into the Spark prep, pre-sample), D2 adopted (preprocessing, CV
      harness and sweep stay sklearn), **D3 adopted** — single MLlib GBT baseline on
      `sample_full`, specified as Phase 4b in §5b. Supersedes nothing in §0–§9 except the
      placement of `features.py` logic; no phase is invalidated.
- [~] Phase 0: Spark verified (4.1.2 on Py3.13/Java21) + FHVHV disk cleanup **pending user `rm`**; `spark/data/` (2.4G redundant) pending user OK
- [x] **Phase 1: prep script + samples — DONE 2026-07-10.** `spark/ml/00_prep_spark.py` →
      `sample_full.parquet` (12,748,027 rows) + `sample_work.parquet` (765,761) + `prep_stats.json`.
      Verified: raw count 128,781,646 (exact), caps match dashboard (Yellow $52/18.7mi, Green $45/14.15mi),
      avg fares $12.87/$12.02, multi-passenger 26.5% (exact), no nulls, OD cardinality 19,953 zone pairs.
      NOTE: optional climate cols (humidity/windSpeed/visibility) are Decimal→object; cast or drop in Phase 2.
- [x] Phase 2: features + leakage-safe preprocessing — **features.py DONE 2026-07-10 (TDD)**.
      `spark/ml/src/features.py` + `tests/unit/ml/test_features.py` (24 tests, RED→GREEN).
      Decimal→float64 cast locked (humidity/windSpeed/visibility kept as features);
      cyclic hour, temp_band ordinal, od_corridor key, leakage exclusions (incl.
      `fare_amount`, `distance_capped`, `ratecodeid`, raw zones), duration ablation flag.
      Real-data smoke on sample_work: X=(765761, 16), 0 nulls, od cardinality 19,953 ✓.
      **preprocess.py DONE 2026-07-10 (TDD, 17 tests):** scaled + tree ColumnTransformer
      variants; OHE (`handle_unknown="ignore"`); od_corridor via **sklearn TargetEncoder**
      (built-in cross-fitted `fit_transform` — no category_encoders dep needed); column
      lists auto-adapt to the duration ablation. Real-data smoke: both variants →
      (765761, 27) float64, finite, ~1s each; te__od_corridor spans $0–$52 (= yellow cap).
      **Phase 2 COMPLETE** — 94 unit tests green.
- [x] Phase 3: evaluation harness — **DONE 2026-07-10 (TDD, 16 tests).**
      `spark/ml/src/evaluate.py`: `mae/rmse/r2` + `compute_metrics` (single source),
      `make_cv` (KFold shuffle seed=42, reproducibility tested), `evaluate()` →
      leaderboard row (negated-scorer flip tested), `leaderboard()` sorted by RMSE.
      110 unit tests green. Smoke leaderboard (100K rows, 3-fold, duration ON):
      tree_d8 MAE $0.51 / R² .981 · ridge $0.84 / .937 · dummy $6.73 / 0 —
      matches plan §5 prediction (tree > linear ≫ mean). High R² expected with
      duration in (metered fare ≈ f(distance, time)) → ablation will quantify it.
- [~] Phase 4: model sweep — registry + runner built; **first work-sized sweep INVALID**,
      superseded by the 2026-08-01 feature fix below. Re-run pending.
- [x] **Feature-set + split correction — 2026-08-01 (TDD).** The 2026-07-31 work sweep
      returned R² −1,192,374 for `ols`/`ridge` (and −36K / −132K for `elasticnet` /
      `voting`). Root cause: `sample_work` row 533491 carries a corrupt
      **8,003,318-mile** odometer reading ($15.50 fare, 22 min). `features.py` had
      excluded `distance_capped` as a "duplicate" and kept the uncapped
      `trip_distance`, disabling the §2 p99 cap built to neutralise exactly this. With
      KFold(seed=42) the row lands in **fold 0's test split**, so the scaler was fit
      without it (σ=3.59 vs 10,222) and the row transformed to 2,228,170 → a **$9.29M
      prediction**; that single row was 100.00% of fold 0's squared error. Trees were
      immune (split-based), which is why the leaderboard top looked healthy.
      **Fixed:** only the derived half of each raw/derived pair now trains —
      `distance_capped` replaces `trip_distance`, `temp_band_ord` replaces
      `temperature`. X: 16 → **15 features**, matrix 27 → **26 columns**, scaled
      `max|x|` 2,228,170 → **52**. Smoke: `ols`/`ridge` R² **0.969**.
      **Added:** §4a split policy — `evaluate.make_holdout` + `--holdout-frac` /
      `--no-holdout`. 208 unit tests green.
      **Rank deficiency — closed 2026-09-01, and it closed earlier than anyone
      recorded.** The line here used to read "known, untouched: rank 25/27, cond ≈ 4e15,
      coefficients unstable". `fc87020` fixed it on 2026-08-01 by dropping a reference
      level per categorical on the scaled variant, but neither this Status nor the
      2026-08-22 audit (item 12) was updated, so a solved defect stayed on two open lists
      for a month. Re-measured on the 612,608-row train split:

      | variant | shape | rank | with intercept | cond (with intercept) |
      |---|---|---|---|---|
      | `scaled` (linear/NN) | 612,608 × 23 | **23 / 23** | **24 / 24** | **6.97e+03** |
      | `tree` (GBT et al.) | 612,608 × 26 | 24 / 26 | 24 / 27 | 2.95e+16 |

      The `scaled` matrix is full rank, so the dummy trap is gone and linear coefficients
      are interpretable again. The `tree` matrix is still deficient by exactly the
      predicted amount — three full one-hot blocks each summing to 1 give two
      dependencies, three against an intercept — and that is **by design, not a defect**:
      trees split rather than invert, keeping every level costs them nothing, and dropping
      one would only make that level harder to split on.
- [x] **Phase 4b: Spark MLlib GBT baseline (§5b) — COMPLETE 2026-09-01.** All three rows
      are in. Helpers `spark/ml/src/mllib.py` + `tests/unit/ml/test_mllib.py` (37 tests)
      and the script `01_mllib_baseline.py`, both on the `sample_work` train split (612,608
      rows — the sweep's own rows), 5 folds, GBT maxIter=100 maxDepth=5.
      - **Row 3, 2026-08-08** — `mllib_gbt_nocorr@work612k`: MAE $0.483 / RMSE 1.258 /
        R² .9833, 51.4s per fold. Tagged `mllib_gbt` at the time, in the belief that it was
        the baseline; it dropped `od_corridor` on a premise the 2026-08-09 correction
        overturned, so it is the corridor **ablation**. Renamed 2026-09-01.
      - **Row 2, 2026-09-01** — `mllib_gbt@work612k`: `StringIndexer` + `TargetEncoder`
        (`targetType="continuous"`, `handleInvalid="keep"`, smoothing 5) on `od_corridor`.
        MAE $0.520 / RMSE 1.450 / R² .9778, 170.5s per fold, 989s for the run.
      - **The result is a negative one, and it replicates.** Restoring the corridor made
        MLlib *worse* than dropping it, and no smoothing value tested (0.067 … 500 across
        seven arms) beat the ablation. §5b carries the table and the reading: the one
        capability MLlib lacks — a cross-fitted target encoder — costs more than the
        feature it encodes.
      - Row 1 stands at MAE $0.350 / 1.0s per fold, so the stack gap on identical rows is
        **1.5x accuracy and 167x wall time**, both in sklearn's favour.
      - Two Spark 4.1.2 facts landed in the code as a result: `targetType` defaults to
        `"binary"` and must be set for a dollar target, and `TargetEncoderModel` copies the
        indexer's nominal metadata onto its numeric output, which makes `VectorAssembler`
        call the feature categorical and GBT fail on `maxBins`. **Both re-measured on
        Spark 4.0.1 on 2026-09-02** and both still hold, so neither is a 4.1-only quirk;
        the explicit `targetType="continuous"` and the demoting `SQLTransformer` stay
        mandatory on the Dataproc Serverless runtime too. See the migration plan 2.4.
- [x] **Baseline set from the D-012 cloud prep — 2026-09-13.** The samples now come from
      `prep-m4-20260913-full` on the rebuilt `dbt_prod.fact_trips`, not the pre-D-012
      local backup. `sample_work.parquet` and `prep_stats.json` came down from
      `gs://primary-data-dtc-506916/ml/prep/`, byte- and CRC32C-equal to the objects.
      - `prep_stats.json`: raw 307,339,039, guarded 304,766,876, `sample_work` 1,828,181,
        `sample_full` 30,482,494. Caps: Yellow $52.0 / 18.5 mi, Green $44.5 / 13.9 mi,
        duration 57.5 min — each inside Q1's exact p98.9–p99.1 band (option A).
      - Split: temporal 133,629 / holdout 338,911 / train 1,355,641 (§4a). The new
        `sample_work_train.parquet` holds the 1,355,641. No holdout or temporal metric
        was computed.
      - `00_prep_spark.py` now requires `--source` (`b4e65e3`).
      - The pre-D-012 samples are archived, not deleted, in
        `../nyc_taxi_migration_backup/prep-pre-d012-local/`: 26 files, 309,398,759 bytes,
        SHA-256-equal after the move.
      - Every result in the §5b table and in the rows above is pre-D-012 and does not
        compare across this line (D-012). Next: the M4 smoke.
- [x] **The MLlib baseline on the D-012 split, local and cloud (M4 smoke) — 2026-09-15.**
      `mllib_gbt@work1355k`: the §5b row 2 configuration (5 folds, seed 42, GBT maxIter=100
      maxDepth=5, `TargetEncoder` smoothing 5) on the 1,355,641-row train split.
      - MAE $0.466188 ± 0.005356, RMSE 1.201377 ± 0.021579, R² 0.984675 ± 0.000561.
      - Local, 2026-09-14: dev container, Spark 4.0.1, `local[8]`, `default_parallelism` 8,
        2,961.4 s. Cloud, 2026-09-15: Dataproc Serverless batch `m4-smoke-20260915`, master
        `dataproc`, 2 executors, `default_parallelism` 8, 4,301.8 s, $1.12. The two agree to
        floating-point precision — the migration plan M4 carries the gate and the costs.
      - This is the first post-D-012 MLlib row. It does not compare with the pre-D-012
        rows in §5b, and no holdout or temporal metric was computed.
      - Next: §5.3's out-of-fold encoder (migration plan §5.3).
- [x] **The mechanism test: row 3 and the out-of-fold row on the D-012 split — 2026-09-15.**
      `src/oof_encode.py` (`8f67d17`) and `01_mllib_baseline.py --oof` (`36cedd5`), test
      first. Local, dev container, Spark 4.0.1, `local[8]`, the same folds as row 2.
      - Row 3, `mllib_gbt_nocorr@work1355k`: MAE 0.453134 ± 0.003220, RMSE 1.138785 ±
        0.006900, R² 0.986234 ± 0.000206, 2,055.2 s.
      - OOF, `mllib_gbt_oof@work1355k`: MAE 0.462342 ± 0.000505, RMSE 1.161502 ± 0.011404,
        R² 0.985679 ± 0.000304, 2,886.7 s.
      - The corridor stays net-negative in MLlib even cross-fitted. On MAE the OOF row is
        inside row 2's fold std; on RMSE and R² it sits materially between rows 2 and 3.
        §5b's D-012 table carries the reading.
      - Next: the owner's §5.4 decision (migration plan §5.4).
- [x] **Phase 4 re-run on the D-012 split — 2026-09-15.** This closes the "re-run pending"
      above. `01_run_sweep.py --tag work_d012` in the dev container: 14 models, 5 folds,
      1,355,641 train rows, holdout 338,911 and temporal 133,629 both SEALED and unscored,
      7,468.0 s. python 3.12.13, numpy 2.4.6, sklearn 1.7.1, lightgbm 4.7.0, xgboost 3.3.0,
      catboost 1.2.10.
      - Row 1, `lightgbm`: MAE 0.331289 ± 0.001310, RMSE 0.936259 ± 0.029962, R² 0.990686 ±
        0.000610, 5.1 s per fold. Against row 2 it is 1.41x more accurate and 105x faster.
      - Top 4 by RMSE: `catboost` 0.928765, `lightgbm` 0.936259, `stacking` 0.937680,
        `extra_trees` 0.947680 — the same four as the voided list, first two swapped, their
        gap inside the fold std. §5c's scope is re-derived in §5c.
      - The owner read §5.4 on MAE, so D2 stands as **D-013**.
      - Next: re-scope §5c for the 30,482,494-row `sample_full` with a lightgbm probe.
- [x] **Phase 5, tuning and diagnostics on the work split — 2026-09-16/17.** The TUNE goal;
      full record in §6, "The TUNE results". `lightgbm` and `catboost` tuned with Optuna,
      5 folds, `work_d012`'s 1,355,641-row train split, holdout and temporal SEALED and
      unscored throughout. Tuning **widens lightgbm's lead**: untuned it already beat
      untuned catboost by MAE (D-013); tuned, lightgbm improves to 0.312976 ± 0.001181
      (from 0.331289) while catboost barely moves, 0.319282 ± 0.001334 (from 0.320963) —
      catboost had less room left to tune. **Champion: `lightgbm`**, by the owner's rule
      (gap 0.006306 outside catboost's std). Diagnosed on out-of-fold predictions: slice
      MAE by borough/temp-band/hour, mean |SHAP| (`distance_capped` and
      `trip_duration_min` dominate), and the duration ablation (+0.912886 MAE without it).
      CatBoost's tuning needed a short-lived GCE VM after the laptop froze three times —
      $0.266, VM deleted after. Next: §5c at scale, then Phase 5's holdout score.
- [ ] **Cloud full-scale run (§5c, decided 2026-08-04).** `sample_full` (12.75M; 30,482,494 since the 2026-09-13 baseline) for
      **both** sklearn and MLlib. Local machine measured at 4.8 GB frame / ~8.2 h for the
      whole sweep on an 18 GiB M3 Pro; scope the cloud run to the top 4 + corridor-dropped
      champion + MLlib GBT rather than all 14. **Rescoped 2026-09-16 (owner):** the champion
      (`lightgbm`, tuned) plus the MLlib row-3 run only — `stacking`/`extra_trees` dropped.
      Both train on `sample_full`, which contains the sealed rows
      (`00_prep_spark.py:334`), so their numbers are **CV only** and never sit beside the
      Phase 5 holdout score.
- [~] Phase 5: tune + diagnose. **Tuning and diagnostics done 2026-09-16/17** on the work
      split, above. The sealed holdout is still unscored — it is scored **once**, after
      §5c refits the champion on `sample_full` (§4a).
- [ ] Phase 6: neural nets — **moved after the end of the project (D-014, 2026-09-16).**
