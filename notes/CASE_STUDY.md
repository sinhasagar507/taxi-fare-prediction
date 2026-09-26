# NYC Taxi Fare Prediction: End-to-End Case Study

**Invoke when:** writing up the project for a resume, a portfolio page or an interview, or
checking a public claim about it. Every number here cites a committed note; re-measure
before you change one (D-009).

This project predicts the fare of a New York City taxi trip from what is known when the
trip is booked. It is also the data platform behind that model: two years of public trip
records flow from the city's open data site into Google Cloud Storage, BigQuery and dbt,
and out to a model trained with scikit-learn and LightGBM, with Spark doing the heavy data
preparation. The target started as trip duration and moved to fare (D-001).

**Stack:** Google Cloud Storage, BigQuery, dbt Core, Apache Airflow, PySpark and Dataproc
Serverless, scikit-learn, LightGBM, Docker, Terraform, GitHub Actions, pytest.

**Sources cited below:** [MP] `notes/2026-09-02-gcp-cloud-migration-plan.md`,
[ML] `spark/2026-07-10-fare-prediction-modeling-plan.md`, [DR] `notes/decisions.md`,
[CB] `notes/2026-09-04-cloud-cost-baseline.md`.

---

## 1. Business Use-Case

### The problem
A rider, a dispatcher or a pricing analyst wants to know what a trip will cost before it
starts. The city publishes every trip, but the raw files are large, messy and split by
service and month. A usable estimate needs a clean, trustworthy history first.

### Who it serves
A personal portfolio project, built by one person. The stand-in users are a trip-planning
app that shows an upfront fare, and an analyst who reads the same tables in a dashboard.

### What success means
A fare estimate that is off by cents, not dollars, on trips the model has never seen, and a
pipeline that rebuilds its own training data from the public source on demand.

### Non-goals
No real-time serving, no live pricing, and no neural network. The neural-network phase is
parked until after the project ends (D-014 [DR]).

## 2. North-Star Metric

### The metric
**Mean absolute error (MAE) in dollars on a sealed holdout**, scored exactly once. MAE
reads directly as "cents off per trip", which suits a fare. RMSE and R² are reported
beside it.

### Why a sealed holdout
Choosing a winner on the same rows used to judge it makes the final number look better
than it is. So 20% of the work sample's pre-cutoff rows were set aside before any model was
compared, and a
guard in code refuses to score it a second time [ML §4a, §6].

### The baseline
The first Spark baseline, a gradient-boosted tree in Spark MLlib, reached a cross-validated
MAE of **$0.466** on the 1,355,641 training rows [DR D-013]. A mean-only predictor was not
re-scored on the corrected data, so no "no-model" baseline is quoted here.

## 3. KPIs, Dimensions and Datasets

### Supporting KPIs
Fold-to-fold spread of the CV score, fit time per fold, and dollars per cloud run.

### Dimensions
Pickup borough, temperature band and pickup hour. The champion's error is lowest in
Manhattan, **$0.282** over 1,153,561 rows, and rises mildly with temperature [ML §6, TUNE
results].

### Guardrails
Cloud spend per run, a cap on how long a cloud machine may live, and no service-account
keyfile on any cloud machine or image.

### Datasets
- TLC yellow and green trip records, 2015-01 to 2016-12: 50 files, 4.10 GiB, **312,790,342**
  trip rows in the bucket [MP Status, M2].
- Daily NYC weather, joined on the pickup date, and the taxi-zone lookup, loaded as a dbt
  seed.
- The dbt fact table `fact_trips`: **307,339,039** rows (yellow 271,905,544, green
  35,433,495), 130.81 GiB [DR D-012].

### The data-quality defect that mattered most
The staging models removed "duplicates" keyed on vendor and pickup second. That key names a
busy second, not a trip: it discarded **63.09%** of real yellow trips, while only **270**
rows in 277,171,036 were true duplicates [DR D-012]. The survivors were biased short. Fixing
it meant a full rebuild and re-running every model result.

## 4. Methodology

### From raw data to a sample
Spark reads the 307,339,039 fact rows, applies range checks and caps each skewed column at
its 99th percentile, keeping **304,766,876** rows [MP Status, M4]. It draws a 10% stratified
sample of 30,482,494 rows and a 1,828,181-row work sample [MP Status, §5c; ML §1].

### The split
The work sample splits three ways: 133,629 rows from the last two months form a **temporal
set**, then 338,911 rows form a **random holdout**, and 1,355,641 rows train [ML §4a].

### Features
Capped distance, the trip-duration estimate, cyclic pickup hour, temperature band, borough,
service type, an airport flag and the pickup-to-dropoff zone pair. The zone pair has
thousands of values, so it is replaced by the average fare of that pair, computed
out-of-fold so no trip sees its own fare.

### Models compared
A 14-model sweep on identical folds, from linear models to stacking [MP Status, §5c]. The top
two were tuned:

- **LightGBM**, 28 trials: CV MAE **$0.312976 ± 0.001181**.
- **CatBoost**, 7 trials: CV MAE $0.319282 ± 0.001334 [ML §6, TUNE results].

LightGBM won by more than CatBoost's own fold spread and fits 4.7x faster.

### Alternatives rejected
- **Spark MLlib as the champion track.** Its tree lost to LightGBM by 1.53x on the same 22.6
  million rows [ML §6, RESCALE]. Spark stays for data preparation.
- **The zone-pair feature in MLlib.** Even computed out-of-fold, it made MLlib worse on
  every metric, so MLlib drops it [DR D-013].

## 5. Design Architecture

### Pipeline
```text
TLC files -> Airflow ingest DAGs -> GCS bucket
          -> Airflow external-table DAGs -> BigQuery external tables
          -> dbt staging + marts (Airflow or CLI) -> fact_trips
          -> Spark prep (Dataproc Serverless or local) -> samples in GCS
          -> scikit-learn / LightGBM in a Docker dev container -> scored model
```

### Why these tools
- **External tables:** BigQuery reads the files where they sit; nothing is stored twice.
- **dbt:** tested SQL models with separate dev, CI and prod targets; CI never writes prod.
- **Dataproc Serverless:** billed per second, scales to zero. Spark is pinned to the cloud
  runtime's version after a bit-identical parity run [DR D-011].
- **One dev container** holds the whole ML stack; the Airflow stack is a separate image.

### Serving and monitoring
The Airflow stack also ran on a cloud VM with no keyfile: the zone ingest DAG and its
external-table DAG both succeeded, in 15 minutes of VM time for about $0.02 [MP M5].
Batch only. The pair of holdout scores below is the first drift check. Scheduled scoring
and retraining are not built.

### Tests that earned their place
- A test fails on two deliberately leaky versions of the zone-pair encoder. It guards
  against a trip's own fare leaking into its feature [DR D-013].
- A test caught a rounding error when Arrow turned decimal weather columns into floats:
  0.123456789 came back one digit off [ML §6, RESCALE].
- The holdout guard's tests prove a second scoring run is refused, so the sealed rows
  stay sealed [ML §6].
- Guard tests fail if a hard-coded keyfile path or a retired project ID comes back.

## 6. Results and Impact

### The one score
The tuned LightGBM, fitted once on the 1,355,641 training rows [ML §6, "The HOLDOUT result"]:

| Set | Rows | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| CV, 5 folds (context) | 1,355,641 | 0.312976 | 0.924104 | 0.990926 |
| **Random holdout**, 2015-01 to 2016-10 | 338,911 | **0.311353** | 0.910174 | 0.991193 |
| **Temporal set**, 2016-11 to 2016-12 | 133,629 | **0.316332** | 0.924683 | 0.991211 |

The model is off by about 31 cents per trip on unseen trips. Against the MLlib baseline's
CV MAE of $0.466, that is 33% less error. On the next two months it is about half a cent worse than
on random trips from its own period, and December is no worse than November. The CV
estimate was not optimistic.

### At scale
On 22,603,549 training rows, LightGBM's CV MAE fell to **$0.304570**, against **$0.465098**
for MLlib on the same rows. The run cost **$2.97** in total [ML §6, RESCALE]. These are CV
numbers only; the holdout above is the headline.

### Cost discipline
The first 30 days of cloud work measured about **$20.7**, 99% of it BigQuery queries,
before any optimisation [CB]. That figure is the "before" for any saving claimed later.

## 7. Risks, Learnings and Future Work

### Risks met
- **A silent row loss.** The dedup defect in section 3. The fix brought the yellow
  distance cap back to 18.51 miles, against 18.67 in the raw source [DR D-012].
- **A corrupt odometer row.** Keeping raw distance disabled the cap, and one 8,003,318-mile
  row drove a linear model to a **$9.29 million** prediction [ML §3].
- **A cloud machine left running.** One VM ran 204 minutes for a 43-minute job. Every VM
  since is created with a hard run limit [ML §6, RESCALE].

### Learnings
Measure a number when you write it down; four defects here came from trusted, unchecked
statements (D-009 [DR]). A leaked key was also purged from git history, and keys now live
only in a gitignored folder.

### Future work
- Finish the dashboard on the corrected tables.
- Train the parked neural network with zone-pair embeddings (D-014).
- Add road distance from an OSRM routing matrix as a feature.

---

## Resume version

- Built an end-to-end pipeline on Google Cloud (GCS, BigQuery, dbt, Airflow, Dataproc) over
  312.8 million NYC taxi trips, and a LightGBM fare model with a $0.311 MAE on a sealed
  holdout, R² 0.991.
- Found and fixed a dbt dedup key that silently dropped 63% of yellow trips; rebuilt the
  307.3 million-row fact table and re-ran every model on the corrected data.
- Compared 14 models on identical folds and tuned the top two; LightGBM beat Spark MLlib by
  1.53x on 22.6 million rows for $2.97 of cloud spend.
- Scored the holdout exactly once behind a guard in code, and checked drift on a later
  two-month set: $0.316 MAE, half a cent from the holdout.
