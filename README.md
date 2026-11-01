# Fraud detection pipeline

An end-to-end fraud detection system: 30M synthetic card transactions, window-SQL feature engineering in DuckDB,
a Random Forest, a serverless **AWS SageMaker** endpoint with a durable per-prediction audit log, **CloudWatch**
monitoring with PSI drift detection, a manager report generator, and an interactive demo.

> **Honest scope.** The data is **synthetic** (a generator with injected fraud patterns), so the numbers show that
> the pipeline works, not what a bank would see. Scoring is **real time** (a live API, about 70 ms average), but
> feature computation is **batch**: this is not a streaming pipeline (see [Roadmap](#roadmap)). The approach was also run on 590k **real** transactions; see [Validation on real data](#validation-on-real-data-ieee-cis), where the numbers are far lower.

## Try it yourself (2 minutes, no AWS account needed)

Enter a transaction's details and see whether the model flags it as fraud. The trained model ships in `models/demo/`,
so there is nothing to train or download. You need Python 3.12 or newer (developed on 3.14).

```bash
git clone https://github.com/Rajdeep7102/fraud-detection-pipeline.git   # or fork it first
cd fraud-detection-pipeline
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS / Linux: source .venv/bin/activate)
pip install -r requirements-demo.txt
streamlit run app/demo.py
```

A browser tab opens. On **Try it**, pick a scenario (everyday purchase, card testing burst, account takeover, genuine
holiday abroad, stealth fraud) or change any of the 12 inputs, then press **Score this transaction**. You get a
verdict, a score, and a chart of which inputs pushed it. The other tabs show how well the model performs, and generate
the one-page manager report (**Manager report** tab, or `python -m fraud.reporting.report --offline`).

Run the tests with `pytest` (the ones that need the 30M-row dataset skip themselves).

**Optional: use your own AWS account.** Everything above runs locally. To score through a real SageMaker endpoint and
see the monitoring tab, put your credentials in place (`aws configure`, or the usual `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` environment variables), then run `python deploy/deploy.py` (about 10 minutes; it creates an S3
bucket, ECR repository, CodeBuild project, two IAM roles and a serverless endpoint) and, for the dashboard and alarms,
`python monitoring/setup_monitoring.py`. In the demo choose **Live SageMaker endpoint**. Idle cost is cents; remove
everything with `python deploy/teardown.py --all`. The model file is a scikit-learn pickle: only load pickles you
trust (this one is produced by `fraud.train.train` in this repo).

## Results (held-out test, 6.66M transactions, never used in training or threshold tuning)

| Metric | Value |
| --- | --- |
| Fraud caught (recall) | **79.3%** (95% CI 78.7 to 80.0) |
| Alerts that are real fraud (precision) | **91.8%** (95% CI 91.4 to 92.2) |
| False alarms | **1.6 per 10,000 transactions** |
| PR-AUC / ROC-AUC | 0.877 / 0.994 |
| Average scoring time on the endpoint | about 70 ms |

| Fraud type | Share of fraud | Caught |
| --- | --- | --- |
| Card testing | 72% | 94.6% |
| Account takeover | 21% | 51.4% |
| Stealth fraud | 7% | 0.1% |

The split is by time (train days 1-63, threshold tuned on days 64-70, test days 71-90). The model is weak on stealth
fraud and half-blind on account takeover, which is what the 12 inputs can express; richer signals are on the roadmap.

## Architecture

```text
 laptop                                              AWS (us-east-1)
 ──────                                              ───────────────
 generator ─► Parquet ─► DuckDB window SQL ─► Random Forest ─► model.tar.gz ─► S3
 (30M rows)   (32 card    (12 features)       (time-based                         │
              buckets)                         split)            ECR image ◄─ CodeBuild
                                                                     │
 replay_traffic.py ──► SageMaker serverless endpoint (Flask + gunicorn container)
                              │ one put_log_events per request, before responding
                              ▼
                       CloudWatch Logs (audit) ─► metric filters ─► dashboard + 5 alarms ─► SNS
                              └─► Logs Insights ─► drift_job.py (PSI per feature) ─► metrics
```

## The 12 features (DuckDB window SQL, per card, backwards-looking only)

Velocity: purchases in 1h and 24h, spend in 24h, seconds since last purchase, distinct merchants in 24h, distinct
countries in 7d. Behavioural: amount vs the card's 30-day average and z-score, first time at merchant, first time
on device, foreign country, hours away from the usual time of day. See `src/fraud/features/sql/features.sql`.

## Engineering notes (things that went wrong, and the fixes)

- **The audit log was silently lossy.** Printing audit lines to stdout lost about 9% of them on a serverless
  endpoint (containers freeze after responding; the platform log forwarder is best-effort). The container now writes
  each request's records to CloudWatch Logs in one `put_log_events` call before it responds; 5,000 of 5,000 replayed
  predictions arrived afterwards.
- **Drift detection was blind to binary features.** Quantile bins put every 0/1 value in one bin, so PSI was always 0
  for `is_foreign_country`. Low-cardinality features now get one bin per value; a regression test covers it.
- **Time-based split, not random.** A random split leaks a card's future behaviour into training.
- **Class imbalance:** fraud is about 0.3% of rows, so legit rows are down-sampled for training and the threshold is
  chosen on an untouched validation period. The score is therefore a ranking, not a calibrated probability.
- **Precision depends on prevalence.** Fraud was rarer in the test period (0.23%) than in validation (0.38%), which
  lowered precision for the same rule.
- **No local Docker needed:** images are built in CodeBuild. Deploys are idempotent; a model is immutable in SageMaker,
  so model names carry the image tag.

## Validation on real data (IEEE-CIS)

The synthetic results show the pipeline works; they say little about real fraud. So the same approach was run on the
public [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) data (590,540 real card-not-present
transactions, 3.5% fraud). The data is not redistributed here: download it from Kaggle into `data/real/` and see
`src/fraud/real/`. Same discipline as before: time-ordered split (train days 1-101, validation 101-130, test 130-183),
thresholds chosen on validation only, 95% bootstrap intervals, resampling applied to training rows only.

| Feature set (Random Forest, test period, 148k transactions) | ROC-AUC | PR-AUC (no skill = 0.035) | Flagging ~1% of transactions: precision / recall |
| --- | --- | --- | --- |
| A. The 12 velocity and behavioural clues, adapted to this data | 0.757 (0.749 to 0.762) | 0.115 | 21% / 4% |
| B. The columns the dataset provides (amount, card, address, C1-14, D1-15, M1-9) | 0.866 (0.861 to 0.871) | 0.419 (0.406 to 0.431) | 85% / 16% |
| C. Both together | 0.866 (0.860 to 0.870) | 0.419 | 85% / 16% |

What this shows:

- **The synthetic numbers do not transfer.** On real data the velocity clues are weak (ROC-AUC 0.76); the dataset's own
  columns carry most of the signal (0.87), and adding the clues on top changed nothing. Choosing the alert threshold on
  validation for 80% precision gave 77% precision and 24% recall for model B.
- **SMOTE hurt.** Compared on model A: no resampling ROC-AUC 0.771, class weights 0.757, random undersampling 0.773,
  SMOTE 0.712 (PR-AUC 0.108 / 0.115 / 0.121 / 0.092). The first three are close; SMOTE is clearly worse. It was applied
  inside an `imblearn` pipeline so it never touched validation or test rows.
- **Caveats.** IEEE-CIS has no card id, so a card is approximated from card, address and first-transaction-day fields;
  22% of rows are the first of their key. No hyperparameter tuning, no gradient boosting, and the 339 anonymised `V`
  columns and identity fields were not used, so competition-grade solutions score higher. The deployed endpoint serves
  the synthetic-trained model; this real-data track is an evaluation, not a deployment.

```bash
python -m fraud.real.prepare      # CSV -> Parquet (about 10 min)
python -m fraud.real.features     # the 12 clues on real data (seconds)
python -m fraud.real.experiment   # train, compare, write reports/real/ (about 20 min)
```

## Run the full pipeline yourself (optional, about an hour)

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt && .venv/Scripts/python -m pip install -e .
python -m fraud.data.generate          # 30M rows to data/raw  (about 3 min)
python -m fraud.features.build         # features for all rows (about 7 min)
python -m fraud.train.train            # fit, choose threshold on validation, write artifacts/ (16-26 min)
python -m fraud.train.evaluate         # score the untouched test days -> reports/
python -m pytest                       # unit and contract tests

python deploy/deploy.py                # S3, ECR, CodeBuild image, IAM, SageMaker serverless endpoint (uses artifacts/ if you trained, else models/demo/)
python monitoring/setup_monitoring.py  # metric filters, 5 alarms, SNS topic, dashboard
python monitoring/replay_traffic.py --rows 5000 [--drift]
python monitoring/drift_job.py --minutes 60 [--request-id-prefix replay-drift]

python -m fraud.reporting.report       # manager report -> reports/manager_report.html (add --offline without AWS)
streamlit run app/demo.py              # interactive demo (binds to localhost only)
python deploy/teardown.py [--all]      # remove the endpoint (and with --all, every resource created)
```

Needs Python 3.12+ (developed on 3.14) and AWS credentials with rights to SageMaker, ECR, S3, IAM, CodeBuild,
CloudWatch and SNS. Idle cost is cents (serverless endpoint scales to zero); CloudWatch custom metrics are about
$0.30 each per month, so tear down after demos.

## Layout

```text
models/demo           ready-made model (used by the demo)   src/fraud/reporting   KPIs, live health, HTML report
src/fraud/data        synthetic generator          src/fraud/real        IEEE-CIS real-data validation
src/fraud/features    SQL + feature spec           src/fraud/monitoring  PSI drift helpers
src/fraud/train       train / evaluate / baseline  serving/              Flask app, audit sink, Dockerfile
deploy/               deploy + teardown            monitoring/           setup, drift job, traffic replay
app/demo.py           Streamlit demo               tests/                generator, features, serving, audit, drift, report
```

## Roadmap

1. **Streaming:** events through Kinesis, a Lambda that maintains per-card state in DynamoDB and computes the same
   features online, scoring on the endpoint, and per-transaction alerts to SNS or EventBridge. Needs a parity test
   against the DuckDB features to avoid training-serving skew.
2. **Richer signals:** account history (account age, prior disputes) and geography (distance from home, impossible
   travel speed between consecutive purchases).
3. Scheduled drift job (EventBridge + Lambda), retraining pipeline, infrastructure as code (CDK or Terraform),
   model card and runbook.
