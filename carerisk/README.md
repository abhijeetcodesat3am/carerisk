# CareRisk: 30-Day Readmission Prediction

An end-to-end pipeline in PySpark and Databricks that predicts whether a Medicare beneficiary will be readmitted within 30 days of a hospital discharge. It uses the CMS DE-SynPUF synthetic claims, tracks experiments in MLflow, explains each prediction with SHAP, audits fairness, and runs batch scoring with PSI drift monitoring that feeds a Power BI dashboard.

```
 SynPUF CSVs ──► 01 ingest ──► silver_* ──► 02 features ──► gold_features ──► 03 train ──► MLflow registry
 (Volume)        bronze→silver   claims,       cohort, label,     (+ split)       LR / RF / XGB   @champion
                 transfer merge  episodes,     Charlson CCI,                      SHAP, fairness      │
                                 dx events     prior utilisation                                        ▼
                                                                  04 batch_score (weekly job) ◄────────┘
                                                                   mapInPandas scoring + SHAP reasons
                                                                   PSI drift vs training population
                                                                         │
                                                gold_risk_scores, gold_drift_metrics ──► Power BI
```

## Repository layout

| Path | What it does |
|---|---|
| `src/carerisk/ingest.py` | Reads the SynPUF beneficiary (2008–10), inpatient and outpatient files. Collapses claim segments and merges overlapping or same-day-transfer claims into **episodes**. |
| `src/carerisk/comorbidity.py` | **Charlson Comorbidity Index** from ICD-9 codes (Quan 2005 mapping) with the severity hierarchy (diabetes, liver, cancer). |
| `src/carerisk/features.py` | Index-stay cohort, 30-day label, exclusions, the 35 patient-level features, and the **patient-grouped, time-based split**. |
| `src/carerisk/train.py` | LR baseline vs Random Forest vs XGBoost. Nested MLflow runs, patient-clustered bootstrap CIs, champion registration. |
| `src/carerisk/explain.py` | SHAP global importance and **per-patient top-3 reason codes**. |
| `src/carerisk/fairness.py` | Subgroup audit by age band and sex: AUROC, TPR (equal opportunity), FPR, PPV, calibration, with flags. |
| `src/carerisk/score.py` | Champion batch scoring with `mapInPandas`, risk tiers, reasons, and drift. |
| `src/carerisk/drift.py` | **Population Stability Index** (quantile bins, categorical and missing handled). |
| `notebooks/0*.py` | Databricks notebook wrappers (one per job task). |
| `databricks.yml`, `resources/jobs.yml` | Asset Bundle with two jobs: `carerisk-training` and a weekly scheduled `carerisk-batch-scoring`. |
| `powerbi/POWERBI.md` | Dashboard connection, data model, DAX measures and page layout. |
| `scripts/` | Real-data downloader and a SynPUF-layout sample generator. |

## Methodology

**Cohort.** Every inpatient episode admitted from 2009-01-01 on is an index stay. The 2008 data is look-back only, so every index stay has a full 365 days of history. Exclusions:
- death in hospital
- death within 30 days with no readmission
- discharge after 2010-12-01, since 30-day follow-up is incomplete

**Label.** `readmit_30 = 1` if the next admission starts 1–30 days after discharge. Transfers were already merged into one episode, so they don't count as readmissions.

**Features.** Every feature uses only information known at discharge:
- **Demographics:** age, sex, race, ESRD.
- **Index stay:** length of stay, number of claims, diagnoses and procedures, log payment.
- **Prior utilisation:** admissions in the prior 365 and 180 days, inpatient days in the prior 365 days, days since the last discharge, outpatient visits in the prior 365 and 30 days.
- **Comorbidity:** the Charlson index and its key components from inpatient and outpatient diagnoses in the look-back window, plus 11 CMS chronic-condition flags and annual costs from the *previous* year's beneficiary file. The same-year file would leak the future.

**Leakage-safe split.** Each beneficiary is hashed into exactly one group: 70% train, 15% val, 15% test patients. Train and val use 2009 stays. Test uses only 2010 stays from test patients. That means the test set has **unseen patients in a future period**, and an assertion in `train.py` checks that no patient appears in two groups.

**Model selection.** Each config is fit on train and scored on val. The best config per model family is evaluated once on test. Test results include patient-clustered bootstrap 95% CIs and a paired bootstrap Δ vs the LR baseline. The champion is the family with the highest **val AUPRC** (AUPRC matters for a rare outcome). It is registered as `carerisk_readmission@champion`, with thresholds, features and metrics stored as version tags.

**Risk tiers.** Thresholds are set on validation by care-management capacity: High is the top 10% of scores and Medium the top 30%. The fairness audit uses the High threshold.

**Drift.** Each scoring run computes PSI per feature and for the model score against the training population. Thresholds: < 0.10 stable, 0.10–0.25 warn, ≥ 0.25 alert. Set `fail_on_alert` in the config to make the job fail on an alert.

## Run locally

Needs Java 17/21 and [uv](https://github.com/astral-sh/uv).

```bash
make setup        # .venv with pyspark, mlflow, xgboost, shap
make sample       # small dataset in the exact SynPUF layout (for a smoke test)
make all          # ingest → features → train → score
make test         # unit tests (label logic, transfer merge, Charlson, split, PSI, fairness)
make mlflow       # MLflow UI at http://localhost:5000
```

Then switch to the real CMS data, Sample 1 (about 120 MB zipped):

```bash
make clean && make data && make all
```

Outputs land in `reports/`: `benchmark.csv`, `test_curves.png`, `shap_summary.png`, `fairness_audit.csv/png` and `powerbi/*.csv`.

## Run on Databricks

1. Create a Unity Catalog schema and volume, for example `main.carerisk` and `/Volumes/main/carerisk/raw`, and upload the unzipped SynPUF CSVs. Adjust `conf/config.yaml → databricks` if your names differ.
2. Install the Databricks CLI, then deploy:
   ```bash
   databricks bundle deploy -t dev
   ```
3. Run training:
   ```bash
   databricks bundle run carerisk_training
   ```
4. The `carerisk-batch-scoring` job then runs every Monday at 06:00 UTC, or start it now with `databricks bundle run carerisk_batch_scoring`.
5. Connect Power BI to the `gold_risk_scores` and `gold_drift_metrics` tables (see `powerbi/POWERBI.md`).

The code detects Databricks through `DATABRICKS_RUNTIME_VERSION`. On Databricks it writes Delta tables in Unity Catalog, uses the `databricks-uc` model registry and the `/Shared/carerisk` experiment. Locally it writes Parquet and uses a SQLite MLflow store. **Free Edition / serverless:** delete the `job_clusters` blocks and `job_cluster_key` lines in `resources/jobs.yml` so tasks run on serverless compute. The notebooks already `%pip install` xgboost and shap.

## Results

See `reports/benchmark.csv` after a run. The table below comes from the generated sample data, not CMS data. Re-run on DE-SynPUF and replace it.

Smoke-test run on generated sample data (20k beneficiaries; test = 1,069 unseen-patient 2010 stays, 29.7% readmitted):

| Model | Val AUPRC | Test AUROC (95% CI) | Test AUPRC (95% CI) | ΔAUPRC vs LR (95% CI) |
|---|---|---|---|---|
| **XGBoost (champion)** | 0.576 | 0.786 (0.745–0.820) | 0.611 (0.517–0.689) | +0.009 (−0.014–0.030) |
| Random Forest | 0.566 | 0.784 (0.742–0.819) | 0.610 (0.512–0.688) | +0.008 (−0.013–0.026) |
| Logistic Regression (baseline) | 0.557 | 0.778 (0.734–0.815) | 0.602 (0.502–0.680) | — |

Observed readmission rate by tier in the scored batch: High 74%, Medium 48%, Low 17%. On sample data the tree models do not significantly beat the baseline (the paired CI includes 0). Report this honestly for the real data too.

## Limitations

- DE-SynPUF is synthetic: variables were perturbed independently to protect privacy, so the relationships between them are weaker than in real claims. Expect modest AUROC (roughly 0.6–0.7). Treat this as a methods and engineering project, not a clinical model.
- Planned readmissions (CMS algorithm) are not excluded, and SynPUF has no discharge disposition.
- Fairness is audited, not enforced. Flagged groups need investigation before any deployment.
