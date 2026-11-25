# CareRisk Power BI dashboard

The batch job writes two tables the report reads:

| Table | Grain | Key columns |
|---|---|---|
| `gold_risk_scores` | one row per scored discharge (latest batch) | `risk_score`, `risk_tier`, `risk_rank_pct`, `reason_1..3`, `age_band`, `sex`, `state_cd`, `discharge_dt`, `observed_readmit_30`, `model_version`, `scored_at` |
| `gold_drift_metrics` | one row per feature per scoring run (appended) | `run_ts`, `feature`, `psi`, `status`, `ref_mean`, `cur_mean`, `model_version` |

## 1. Connect

**Databricks (recommended):** Power BI Desktop → *Get data* → **Azure Databricks** (or **Databricks**) →
Server hostname and HTTP path from your SQL warehouse's *Connection details* → choose
`main.carerisk.gold_risk_scores` and `main.carerisk.gold_drift_metrics` → **DirectQuery** (always shows the latest batch).
Alternatively, in Databricks open the SQL warehouse → *Connection details* → **Partner Connect → Power BI** to download a ready `.pbids` file.

**Local run:** *Get data → Text/CSV* → `reports/powerbi/risk_scores.csv` and `reports/powerbi/drift_metrics.csv`.

Power BI Desktop is Windows-only. On a Mac, use a Windows VM or build the report in Power BI Service (app.powerbi.com) with the same connection.

## 2. Model

- Mark `discharge_dt` as a date, then add a Date table: `Date = CALENDAR(DATE(2008,1,1), DATE(2010,12,31))` and relate `Date[Date]` → `gold_risk_scores[discharge_dt]`.
- Sort `risk_tier` by a helper column: `Tier Order = SWITCH([risk_tier], "High", 1, "Medium", 2, 3)`.
- `gold_drift_metrics` stands alone (no relationship needed).

## 3. DAX measures

```DAX
Patients Scored      = DISTINCTCOUNT(gold_risk_scores[DESYNPUF_ID])
Discharges Scored    = COUNTROWS(gold_risk_scores)
High Risk Count      = CALCULATE([Discharges Scored], gold_risk_scores[risk_tier] = "High")
High Risk %          = DIVIDE([High Risk Count], [Discharges Scored])
Avg Predicted Risk   = AVERAGE(gold_risk_scores[risk_score])
Observed Readmit %   = AVERAGE(gold_risk_scores[observed_readmit_30])
Calibration Ratio    = DIVIDE([Avg Predicted Risk], [Observed Readmit %])
High-Tier Precision  = CALCULATE([Observed Readmit %], gold_risk_scores[risk_tier] = "High")
Readmits Captured %  = DIVIDE(
                          CALCULATE(SUM(gold_risk_scores[observed_readmit_30]), gold_risk_scores[risk_tier] = "High"),
                          SUM(gold_risk_scores[observed_readmit_30]))
Latest Run           = MAX(gold_drift_metrics[run_ts])
Latest Score PSI     = CALCULATE(MAX(gold_drift_metrics[psi]),
                          gold_drift_metrics[feature] = "risk_score",
                          gold_drift_metrics[run_ts] = [Latest Run])
Features Drifting    = CALCULATE(COUNTROWS(gold_drift_metrics),
                          gold_drift_metrics[status] IN {"warn", "alert"},
                          gold_drift_metrics[run_ts] = [Latest Run])
```

`observed_readmit_30` exists only because this is historical data. With live data it fills in 30 days later. Until then, the calibration visuals show blank for recent discharges.

## 4. Pages

1. **Worklist (care managers):** slicers for tier, state, age band, sex and discharge date. A table of High and Medium patients sorted by `risk_score` desc, with `reason_1..3` columns so each flag comes with its SHAP explanation. Add conditional formatting on `risk_score`.
2. **Population overview:** KPI cards (Discharges Scored, High Risk %, Avg Predicted Risk). A donut of tier mix. Bars of Avg Predicted Risk by age band and by sex. A filled map by `state_cd` (map the SSA state code to a state name if you want a geographic map).
3. **Model performance:** a line of Avg Predicted Risk vs Observed Readmit % by discharge week. Cards for High-Tier Precision, Readmits Captured % and Calibration Ratio. Use the same visuals split by age band and sex to show the fairness audit in production.
4. **Drift monitor:** a card for Latest Score PSI with conditional colour (green < 0.10, amber 0.10–0.25, red ≥ 0.25). A bar chart of `psi` by `feature` filtered to Latest Run, coloured by `status`. A line of `psi` over `run_ts` for `risk_score` and the top features.

## 5. Refresh

With DirectQuery the report always reflects the last `carerisk-batch-scoring` run (Mondays 06:00 UTC). With Import mode, schedule a dataset refresh in Power BI Service for about 07:00 UTC so it lands after the job.
