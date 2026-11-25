"""Batch scoring with the champion model (distributed with mapInPandas) + PSI drift monitoring.

Outputs (Delta tables on Databricks, Parquet + CSV locally) that the Power BI dashboard reads:
  gold_risk_scores    latest batch: one row per discharge with score, tier and top SHAP reasons
  gold_drift_metrics  appended each run: PSI per feature + model score vs the training population
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from . import drift, explain, tracking
from .config import on_databricks, reports_dir
from .features import GROUP_COLS, KEYS, LABEL

SCORE_SCHEMA = T.StructType([
    T.StructField("DESYNPUF_ID", T.StringType()),
    T.StructField("episode_id", T.StringType()),
    T.StructField("admit_dt", T.DateType()),
    T.StructField("discharge_dt", T.DateType()),
    T.StructField("age_band", T.StringType()),
    T.StructField("sex", T.StringType()),
    T.StructField("state_cd", T.StringType()),
    T.StructField("risk_score", T.DoubleType()),
    T.StructField("risk_tier", T.StringType()),
    T.StructField("risk_rank_pct", T.DoubleType()),
    T.StructField("reason_1", T.StringType()),
    T.StructField("reason_2", T.StringType()),
    T.StructField("reason_3", T.StringType()),
    T.StructField("reasons_json", T.StringType()),
    T.StructField("observed_readmit_30", T.IntegerType()),
])


def _scorer(model, meta: dict, n_reasons: int):
    feats, hi, med = meta["features"], meta["threshold_high"], meta["threshold_medium"]

    def score_batches(it):
        for pdf in it:
            X = pdf[feats].astype(float)
            p = model.predict_proba(X)[:, 1]
            reasons = explain.reason_codes(explain.shap_values(model, X), X, k=n_reasons)
            out = pdf[KEYS + GROUP_COLS].copy()
            out["risk_score"] = p
            out["risk_tier"] = ["High" if s >= hi else "Medium" if s >= med else "Low" for s in p]
            out["risk_rank_pct"] = float("nan")
            for k in range(3):
                out[f"reason_{k + 1}"] = [explain.format_reason(r[k]) if len(r) > k else None for r in reasons]
            out["reasons_json"] = [json.dumps(r) for r in reasons]
            out["observed_readmit_30"] = pdf[LABEL].astype("Int64")
            yield out[[f.name for f in SCORE_SCHEMA.fields]]

    return score_batches


def run(spark, store, cfg: dict) -> dict:
    model, meta = tracking.load_champion(cfg)
    sc = cfg["scoring"]
    feats = store.read("gold_features")
    batch = feats.where(F.col("discharge_dt").between(sc["batch_start"], sc["batch_end"]))
    scored = batch.select(*KEYS, *GROUP_COLS, *meta["features"], LABEL).mapInPandas(
        _scorer(model, meta, sc["n_reasons"]), SCORE_SCHEMA)
    run_ts = datetime.now(timezone.utc).replace(microsecond=0)
    scored = (scored.withColumn("risk_rank_pct", F.percent_rank().over(Window.orderBy("risk_score")))
        .withColumn("model_name", F.lit(meta["model_name"]))
        .withColumn("model_version", F.lit(meta["model_version"]))
        .withColumn("scored_at", F.lit(run_ts).cast("timestamp")))
    store.write(scored, "gold_risk_scores")
    scores = store.read("gold_risk_scores").toPandas()

    # ---- drift: training population (features + champion score) vs this batch
    ref = feats.where("split = 'train'").select(*meta["features"]).toPandas().astype(float)
    ref["risk_score"] = model.predict_proba(ref[meta["features"]])[:, 1]
    cur = batch.select(*meta["features"]).toPandas().astype(float)
    cur["risk_score"] = model.predict_proba(cur[meta["features"]])[:, 1]
    rep = drift.drift_report(ref, cur, ["risk_score", *meta["features"]], cfg)
    rep.insert(0, "run_ts", pd.Timestamp(run_ts).tz_localize(None))
    rep.insert(1, "model_version", meta["model_version"])
    rep.insert(2, "batch_start", sc["batch_start"]); rep.insert(3, "batch_end", sc["batch_end"])
    rep["n_reference"], rep["n_current"] = len(ref), len(cur)
    mode = "append" if store.exists("gold_drift_metrics") else "overwrite"
    store.write(spark.createDataFrame(rep), "gold_drift_metrics", mode=mode)

    if not on_databricks():  # flat files for Power BI Desktop when running locally
        pbi = reports_dir(cfg) / "powerbi"; pbi.mkdir(exist_ok=True)
        scores.to_csv(pbi / "risk_scores.csv", index=False)
        store.read("gold_drift_metrics").toPandas().to_csv(pbi / "drift_metrics.csv", index=False)

    score_psi = float(rep.loc[rep["feature"] == "risk_score", "psi"].iloc[0])
    alerts = rep.loc[rep["status"] == "alert", "feature"].tolist()
    if alerts and cfg["drift"]["fail_on_alert"]:
        raise RuntimeError(f"PSI drift alert on: {alerts}")
    return {"n_scored": len(scores), "model_version": meta["model_version"],
            "tiers": scores["risk_tier"].value_counts().to_dict(), "score_psi": score_psi,
            "alerts": alerts, "warnings": rep.loc[rep["status"] == "warn", "feature"].tolist(),
            "drift_top": rep.head(8)[["feature", "psi", "status"]]}
