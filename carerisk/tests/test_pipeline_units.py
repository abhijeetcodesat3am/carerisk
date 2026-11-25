from datetime import date

import numpy as np
import pandas as pd
import pytest

from carerisk import comorbidity, drift, fairness
from carerisk.features import assign_split, label_and_history
from carerisk.ingest import build_episodes

COHORT = {"lookback_days": 365, "label_window_days": 30, "test_start": "2010-01-01"}


@pytest.fixture(scope="session")
def spark():
    from carerisk.config import get_spark
    return get_spark()


# ---------- PSI
def test_psi_identical_is_zero():
    x = np.random.default_rng(0).normal(size=5000)
    assert drift.psi(x, x) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    assert drift.psi(rng.normal(size=5000), rng.normal(1.0, 1, 5000)) > 0.25


def test_psi_binary_and_missing():
    ref = np.array([0, 1] * 500, dtype=float)
    cur = np.array([1] * 900 + [np.nan] * 100, dtype=float)
    assert drift.psi(ref, cur) > 0.25


# ---------- Charlson
def test_charlson_ranges_expand():
    assert "4256" in comorbidity._expand("4254-4259")
    assert comorbidity._expand("V420") == ["V420"]
    assert comorbidity._expand("042-044") == ["042", "043", "044"]


def test_charlson_score_with_hierarchy(spark):
    codes = spark.createDataFrame(
        [("a", "4280"), ("a", "25040"), ("a", "25000"), ("a", "1970"), ("a", "1749"), ("b", "4019")],
        ["k", "code"])
    cats = comorbidity.categorize_codes(spark, codes)
    res = {r["k"]: r for r in comorbidity.charlson_score(spark, cats, "k").collect()}
    # chf 1 + dm_comp 2 (dm suppressed) + mets 6 (cancer suppressed) = 9
    assert res["a"]["cci"] == 9 and res["a"]["cci_dm"] == 0 and res["a"]["cci_cancer"] == 0
    assert "b" not in res  # hypertension alone is not a Charlson condition


# ---------- episodes + label
def test_transfer_merge_and_label(spark):
    rows = [  # patient p: stay1 + same-day transfer, readmit 10 days later, then a stay 100 days later
        ("p", "1", date(2009, 1, 1), date(2009, 1, 5)),
        ("p", "2", date(2009, 1, 5), date(2009, 1, 8)),
        ("p", "3", date(2009, 1, 18), date(2009, 1, 20)),
        ("p", "4", date(2009, 4, 30), date(2009, 5, 2)),
    ]
    claims = spark.createDataFrame(rows, ["DESYNPUF_ID", "CLM_ID", "admit_dt", "discharge_dt"]) \
        .selectExpr("*", "1.0 as pmt", "array('4280') as dx", "0 as n_proc")
    ep = build_episodes(claims)
    assert ep.count() == 3
    h = label_and_history(ep, COHORT).orderBy("admit_dt").collect()
    assert [r["readmit_30"] for r in h] == [1, 0, 0]
    assert h[0]["discharge_dt"] == date(2009, 1, 8) and h[0]["los"] == 7
    assert [r["prior_admits_365"] for r in h] == [0, 1, 2]
    assert h[1]["days_since_last_discharge"] == 10


def test_split_is_patient_grouped_and_temporal(spark):
    rows = [(f"id{i}", date(2009 + (i % 2), 6, 1)) for i in range(400)]
    df = assign_split(spark.createDataFrame(rows, ["DESYNPUF_ID", "admit_dt"]), COHORT,
                      {"seed": 1, "train_pct": 70, "val_pct": 15}).toPandas()
    test = df[df["split"] == "test"]
    assert (pd.to_datetime(test["admit_dt"]) >= "2010-01-01").all()
    assert (pd.to_datetime(df[df["split"].isin(["train", "val"])]["admit_dt"]) < "2010-01-01").all()
    assert df.dropna().groupby("DESYNPUF_ID")["split"].nunique().max() == 1


# ---------- fairness
def test_fairness_audit_shapes():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.integers(0, 2, 1000), "s": rng.random(1000),
                       "age_band": rng.choice(["<65", "65-74"], 1000), "sex": rng.choice(["F", "M"], 1000)})
    out = fairness.audit(df, "s", "y", 0.9)
    assert set(out["attribute"]) == {"overall", "age_band", "sex"}
    assert out["tpr_ratio"].dropna().max() == pytest.approx(1.0)
